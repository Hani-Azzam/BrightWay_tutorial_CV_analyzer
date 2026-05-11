from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import embeddings
from pinecone import Pinecone, ServerlessSpec
from base.retriever_base import RetrieverBase, RetrievalResult

from services.embedding_service import EmbeddingService

# Dimension of Google gemini-embedding-001 output vectors.
_EMBEDDING_DIMENSION = 3072


@dataclass
class ChunkConfig:
    chunk_size: int = 500       # max characters per chunk
    chunk_overlap: int = 50     # number of shared characters between neighbouring chunks
    """ Controls how the text splitter cuts your file up. The separators are tried in order — it prefers to split on double newlines first (paragraph breaks), 
    then single newlines, then sentences, then words, then characters as a last resort."""
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]   
    )

""" PineconeConfig(api_key=..., index_name=..., namespace="tutorial_04")
Controls where in Pinecone the vectors go. The index is the whole database. The namespace is a labelled section inside it — like a folder. 
Two DocumentStore instances pointing at the same index but different namespaces are completely isolated from each other. 
This is exactly how HiringAgent will isolate candidates."""
@dataclass
class PineconeConfig:
    api_key: str
    index_name: str
    namespace: str = "tutorial_04"
    cloud: str = "aws"
    region: str = "us-east-1"


class DocumentStore(RetrieverBase):
    def __init__(
            self,
            embedding_service: EmbeddingService,
            pinecone_config: PineconeConfig,
            chunk_config: ChunkConfig = ChunkConfig(),
            cleanup_on_exit: bool = False,
    ) -> None:
        if not pinecone_config.api_key:
            raise ValueError("PineconeConfig.api_key is required and cannot be empty.")
        if not pinecone_config.index_name:
            raise ValueError("PineconeConfig.index_name is required and cannot be empty.")
        if not pinecone_config.namespace:
            raise ValueError("PineconeConfig.namespace is required and cannot be empty.")

        self._embedder = embedding_service
        self._pc_cfg = pinecone_config
        self._cleanup_on_exit = cleanup_on_exit
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_config.chunk_size,
            chunk_overlap=chunk_config.chunk_overlap,
            separators=chunk_config.separators,
        )
        self._store: PineconeVectorStore | None = None
        # Connect to Pinecone and create the index if it doesn't exist yet.

        pc = Pinecone(api_key=pinecone_config.api_key)
        existing = [idx.name for idx in pc.list_indexes()]
        if pinecone_config.index_name not in existing:
            pc.create_index(
                name=pinecone_config.index_name,
                dimension=_EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=pinecone_config.cloud,
                    region=pinecone_config.region,
                ),
            )
        self._pc_index = pc.Index(pinecone_config.index_name)

    def load_file(self, path: str, metadata: dict | None = None) -> list[Document]:
        """ Load a text file, split it into chunks, and return a list of Documents with metadata.
        Args:
            path (str): The file path to the text document to be loaded.
            metadata (dict, optional): Additional metadata to attach to each chunk. Defaults to None.   
        Returns:
            list[Document]: A list of Document objects representing the chunks of the text file.
        """
        text = Path(path).read_text(encoding="utf-8")
        base_meta = {"source": Path(path).name, **(metadata or {})}
        chunks = self._splitter.create_documents([text], metadatas=[base_meta])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        return chunks

    def index(self, documents: list[Document]) -> None:
        """ Index a list of Documents into Pinecone by embedding them and storing the vectors.
        Args:
            documents (list[Document]): A list of Document objects to be indexed. 
            Each Document should have page_content and metadata.
        """
        self._store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self._embedder.get_model(),
            index_name=self._pc_cfg.index_name,
            pinecone_api_key=self._pc_cfg.api_key,
            namespace=self._pc_cfg.namespace,
        )

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        if self._store is None:
            return []
        raw = self._store.similarity_search_with_score(query, k=top_k)
        return [RetrievalResult(document=doc, score=score) for doc, score in raw]

    def retrieve_mmr(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        if self._store is None:
            return []
        docs = self._store.max_marginal_relevance_search(
            query, k=top_k, fetch_k=top_k * 4,
            namespace=self._pc_cfg.namespace,
        )
        return [RetrievalResult(document=doc, score=1.0) for doc in docs]

    def clear(self) -> None:
        self._pc_index.delete(delete_all=True, namespace=self._pc_cfg.namespace)
        self._store = None

    def __enter__(self) -> "DocumentStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._cleanup_on_exit:
            self.clear()
            print("clean up done")



""" 
 ---                                                                                                                                         
  What DocumentStore Does                                                                                                                     
                                                                                                                                              
  It has one job: take a text file, cut it into chunks, turn those chunks into vectors, and store them in Pinecone — then retrieve relevant   
  chunks later when you ask a question.                                                                                                     

  ---
  The Three Configs

  ChunkConfig(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""])
  Controls how text gets cut up. chunk_overlap=50 means each chunk shares 50 characters with the next — so a sentence split across a boundary
  isn't lost.

  PineconeConfig(api_key=..., index_name=..., namespace="tutorial_04")
  Controls where in Pinecone the vectors go. The index is the whole database. The namespace is a labelled section inside it — like a folder.
  Two DocumentStore instances pointing at the same index but different namespaces are completely isolated from each other. This is exactly how
   HiringAgent will isolate candidates.

  ---
  The Four Methods

  load_file(path) — lines 75–81
  Reads the .txt file, runs it through the splitter, returns a list of Document objects. Each document has page_content (the chunk text) and
  metadata (source filename, chunk index, any extras you pass in like pii_risk). Nothing goes to Pinecone yet.

  index(documents) — lines 83–90
  Takes those Document objects and sends them to Pinecone. This is where embedding happens — each chunk gets converted to a vector and stored.
   After this call, the data lives in Pinecone under the configured namespace.

  retrieve(query) — lines 92–96
  Converts the query to a vector, does a cosine similarity search in Pinecone, returns the top-k most similar chunks with their scores. This
  is what RagPipeline calls when you chat().

  clear() — lines 107–109
  self._pc_index.delete(delete_all=True, namespace=self._pc_cfg.namespace)
  Deletes only the namespace — not the whole Pinecone index. So if you have candidate_alice and candidate_bob as namespaces, calling clear()
  on Alice's store doesn't touch Bob's data.

  ---
  The Full Flow Visualised

  cv_alice.txt
       │
       │  load_file()
       ▼
  [Document(chunk1), Document(chunk2), ...]   ← in memory, not yet in Pinecone
       │
       │  index()
       ▼
  Pinecone index: "x-engineer"
    namespace: "candidate_alice"
      vector_1 → "Alice has 5 years Python experience..."
      vector_2 → "Led a team of 4 engineers at Acme Corp..."
       │
       │  retrieve("does she know Python?")
       ▼
  [RetrievalResult(document=chunk1, score=0.91), ...]   ← back to RagPipeline

  ---
  The Key Thing for Your Assignment

  The namespace parameter is the lever. For Phase 2 you'll do:

  # Alice gets her own isolated corner of Pinecone
  PineconeConfig(..., namespace="candidate_alice")

  # Bob gets his own — completely separate, same index, zero interference
  PineconeConfig(..., namespace="candidate_bob")

  On exit, call clear() on each store — it deletes only that namespace, leaving the Pinecone index itself intact for future runs.

"""
