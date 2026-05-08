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
    chunk_size: int = 500
    chunk_overlap: int = 50  #
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )


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
        text = Path(path).read_text(encoding="utf-8")
        base_meta = {"source": Path(path).name, **(metadata or {})}
        chunks = self._splitter.create_documents([text], metadatas=[base_meta])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        return chunks

    def index(self, documents: list[Document]) -> None:
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
