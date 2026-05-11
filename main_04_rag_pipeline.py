import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from services.document_store import ChunkConfig, DocumentStore, PineconeConfig
from services.embedding_service import EmbeddingService, EmbeddingConfig
from services.llm_client import LlmClient, LlmConfig
from services.rag_pipeline import RagConfig, RagPipeline

load_dotenv()

llm_client = LlmClient(
    LlmConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name=os.getenv("GEMINI_MODEL_NAME"),
        temperature=float(os.getenv("GEMINI_TEMPERATURE")),
    )
)
embedder = EmbeddingService(
    EmbeddingConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name=os.getenv("GEMINI_EMBEDDING_MODEL"),
    )
)
pc_config = PineconeConfig(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    namespace=os.getenv("PINECONE_NAMESPACE"),
)
store = DocumentStore(embedder, pc_config, ChunkConfig(chunk_size=400, chunk_overlap=40), cleanup_on_exit=True)
rag = RagPipeline(llm_client, store, RagConfig(refuse_threshold=0.60))

CORPUS = "data/corpus"
print(f"=== Indexing corpus from {CORPUS}/ ===")
docs = []
for path in Path(CORPUS).glob("*.txt"):
    file_docs = store.load_file(str(path))
    docs.extend(file_docs)
    print(f"   {path.name:<40} {len(file_docs)} chunk(s)")
store.index(docs)
print(f" Total chunks indexed: {len(docs)}\n")

print("=== Tutorial 04 - RAG Pipeline ===")
print("Try: 'What are the NCP-AAI blueprint weights?'")
print("Try: 'What is chunk overlap and why does it matter?'")
print("Try: 'What is MMR?' (observe source cited)")
print("Try: 'What is the best pizza recipe?' (should refuse)")
print("Commands: 'mmr' - toggle MMR | 'exit' - quit\n")

with store:
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("Bye")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye")
            sys.exit(0)
        if user_input.lower() == "mmr":
            rag.config.use_mmr = not rag.config.use_mmr
            print(f"[MMR retrieval: {'ON - diverse results' if rag.config.use_mmr else 'OFF - similarity only'}]\n")
            continue

        answer, sources = rag.answer_with_sources(user_input)
        print(f"\nAgent: {answer}")
        if sources:
            seen: set[str] = set()
            for r in sources:
                src = r.document.metadata.get("source", "?")
                if src not in seen:
                    score_str = f"source:{r.score:.2f}" if r.score < 1.0 else "MMR"
                    print(f"   > Source: {src}   ({score_str})")
                    seen.add(src)
        print()
