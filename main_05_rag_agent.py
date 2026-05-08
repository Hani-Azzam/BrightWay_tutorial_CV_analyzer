import os
import sys

from dotenv import load_dotenv

from agents.rag_agent import RagAgent, DataCard
from services.document_store import ChunkConfig, DocumentStore, PineconeConfig
from services.embedding_service import EmbeddingConfig, EmbeddingService
from services.llm_client import LlmClient, LlmConfig
from services.rag_pipeline import RagConfig, RagPipeline
from services.vector_memory_store import VectorMemoryStore

# Load environment variables from .env file
load_dotenv(override=True)

# --- LLM client (Gemini) ---
llm_client = LlmClient(
    LlmConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
        # Default 0.7 guards against missing env var at runtime
        temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
    )
)

# --- Embedding service (also uses the Gemini API key) ---
embedder = EmbeddingService(
    EmbeddingConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        # env var name matches GEMINI_EMBEDDING_MODEL_NAME in .env
        model_name=os.getenv("GEMINI_EMBEDDING_MODEL"),
    )
)

# In-process vector store that holds the conversation history
vector_memory = VectorMemoryStore(embedder)

# --- Pinecone document store (corpus index) ---
pc_config = PineconeConfig(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    namespace=os.getenv("PINECONE_NAMESPACE"),
)

# cleanup_on_exit=False keeps the Pinecone index alive between runs
store    = DocumentStore(embedder, pc_config, ChunkConfig(), cleanup_on_exit=False)
# refuse_threshold=0.60: answers with relevance below 60% are refused
pipeline = RagPipeline(llm_client, store, RagConfig(refuse_threshold=0.60))

# --- DataCards — metadata for every corpus document ---
DATA_CARDS = [
    DataCard("ncp_aai_blueprint.txt",     license="public-NVIDIA",   pii_risk="none", refresh_cadence="on exam update"),
    DataCard("rag_concepts.txt",           license="public-tutorial", pii_risk="none", refresh_cadence="manual"),
    DataCard("memory_systems.txt",         license="public-tutorial", pii_risk="none", refresh_cadence="manual"),
    DataCard("llm_safety.txt",             license="public-tutorial", pii_risk="none", refresh_cadence="manual"),
    DataCard("dirty_document_example.txt", license="example",         pii_risk="low",  refresh_cadence="n/a"),
]

# Instantiate the agent with all services and the data-card catalogue
agent = RagAgent(llm_client, vector_memory, pipeline, store, DATA_CARDS)

# --- Banner and data-card summary ---
print("=== tutorial_05 - Full RAG Agent ===\n")
print("Data cards:")
for card in DATA_CARDS:
    print(f"  {card.source:<40} license={card.license:<18} pii_risk={card.pii_risk}")

# Index every .txt file in data/corpus using the DataCard metadata
print("\nIndexing corpus...")
agent.index_corpus("data/corpus")
print("Done.\n")

print("Commands: 'memory <query>' - search vector memory | 'exit' - quit")
print("Tip: ask the same question twice across sessions to trigger contradiction check.\n")

# --- Main REPL loop — runs inside a context manager so __exit__ fires on quit ---
with agent:
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl-C or piped input exhausted — clean exit
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            sys.exit(0)

        # Special command: inspect vector memory directly
        if user_input.lower().startswith("memory "):
            query = user_input[7:].strip()
            hits  = vector_memory.search(query, top_k=3)
            print(f"\n[Memory search: '{query}']")
            for e in hits:
                # Show role tag and first 90 chars of each hit
                print(f"  [{e.role}] {e.content[:90]}")
            print()
            continue

        # Normal turn — send to agent and print the answer
        print(f"\nAgent: {agent.chat(user_input)}\n")
