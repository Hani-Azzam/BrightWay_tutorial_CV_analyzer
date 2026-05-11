# ============================================================
# COMMANDS
# ============================================================
# summary              — structured summary (name, level, top skills)
# skills               — two lists: technical and soft skills
# improve <SECTION>    — improvement suggestions for a named section
# gap                  — top 3 gaps vs the job description (Bonus A)
# memory <query>       — search vector memory directly
# exit                 — save audit log and quit
# ============================================================

import os
import sys
from dotenv import load_dotenv

from agents.cv_analyser_agent import CvAnalyserAgent
from agents.rag_agent import DataCard
from services.document_store import ChunkConfig, DocumentStore, PineconeConfig
from services.embedding_service import EmbeddingConfig, EmbeddingService
from services.llm_client import LlmClient, LlmConfig
from services.rag_pipeline import RagConfig, RagPipeline
from services.vector_memory_store import VectorMemoryService

# Load environment variables from .env file
load_dotenv(override=True)

# --- LLM client (Gemini) ---
llm_client = LlmClient(
    LlmConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
        temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
    )
)

# --- Embedding service (also uses the Gemini API key) ---
embedder = EmbeddingService(
    EmbeddingConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name=os.getenv("GEMINI_EMBEDDING_MODEL_NAME"),
    )
)

# In-process vector store that holds the conversation history
vector_memory = VectorMemoryService(embedder)

# --- Pinecone document store (corpus index) ---
pc_config = PineconeConfig(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    namespace="cv_phase1",
)

chunck_config = ChunkConfig(                                                                                                                                           
    chunk_size=200,
    chunk_overlap=20,
    separators=[
        "\n--------",    # splits on the dashed dividers in the CV
        "\n\n",          # paragraph breaks
        "\n",            # single line breaks
        " ",             # words
        ""               # characters (last resort)
    ]
)

# cleanup_on_exit=False keeps the Pinecone index alive between runs
store    = DocumentStore(embedder, pc_config, chunk_config=chunck_config, cleanup_on_exit=True)
# refuse_threshold=0.60: answers with relevance below 60% are refused
pipeline = RagPipeline(llm_client, store, RagConfig(top_k=6, refuse_threshold=0.60))

CV_PATH = "data/cv_alice.txt"

cv_data_card = DataCard(
    source="cv_alice.txt",
    license="private",
    pii_risk="high",
    refresh_cadence="on candidate update",
)

# Instantiate the agent with all services and the data-card catalogue
agent = CvAnalyserAgent(llm_client, vector_memory, pipeline, store)


print("=== Phase 1 — CV Analyser Agent ===\n")
print(f"DataCard: {cv_data_card.source} | pii_risk={cv_data_card.pii_risk}\n")

print("Indexing CV...")
agent.index_cv(CV_PATH, cv_data_card)
print("Done.\n")


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
            
        if user_input.lower() == "summary":
            print(f"\nAgent:\n{agent.summarise_cv()}\n")
        
        elif user_input.lower() == "skills":
            print(f"\nAgent:\n{agent.extract_skills()}\n")

        elif user_input.lower().startswith("improve "):
            section = user_input[8:].strip()
            print(f"\nAgent:\n{agent.suggest_improvements(section)}\n")
            
        elif user_input.lower() == "gap":
            print(f"\nAgent:\n{agent.gap_analysis('data/job_description.txt')}\n")
            
        # Special command: inspect vector memory directly
        elif user_input.lower().startswith("memory "):
            query = user_input[7:].strip()
            hits  = vector_memory.search(query, top_k=3)
            print(f"\n[Memory search: '{query}']")
            for e in hits:
                # Show role tag and first 90 chars of each hit
                print(f"  [{e.role}] {e.content[:90]}")
            print()
            continue

        # Normal turn — send to agent and print the answer
        else:
            print(f"\nAgent: {agent.chat(user_input)}\n")
