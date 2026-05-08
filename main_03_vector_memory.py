import os
import sys
from dotenv import load_dotenv
from agents.conversation_agent import ConversationAgent
from services.embedding_service import EmbeddingService, EmbeddingConfig
from services.llm_client import LlmClient, LlmConfig
from services.vector_memory_store import VectorMemoryStore

load_dotenv()

llm_config = LlmConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_MODEL_NAME"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE")),
)

emb_config = EmbeddingConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
)


## open/closed if there is a change will be only in the memory embedder and agenet can be used everywhere in the same way
embedder = EmbeddingService(emb_config)
memory = VectorMemoryStore(embedder)
agent = ConversationAgent(LlmClient(llm_config))

with agent:
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("\nBye!")
            sys.exit(0)

        if user_input.lower() == "clear": ## test it
            memory.clear()
            agent.reset()
            print("\nSession memory cleared.\n")
            continue

        if user_input.lower().startswith("search "):
            query = user_input[7:].strip()
            hits = memory.search(query, top_k=3)
            print(f"\n[Semantic search: '{query}'] - {len(hits)} results(s)")
            for i, entry in enumerate(hits, 1):
                print(f"  (i). [{entry.role}] {entry.content[:90]}")
            print(
                f"\n (total stored turns: {len(memory)}   |"
                f"a flat list would return the 3 most RECENT turns instead)\n"
            )
            continue

        response = agent.chat(user_input)

        memory.add("user", user_input)
        memory.add("assistant", response)

        print(f"\nAgent: {response}\n")
