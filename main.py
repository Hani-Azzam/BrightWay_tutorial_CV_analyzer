import os
import sys

from dotenv import load_dotenv

from agents.conversation_agent import ConversationAgent
from services.llm_client import LlmClient, LlmConfig

load_dotenv()

config = LlmConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_MODEL_NAME"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE")),
)

with ConversationAgent(LlmClient(config)) as agent:
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() == ("exit", "quit"):
            print("Goodbye!")
            sys.exit(0)
        if user_input.lower() == "reset":
            agent.reset()
            print("[history cleared]\n")
            continue
        if user_input.lower() == "history":
            print(agent.history_text() + '\n')
            continue

        print(f"\nAgent: {agent.chat(user_input)}\n")
