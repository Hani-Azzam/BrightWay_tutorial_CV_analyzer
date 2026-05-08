from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from base.agent_base import AgentBase
from services.llm_client import LlmClient


class ConversationAgent(AgentBase):
    _SYSTEM = "You are a helpful AI assistant specialising in agentic AI systems."

    def __init__(self, llm_client: LlmClient) -> None:
        self._llm = llm_client
        self._history: list[BaseMessage] = []
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", self._SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "{question}")
        ])
        self._chain = llm_client.build_chain(self._prompt)

    def chat(self, user_input: str) -> str:
        responce = self._chain.invoke(
            {"history": self._history, "question": user_input}
        )

        self._history.append(HumanMessage(content=user_input))
        self._history.append(AIMessage(content=responce))
        return responce

    def reset(self) -> None:
        self._history.clear()

    def history_text(self) -> str:
        if not self._history:
            return "(no history)"
        lines = []
        for msg in self._history:
            role = "You" if isinstance(msg, HumanMessage) else "Agent"
            lines.append(f"[{role}] {msg.content[:100]}") #take first 100 from 0 index to index 99
        return "\n".join(lines)
