import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from base.agent_base import AgentBase
from services.document_store import DocumentStore
from services.llm_client import LlmClient
from services.rag_pipeline import RagPipeline
from services.vector_memory_store import VectorMemoryStore


@dataclass
class DataCard:
    source: str
    license: str
    pii_risk: str           # 'none' / 'low' / 'high'
    refresh_cadence: str    # e.g. 'on exam update', 'quarterly', 'n/a'


def _redact_pii(text:str) -> str:
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.\w+\b", "[EMAIL]", text)
    text = re.sub(r"\b05\d[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
    return text


class RagAgent(AgentBase):

    def __init__(
        self,
        llm_client: LlmClient,
        vector_memory: VectorMemoryStore,
        rag_pipeline: RagPipeline,
        document_store: DocumentStore,
        data_cards: list[DataCard] | None = None,
    ) -> None:
        self._llm = llm_client
        self._memory = vector_memory
        self._rag = rag_pipeline
        self._store = document_store
        self.data_cards = data_cards or []
        self._audit: list[dict] = []


    def index_corpus(self, directory: str) -> None:
        docs = []
        for path in Path(directory).glob("*.txt"):
            card = next((c for c in self.data_cards if c.source == path.name), None)
            meta = {"pii_risk": card.pii_risk, "license": card.license} if card else {}
            docs.extend(self._store.load_file(str(path), metadata=meta))
        self._store.index(docs)


    def chat(self, user_input: str) -> str:
        clean = _redact_pii(user_input)

        # 1. Retrieve semantically relevant post turns
        past = self._memory.search(clean, top_k=2)
        mem_context = " | ".join(f"{e.role}: {e.content[:60]}" for e in past)

        # 2. RAG answer
        rag_answer, sources = self._rag.answer_with_sources(clean)

        # 3. Contradiction detection
        if past and not rag_answer.startswith("I don't have"):
            if self._detect_contradiction(mem_context, rag_answer):
                rag_answer += (
                    "\n\n Note: this may differ from a past session - "
                    "the document source should be trusted for factual claims."
                )

        # 4. Store in vector memory
        self._memory.add("user", clean)
        self._memory.add("assistant", rag_answer)

        # 5. Audit trail (NCP-AAI: data retention)
        self._audit.append({
            "timestamp": datetime.now().isoformat(),
            "user": clean,
            "response": rag_answer,
            "sources": [r.document.metadata.get("sources") for r in sources],
        })
        return rag_answer


    def reset(self) -> None:
        self._save_audit()
        self._memory.clear()


    def _detect_contradiction(self, memory_context: str, rag_answer: str) -> bool:
        prompt = (
            f"Memory says:\n{memory_context}\n\n"
            f"Document says:\n{rag_answer}\n\n"
            "Do these two statements directly contradict each other? "
            "Reply with YES or NO only."
        )
        result = self._llm.invoke([HumanMessage(content=prompt)])
        return result.strip().upper().startswith("YES")


    def _save_audit(self) -> None:
        if self._audit:
            Path("data").mkdir(exist_ok=True)
            Path("data/audit_log.json").write_text(
                json.dumps(self._audit, indent=2), encoding="utf-8"
            )
            print(f"\n[Audit log saved → data/audit_log.json ({len(self._audit)} entries)]")


    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.reset()
        self._store.clear()





### copied from tutor Rasheed
#
# def _redact_pii(text: str) -> str:
#
#     """
#     Concept: PII Redaction Before Storage — Knowledge Integration and Data Handling (10%)
#     Removes personally identifiable information from text before it is
#     stored in vector memory or sent to the LLM.
#     Redacting at the INPUT boundary (not after storage) ensures PII never
#     enters the vector index or the audit log.
#     This is a regex stub — it handles email addresses and Israeli mobile
#     phone numbers (05X-XXX-XXXX format).
#     Production systems use dedicated NLP classifiers (spaCy, Microsoft
#     Presidio) that detect names, addresses, ID numbers, and more.
#     """
#
#     text = re.sub(r"\b[\w.+-]+@[\w-]+\.\w+\b", "[EMAIL]", text)
#     text = re.sub(r"\b05\d[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
#     return text
#
#
# clean = _redact_pii(user_input)
#
#         # 1. Retrieve semantically relevant past turns
#         past = self._memory.search(clean, top_k=2)
#         mem_context = " | ".join(f"{e.role}: {e.content[:60]}" for e in past)
#
#         # 2. RAG answer
#         rag_answer, sources = self._rag.answer_with_sources(clean)
#
#         # 3. Contradiction detection
#         if past and not rag_answer.startswith("I don't have"):
#             if self._detect_contradiction(mem_context, rag_answer):
#                 rag_answer += (
#                     "\n\n  Note: this may differ from a past session — "
#                     "the document source should be trusted for factual claims."
#                 )
#
#         # 4. Store in vector memory
#         self._memory.add("user", clean)
#         self._memory.add("assistant", rag_answer)
#
#         # 5. Audit trail (NCP-AAI: data retention)
#         self._audit.append({
#             "timestamp": datetime.now().isoformat(),
#             "user": clean,
#             "response": rag_answer,
#             "sources": [r.document.metadata.get("source") for r in sources],
#
#         })
#         return rag_answer
#
#
#     def _detect_contradiction(self, memory_context: str, rag_answer: str) -> bool:
#         prompt = (
#             f"Memory says:\n{memory_context}\n\n"
#             f"Document says:\n{rag_answer}\n\n"
#             "Do these two statements directly contradict each other? "
#             "Reply with YES or NO only."
#         )
#         result = self._llm.invoke([HumanMessage(content=prompt)])
#         return result.strip().upper().startswith("YES")
#