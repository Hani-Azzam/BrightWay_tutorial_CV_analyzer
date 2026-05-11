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
from services.vector_memory_store import VectorMemoryService


# Metadata card describing a document's provenance and sensitivity.
# Used to attach PII risk level, licensing info, and refresh schedule
# to each corpus file before it is indexed.
@dataclass
class DataCard:
    source: str
    license: str
    pii_risk: str           # 'none' | 'low' | 'high'
    refresh_cadence: str    # e.g. 'on exam update', 'quarterly', 'n/a'


# Strips personally identifiable information from free-text before it
# is stored in vector memory or forwarded to the LLM.
# Replaces e-mail addresses and US-style phone numbers with safe tokens.
def _redact_pii(text: str) -> str:
    # Replace email addresses with [EMAIL]
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[EMAIL]", text)
    # Replace phone numbers (e.g. 555-123-4567) with [PHONE]
    text = re.sub(r"\b05\d[-.\\s]?\d{3}[-.\\s]?\d{4}\b", "[PHONE]", text)
    return text


class RagAgent(AgentBase):

    def __init__(
        self,
        llm_client: LlmClient,
        vector_memory: VectorMemoryService,
        rag_pipeline: RagPipeline,
        document_store: DocumentStore,
        data_cards: list[DataCard] | None = None,
    ) -> None:
        # Wire up all injected services and initialise the audit trail list
        self._llm = llm_client
        self._memory = vector_memory
        self._rag = rag_pipeline
        self._store = document_store
        self.data_cards = data_cards or []
        self._audit: list[dict] = []

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def index_corpus(self, directory: str) -> None:
        """Load every .txt file in *directory*, attach DataCard metadata
        where available, and index the documents into the document store."""
        docs = []
        for path in Path(directory).glob("*.txt"):
            # Look up the DataCard whose source name matches this file
            card = next((c for c in self.data_cards if c.source == path.name), None)
            # Only carry pii_risk and license into the vector index
            meta = {"pii_risk": card.pii_risk, "license": card.license} if card else {}
            docs.extend(self._store.load_file(str(path), metadata=meta))
        self._store.index(docs)

    # ------------------------------------------------------------------
    # Main conversational entry-point
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        # 1. Redact PII from the raw user input before any processing
        clean = _redact_pii(user_input)

        # 2. Retrieve semantically relevant past turns from vector memory
        #    (top-2) and format them as a compact context string
        past = self._memory.search(clean, top_k=2)
        mem_context = " | ".join(f"{e.role}: {e.content[:60]}" for e in past)

        # 3. Get the RAG answer together with the source documents used
        rag_answer, sources = self._rag.answer_with_sources(clean)

        # 4. Contradiction detection — if there is prior memory and the
        #    RAG answer is not an "I don't have" fallback, check whether
        #    the new answer contradicts what the agent previously said
        if past and not rag_answer.startswith("I don't have"):
            if self._detect_contradiction(mem_context, rag_answer):
                rag_answer += (
                    "\n\nNote: this may differ from a past session — "
                    "the document source should be trusted for factual claims."
                )

        # 5. Persist both turns in vector memory so future calls can
        #    retrieve them for context
        self._memory.add("user", clean[:500])   # store only the first 500 chars to save space (embedding model has a token limit)
        self._memory.add("assistant", rag_answer[:500])

        # 6. Append an audit trail entry (MCP-AAI: data retention)
        self._audit.append({
            "timestamp": datetime.now().isoformat(),
            "user": clean,
            "response": rag_answer,
            "sources": [r.document.metadata.get("source") for r in sources],
        })

        return rag_answer

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Persist the audit log and wipe the in-memory conversation."""
        self._save_audit()
        self._memory.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_contradiction(self, memory_context: str, rag_answer: str) -> bool:
        """Ask the LLM whether the new RAG answer directly contradicts
        the summarised memory context. Returns True when it does."""
        prompt = (
            f"Memory says:\n{memory_context}\n\n"
            f"Document says:\n{rag_answer}\n\n"
            "Do these two statements directly contradict each other? "
            "Reply with YES or NO only."
        )
        result = self._llm.invoke([HumanMessage(content=prompt)])
        return result.strip().upper().startswith("YES")

    def _save_audit(self) -> None:
        """Write the accumulated audit entries to data/audit_log.json."""
        if self._audit:
            Path("data").mkdir(exist_ok=True)
            Path("data/audit_log.json").write_text(
                json.dumps(self._audit, indent=2), encoding="utf-8"
            )
            print(f"\nAudit log saved -> data/audit_log.json ({len(self._audit)} entries)")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Ensure cleanup runs even if the session ends with an exception
        self.reset()
        self._store.clear()
