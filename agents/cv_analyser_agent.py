import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from base.agent_base import AgentBase
from services.document_store import DocumentStore, ChunkConfig
from services.llm_client import LlmClient
from services.rag_pipeline import RagPipeline
from services.vector_memory_store import VectorMemoryStore
from agents.rag_agent import RagAgent, DataCard


# Index a single CV file into Pinecone (with appropriate chunking for short, structured sections)

# Answer free-form questions about the CV through the standard chat loop
# Suggest improvements for a named section (e.g. SUMMARY, WORK EXPERIENCE)

# [command] Produce a structured summary of the candidate on demand
# [command] Extract technical and soft skills as separate lists
class CVAnalyserAgent(RagAgent):
    def __init__(
            self,
            llm_client: LlmClient,
            vector_memory: VectorMemoryStore,
            rag_pipeline: RagPipeline,
            document_store: DocumentStore,
            data_cards: list[DataCard] | None = None,
    ):
        super().__init__(llm_client, vector_memory, rag_pipeline, document_store, data_cards)


    # def __init__(self, config: ChunkConfig, agent: RagAgent):
    #     super().__init__(config, agent)
    #     config.separators = field(
    #     default_factory=lambda: ["\n\n", "\n", ". ", " ", "", "**CONTACT**", "**ABOUT**", "**EDUCATION**", "**JOB EXPERIENCE**",
    #                              "**SKILLS**", "**PROJECTS**"]
    # )


    # Index a single CV file into Pinecone (with appropriate chunking for short, structured sections)
    def index_single_cv(self, cv_file_path: str):
        path = cv_file_path
        card = next((c for c in self.data_cards if c.source == path.name), None)
        meta = {"pii_risk": card.pii_risk, "license": card.license} if card else {}
        chunks = self._store.load_file(str(path), metadata=meta)
        self._store.index(chunks)



