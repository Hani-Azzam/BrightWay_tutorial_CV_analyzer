from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage
from base.retriever_base import RetrievalResult
from services.document_store import DocumentStore
from services.llm_client import LlmClient


@dataclass
class RagConfig:
    top_k: int = 4  # how many chunks to retrieve from Pinecone
    refuse_threshold: float = 0.30  # minimum similarity score to answer
    use_mmr: bool = False   # retrieval strategy - standard similarity search vs Maximal Marginal Relevance (MMR) 
    """ 
        MMR vs regular retrieval

        use_mmr: bool = False
        - Regular (False) — returns the top_k chunks most similar to the question. Can return near-duplicate chunks.
        - MMR (True) — Maximum Marginal Relevance. Returns chunks that are relevant but also diverse from each other. Better when the CV has
        repetitive content.
    """


class RagPipeline:
    def __init__(self, llm_client: LlmClient, document_store: DocumentStore, config: RagConfig = RagConfig()):
        self._llm = llm_client
        self._store = document_store
        self.config = config

    def answer(self, question: str) -> str:
        # calls answer_with_sources but throws away the source documents and returns only the text string.
        text, _ = self.answer_with_sources(question)
        return text

    def _should_refuse(self, results: list[RetrievalResult]) -> bool:
        """   
        If no chunk scored above refuse_threshold, the pipeline refuses to answer rather than making something up. 
        This is the hallucination guard.
        In main_05 it's set to 0.60 — if the best chunk is less than 60% similar to the question, it refuses.
        """
        if not results:
            return True
        if self.config.use_mmr:
            return False
        return max(r.score for r in results) < self.config.refuse_threshold

    def answer_with_sources(self, question: str) -> tuple[str, list[RetrievalResult]]:
        """ Stage 1 — Retrieve chunks """
        # Goes to Pinecone, finds the top_k most similar chunks to your question, returns them with similarity scores (0.0–1.0).
        results = (
            self._store.retrieve_mmr(question, self.config.top_k)
            if self.config.use_mmr
            else self._store.retrieve(question, self.config.top_k)
        )

        """ Stage 2 — Should we refuse? """
        # If no retrieved chunk is similar enough to the question, refuse to answer rather than risk hallucination.
        if self._should_refuse(results):
            return (
                "I don't have reliable information on that topic in my knowledge base.",
                [],
            )

        """ Stage 3 — Build the context string """
        # Formats the retrieved chunks into one block of text. Each chunk is labelled with its source filename.
        """ 
            context = "\n\n".join(
                f"[{source}]\n{chunk_text}"
                for r in results
            )
        """
        context = "\n\n".join(
            f"[{r.document.metadata.get('source', '?')}]\n{r.document.page_content}"
            for r in results
        )
        
        """ Stage 4 — Ask the LLM """
        # Sends the context + question to the LLM. The system message forces it to only use what's in the context — not its training data.
        messages = [
            SystemMessage(
                content=(
                    "Answer using ONLY the provided context. "
                    "Cite the source filename at the end of your answer"
                )
            ),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        return self._llm.invoke(messages), results
