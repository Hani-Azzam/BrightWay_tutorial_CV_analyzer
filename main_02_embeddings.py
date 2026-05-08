import os
from dotenv import load_dotenv
from services.embedding_service import EmbeddingService, EmbeddingConfig

load_dotenv()
embedder = EmbeddingService(
    EmbeddingConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
    )
)

#Check the similarity
pairs = [
    ("episodic memory", "long-term memory"),
    ("vector database", "embedding index"),
    ("episodic memory", "pizza recipe"),
    ("RAG pipeline", "retrieval-augmented generation"),
    ("agent loop", "pizza recipe")
]

print("=== tutorial 02 - Embedding Similarity Demo ===\n")
print(f" {'Phrase A':<30} {'Phrase B':<35}")
#continue from the slide

for text_a, text_b in pairs:
    vec_a = embedder.embed(text_a)
    vec_b = embedder.embed(text_b)
    score = embedder.similarity(vec_a, vec_b)
    bar ="#" * int(score * 20)
    print(f" {text_a:<30} {text_b:<35} {score:.3f} {bar}")

print()
print("Observation:")
print(" Scores near 1.0 -> semantically related phrases")
print(" Scores near 0.0 -> semantically unrelated phrases")
print()
print("Each vector has", len(embedder.embed("text")), "dimensions.")