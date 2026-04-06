# app/rag.py
from sentence_transformers import SentenceTransformer
from typing import List

# Load local model automatically (downloads once on first run)
model = SentenceTransformer("all-MiniLM-L6-v2")

async def get_embedding(text: str) -> list:
    """Single embedding for query-time use."""
    return model.encode(text, normalize_embeddings=True).tolist()

async def get_embeddings_batch(texts: List[str], batch_size: int = 64) -> List[list]:
    """Batch embeddings for upload — much faster on large files.
    Processes in batches of 64 to avoid memory spikes.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
    return all_embeddings