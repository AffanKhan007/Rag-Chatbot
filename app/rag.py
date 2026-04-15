import asyncio
from typing import Iterable

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME, ENABLE_RERANK, RERANK_MODEL_NAME


embedding_model: SentenceTransformer | None = None
rerank_model: CrossEncoder | None = None
_model_lock = asyncio.Lock()


async def ensure_models_loaded() -> None:
    global embedding_model, rerank_model

    if embedding_model is not None and (not ENABLE_RERANK or rerank_model is not None):
        return

    async with _model_lock:
        if embedding_model is None:
            embedding_model = await asyncio.to_thread(
                lambda: SentenceTransformer(EMBEDDING_MODEL_NAME)
            )
        if ENABLE_RERANK and rerank_model is None:
            rerank_model = await asyncio.to_thread(
                lambda: CrossEncoder(RERANK_MODEL_NAME)
            )


async def get_embedding(text: str) -> list[float]:
    await ensure_models_loaded()
    assert embedding_model is not None
    return await asyncio.to_thread(
        lambda: embedding_model.encode(text, normalize_embeddings=True).tolist()
    )


async def get_embeddings_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    await ensure_models_loaded()
    assert embedding_model is not None
    all_embeddings: list[list[float]] = []

    for index in range(0, len(texts), batch_size):
        batch = texts[index:index + batch_size]
        batch_embeddings = await asyncio.to_thread(
            lambda current_batch=batch: embedding_model.encode(
                current_batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
        )
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


async def rerank_candidates(question: str, chunk_texts: Iterable[str]) -> list[float]:
    texts = list(chunk_texts)
    if not texts:
        return []

    await ensure_models_loaded()
    if rerank_model is None:
        return [0.0 for _ in texts]

    pairs = [[question, chunk_text] for chunk_text in texts]
    scores = await asyncio.to_thread(
        lambda: rerank_model.predict(pairs, batch_size=16, show_progress_bar=False).tolist()
    )
    return [float(score) for score in scores]
