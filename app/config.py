import os
from pathlib import Path

from dotenv import load_dotenv


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    return int(raw_value) if raw_value is not None else default


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
ENABLE_RERANK = _env_bool("ENABLE_RERANK", False)
ENABLE_GROQ_GENERATION = _env_bool("ENABLE_GROQ_GENERATION", False)
USE_HNSW_FOR_LARGE_DATA = _env_bool("USE_HNSW_FOR_LARGE_DATA", True)
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHUNK_SIZE = _env_int("CHUNK_SIZE", 900)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 120)
VECTOR_TOP_K = _env_int("VECTOR_TOP_K", 6)
KEYWORD_TOP_K = _env_int("KEYWORD_TOP_K", 6)
FINAL_CONTEXT_K = _env_int("FINAL_CONTEXT_K", 4)
RERANK_TOP_K = _env_int("RERANK_TOP_K", 4)
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 96)
DEFAULT_WORKSPACE_NAME = os.getenv("DEFAULT_WORKSPACE_NAME", "default")
HNSW_CHUNK_THRESHOLD = _env_int("HNSW_CHUNK_THRESHOLD", 10000)
HNSW_EF_SEARCH = _env_int("HNSW_EF_SEARCH", 80)

if not DATABASE_URL:
    raise RuntimeError(
        f"DATABASE_URL is not set. Add .env in {ROOT} with DATABASE_URL=..."
    )
