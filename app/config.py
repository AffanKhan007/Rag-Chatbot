# app/config.py
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (not the current shell working directory).
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not DATABASE_URL:
    raise RuntimeError(
        f"DATABASE_URL is not set. Add .env in {_ROOT} with DATABASE_URL=..."
    )