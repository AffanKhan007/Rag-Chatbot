# app/main.py
import csv
import io
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, UploadFile
import requests
from sqlalchemy import text
from app.config import DATABASE_URL, GROQ_API_KEY, GROQ_MODEL
from app.db import SessionLocal
from app.rag import get_embedding

app = FastAPI()


def _chunk_text(text_value: str, chunk_size: int = 250, overlap: int = 25):
    """Character-based chunks with overlap for better retrieval continuity."""
    clean_text = " ".join(text_value.split())
    if not clean_text:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 10)

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(clean_text), step):
        chunk = clean_text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(clean_text):
            break
    return chunks


def _extract_answer(question: str, content: str, max_sentences: int = 3) -> str:
    """Pull only the most relevant sentence(s) from a chunk instead of the full chunk."""
    # Split on sentence boundaries AND on newlines (preserves heading structure)
    parts = re.split(r'(?<=[.!?])\s+|\n+', content.strip())
    sentences = [s.strip() for s in parts if s.strip()]

    if not sentences:
        return content

    # Score each sentence by keyword overlap with the question
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    stop_words = {"what", "is", "are", "the", "a", "an", "of", "in", "and", "or", "how", "why", "when", "who"}
    q_tokens -= stop_words

    scored = []
    for s in sentences:
        s_tokens = set(re.findall(r"[a-z0-9]+", s.lower()))
        score = len(q_tokens & s_tokens)
        scored.append((score, s))

    # Sort by score descending, pick the single best sentence
    scored.sort(key=lambda x: x[0], reverse=True)
    best_scored = [(score, s) for score, s in scored if score > 0]

    if not best_scored:
        # Nothing matched — fall back to first 2 sentences
        return " ".join(sentences[:max_sentences])

    # Take the top-matched sentence and include the next sentence for context
    # (captures formulas/details that follow a definition)
    order = {s: i for i, s in enumerate(sentences)}
    top_sentence = best_scored[0][1]
    top_idx = order.get(top_sentence, 0)

    selected_indices = {top_idx}
    # Include next sentence if it exists (e.g. "Formula: KE = 1/2 mv²")
    if top_idx + 1 < len(sentences):
        selected_indices.add(top_idx + 1)
    # Include one more top match if it's different and scored > 0
    if len(best_scored) > 1:
        second = best_scored[1][1]
        second_idx = order.get(second, 999)
        if second_idx not in selected_indices:
            selected_indices.add(second_idx)

    # Re-order selected sentences by original position for natural reading
    result = [sentences[i] for i in sorted(selected_indices) if i < len(sentences)]
    return " ".join(result)


def _extract_text(filename: str, file_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    try:
        # utf-8-sig strips BOM from Windows Notepad "UTF-8" saves
        decoded = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded text."
        ) from exc

    if suffix in {".txt", ".md"}:
        return decoded

    if suffix == ".csv":
        rows = []
        reader = csv.reader(io.StringIO(decoded))
        for row in reader:
            rows.append(" ".join(cell.strip() for cell in row if cell.strip()))
        return "\n".join(rows)

    if suffix == ".json":
        data = json.loads(decoded)
        return json.dumps(data, indent=2)

    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Use .txt, .md, .csv, or .json"
    )


def _keyword_score(question: str, content: str) -> int:
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    c_tokens = set(re.findall(r"[a-z0-9]+", content.lower()))
    if not q_tokens:
        return 0
    return len(q_tokens.intersection(c_tokens))


def _keyword_search_words(question: str):
    """Words long enough to use in ILIKE fallback (avoids tiny tokens)."""
    return [
        w
        for w in re.findall(r"[a-z0-9]+", question.lower())
        if len(w) >= 3
    ]

def _db_target():
    u = urlparse(DATABASE_URL)
    return {
        "host": u.hostname,
        "port": u.port,
        "database": (u.path or "").lstrip("/") or None,
    }


def _groq_generate_answer(question: str, context: str):
    """Return (answer_text, error_message). If API key missing or request fails, error_message is set."""
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY not configured"

    system_prompt = (
        "You are a grounded RAG assistant. "
        "Answer only using the provided context. "
        "If the answer is not in the context, say: 'I could not find that in the provided file.' "
        "Keep responses concise and readable."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Return only the final answer."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return answer, None
    except Exception as exc:
        return None, str(exc)


@app.get("/")
def home():
    return {
        "message": "RAG API is running",
        "docs": "/docs",
        "endpoints": ["/upload", "/query", "/reset", "/stats"],
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    text_value = _extract_text(file.filename, file_bytes)
    chunks = _chunk_text(text_value)
    if not chunks:
        raise HTTPException(status_code=400, detail="No valid text content found.")

    async with SessionLocal() as db:
        for idx, chunk in enumerate(chunks, start=1):
            emb = await get_embedding(chunk)
            await db.execute(
                text(
                    """
                    INSERT INTO knowledge (topic, content, embedding)
                    VALUES (:topic, :content, CAST(:embedding AS vector))
                    """
                ),
                {
                    "topic": f"{file.filename} | chunk {idx}",
                    "content": chunk,
                    "embedding": str(emb),
                },
            )
        await db.commit()
        total = (
            await db.execute(text("SELECT COUNT(*) FROM knowledge"))
        ).scalar_one()

    return {
        "message": "File ingested successfully.",
        "filename": file.filename,
        "chunks_indexed": len(chunks),
        "knowledge_rows": int(total),
    }


@app.post("/reset")
async def reset_knowledge():
    async with SessionLocal() as db:
        await db.execute(text("TRUNCATE TABLE knowledge RESTART IDENTITY;"))
        await db.commit()
    return {"message": "Knowledge table reset successfully."}


@app.get("/stats")
async def stats():
    """How many chunks are in the DB (upload + seed). Use this if /query says no data."""
    async with SessionLocal() as db:
        total = (
            await db.execute(text("SELECT COUNT(*) FROM knowledge"))
        ).scalar_one()
    payload = {"knowledge_rows": int(total), "database_target": _db_target()}
    return payload


@app.get("/query")
async def query(q: str):
    embedding = await get_embedding(q)

    async with SessionLocal() as db:
        result = await db.execute(
            text("""
                SELECT topic, content
                FROM knowledge
                -- Use cosine distance (<=>) for semantic similarity.
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT 3
            """),
            {"embedding": str(embedding)}
        )
        rows = result.fetchall()

        # If vector search returns nothing but rows exist, use text match (same DB/session).
        if not rows:
            count = (
                await db.execute(text("SELECT COUNT(*) FROM knowledge"))
            ).scalar_one()
            if int(count) == 0:
                return {
                    "message": "No data found",
                    "hint": "Upload a file first, or call POST /reset then upload again. Open /stats to confirm rows > 0.",
                    "knowledge_rows": 0,
                }

            words = _keyword_search_words(q)
            for word in sorted(words, key=len, reverse=True)[:8]:
                r2 = await db.execute(
                    text(
                        """
                        SELECT topic, content
                        FROM knowledge
                        WHERE content ILIKE :pattern
                        LIMIT 5
                        """
                    ),
                    {"pattern": f"%{word}%"},
                )
                rows = r2.fetchall()
                if rows:
                    break

    if rows:
        row = max(rows, key=lambda item: _keyword_score(q, item.content))
        local_answer = _extract_answer(q, row.content)
        groq_answer, groq_error = _groq_generate_answer(q, row.content)
        answer = groq_answer if groq_answer else local_answer
        return {
            "question": q,
            "answer": answer,
            "answer_generator": "groq" if groq_answer else "local_extract",
            "groq_error": groq_error if not groq_answer else None,
            "source_chunk": row.content,   # full chunk for debugging
            "topic": row.topic,
        }

    async with SessionLocal() as db:
        total = (
            await db.execute(text("SELECT COUNT(*) FROM knowledge"))
        ).scalar_one()
    return {
        "message": "No data found",
        "hint": "Table has rows but no chunk matched your question. Try different wording or check /stats.",
        "knowledge_rows": int(total),
    }