import asyncio
import io
import json
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import fitz
import requests
from docx import Document as DocxDocument
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, insert, select, text

from app.config import (
    ALLOWED_ORIGINS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATABASE_URL,
    DEFAULT_WORKSPACE_NAME,
    EMBEDDING_BATCH_SIZE,
    ENABLE_GROQ_GENERATION,
    ENABLE_RERANK,
    FINAL_CONTEXT_K,
    GROQ_API_KEY,
    GROQ_MODEL,
    HNSW_CHUNK_THRESHOLD,
    HNSW_EF_SEARCH,
    KEYWORD_TOP_K,
    RERANK_TOP_K,
    SERVICE_API_KEY,
    USE_HNSW_FOR_LARGE_DATA,
    VECTOR_TOP_K,
)
from app.db import SessionLocal, engine
from app.models import Base, Chunk, DocumentRecord, Workspace
from app.rag import ensure_models_loaded, get_embedding, get_embeddings_batch, rerank_candidates
from app.schemas import AskDocsResponse, QueryRequest


@dataclass
class RetrievedChunk:
    chunk_id: int
    chunk_index: int | None
    document_id: int
    workspace_id: int
    filename: str
    content: str
    vector_score: float | None = None
    keyword_score: float | None = None
    merged_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class RetrievalMode:
    use_hnsw: bool
    chunk_count: int
    mode_label: str


default_workspace_id_cache: int | None = None
logger = logging.getLogger("rag_backend")

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _log_event(event: str, **fields: object) -> None:
    logger.info("%s | %s", event, json.dumps(fields, default=str, ensure_ascii=True))


async def initialize_database() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                ON chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            )
        )
        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS chunks_tsv_gin_idx
                ON chunks
                USING GIN (tsv);
                """
            )
        )


async def ensure_default_workspace() -> Workspace:
    async with SessionLocal() as db:
        result = await db.execute(
            select(Workspace).where(Workspace.name == DEFAULT_WORKSPACE_NAME)
        )
        workspace = result.scalar_one_or_none()
        if workspace is None:
            workspace = Workspace(name=DEFAULT_WORKSPACE_NAME)
            db.add(workspace)
            await db.commit()
            await db.refresh(workspace)
        return workspace


async def get_default_workspace_id() -> int:
    global default_workspace_id_cache
    if default_workspace_id_cache is not None:
        return default_workspace_id_cache

    workspace = await ensure_default_workspace()
    default_workspace_id_cache = workspace.id
    return default_workspace_id_cache


@asynccontextmanager
async def lifespan(_: FastAPI):
    await initialize_database()
    await ensure_default_workspace()
    await ensure_models_loaded()
    yield


app = FastAPI(
    title="Hybrid RAG API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _db_target() -> dict[str, str | int | None]:
    parsed = urlparse(DATABASE_URL)
    return {
        "host": parsed.hostname,
        "port": parsed.port,
        "database": (parsed.path or "").lstrip("/") or None,
    }


def _service_mode_label() -> str:
    if ENABLE_RERANK and ENABLE_GROQ_GENERATION:
        return "ask_docs_rerank_groq"
    if ENABLE_RERANK:
        return "ask_docs_rerank_local"
    if ENABLE_GROQ_GENERATION:
        return "ask_docs_groq"
    return "ask_docs_local"


async def verify_service_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not SERVICE_API_KEY:
        return
    if x_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def _normalize_document_ids(document_ids: list[int] | None) -> list[int] | None:
    if not document_ids:
        return None
    normalized = sorted({int(document_id) for document_id in document_ids if int(document_id) > 0})
    return normalized or None


def _request_metadata(request: Request, x_client_app: str | None) -> dict[str, object]:
    return {
        "client_app": x_client_app or request.headers.get("x-client-source") or "unknown",
        "client_host": request.client.host if request.client else None,
        "origin": request.headers.get("origin"),
        "user_agent": request.headers.get("user-agent"),
        "method": request.method,
        "path": request.url.path,
    }


async def get_documents_for_scope(
    workspace_id: int,
    document_ids: list[int] | None,
) -> list[DocumentRecord]:
    async with SessionLocal() as db:
        query = (
            select(DocumentRecord)
            .where(DocumentRecord.workspace_id == workspace_id)
            .order_by(DocumentRecord.created_at.desc(), DocumentRecord.id.desc())
        )
        if document_ids:
            query = query.where(DocumentRecord.id.in_(document_ids))
        result = await db.execute(query)
        documents = result.scalars().all()

    if document_ids:
        found_ids = {document.id for document in documents}
        missing_ids = [document_id for document_id in document_ids if document_id not in found_ids]
        if missing_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Document IDs not found in the available scope: {missing_ids}",
            )

    return documents


def _document_debug_items(documents: list[DocumentRecord]) -> list[dict[str, object]]:
    return [
        {
            "document_id": document.id,
            "filename": document.filename,
            "file_type": document.file_type,
            "chunk_count": document.chunk_count,
            "created_at": document.created_at.isoformat() if document.created_at else None,
        }
        for document in documents
    ]


def _chunk_debug_items(chunks: list[RetrievedChunk], include_content: bool = True) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for chunk in chunks:
        item = {
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "vector_score": chunk.vector_score,
            "keyword_score": chunk.keyword_score,
            "merged_score": chunk.merged_score,
            "rerank_score": chunk.rerank_score,
        }
        if include_content:
            item["content"] = chunk.content
        items.append(item)
    return items


def _normalize_text(raw_text: str) -> str:
    text_value = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text_value = re.sub(r"\n{3,}", "\n\n", text_value)
    text_value = re.sub(r"[ \t]+", " ", text_value)
    return text_value.strip()


def _find_split_point(text_value: str, start: int, target_end: int) -> int:
    if target_end >= len(text_value):
        return len(text_value)

    search_start = min(len(text_value), start + max(300, CHUNK_SIZE // 2))
    if search_start >= target_end:
        search_start = start

    paragraph_break = text_value.rfind("\n\n", search_start, target_end)
    if paragraph_break != -1:
        return paragraph_break

    sentence_breaks = [
        text_value.rfind(marker, search_start, target_end)
        for marker in (". ", "? ", "! ", "\n")
    ]
    sentence_break = max(sentence_breaks)
    if sentence_break != -1:
        return sentence_break + 1

    whitespace_break = text_value.rfind(" ", search_start, target_end)
    if whitespace_break != -1:
        return whitespace_break

    return target_end


def chunk_text_paragraph_aware(
    text_value: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    normalized = _normalize_text(text_value)
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        target_end = min(text_length, start + chunk_size)
        split_at = _find_split_point(normalized, start, target_end)
        if split_at <= start:
            split_at = target_end

        chunk = normalized[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        if split_at >= text_length:
            break

        next_start = max(0, split_at - overlap)
        while next_start < text_length and normalized[next_start].isspace():
            next_start += 1
        if next_start <= start:
            next_start = min(text_length, start + max(1, chunk_size - overlap))
        start = next_start

    return chunks


def extract_text_from_upload(filename: str, file_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        try:
            text_value = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("TXT files must be UTF-8 encoded.") from exc
        normalized = _normalize_text(text_value)
        if not normalized:
            raise ValueError("The uploaded TXT file did not contain any text.")
        return normalized

    if suffix == ".pdf":
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf_document:
                pages = [page.get_text("text") for page in pdf_document]
        except Exception as exc:
            raise ValueError("The PDF could not be parsed.") from exc

        extracted = _normalize_text("\n\n".join(pages))
        if not extracted:
            raise ValueError(
                "No extractable text was found in the PDF. Scanned PDFs and OCR are not supported."
            )
        return extracted

    if suffix == ".docx":
        try:
            document = DocxDocument(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ValueError("The DOCX file could not be parsed.") from exc

        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        extracted = _normalize_text("\n\n".join(paragraphs))
        if not extracted:
            raise ValueError("The DOCX file did not contain any extractable paragraph text.")
        return extracted

    raise ValueError("Unsupported file type. Only .pdf, .txt, and .docx are supported.")


def _tokenize(text_value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text_value.lower())
        if len(token) > 2
    }


def _first_alpha(text_value: str) -> str | None:
    for char in text_value:
        if char.isalpha():
            return char
    return None


def _starts_mid_sentence(text_value: str) -> bool:
    first_alpha = _first_alpha(text_value.lstrip())
    return bool(first_alpha and first_alpha.islower())


def _ends_like_complete_sentence(text_value: str) -> bool:
    return bool(re.search(r'[.!?]["\')\]]?$', text_value.strip()))


def _clean_fallback_excerpt(text_value: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text_value)
    for index, part in enumerate(parts):
        sentence = part.strip()
        if not sentence:
            continue
        if index == 0 and _starts_mid_sentence(sentence):
            continue
        if _ends_like_complete_sentence(sentence):
            return sentence
    return text_value[:260].strip()


def _sentence_candidates(chunks: list[RetrievedChunk]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for chunk in chunks:
        parts = re.split(r"(?<=[.!?])\s+|\n+", chunk.content)
        total_parts = len(parts)
        for index, part in enumerate(parts):
            sentence = part.strip()
            if sentence:
                if index == 0 and _starts_mid_sentence(sentence):
                    continue
                if index == total_parts - 1 and total_parts > 1 and not _ends_like_complete_sentence(sentence):
                    continue
                candidates.append((chunk.filename, sentence))
    return candidates


def build_local_grounded_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
    filenames = sorted({chunk.filename for chunk in chunks})
    if not chunks:
        return "not found in the uploaded documents", filenames

    question_tokens = _tokenize(question)
    scored_sentences: list[tuple[int, str, str]] = []
    for filename, sentence in _sentence_candidates(chunks):
        score = len(question_tokens.intersection(_tokenize(sentence)))
        scored_sentences.append((score, filename, sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    best_sentences = [sentence for score, _, sentence in scored_sentences if score > 0][:3]

    if not best_sentences:
        best_sentences = [_clean_fallback_excerpt(chunk.content) for chunk in chunks[:2]]

    answer = " ".join(best_sentences).strip()
    if not answer:
        answer = "not found in the uploaded documents"

    return answer, filenames


def merge_candidates(
    vector_hits: Iterable[RetrievedChunk],
    keyword_hits: Iterable[RetrievedChunk],
) -> list[RetrievedChunk]:
    merged: dict[int, RetrievedChunk] = {}
    fusion_constant = 40.0

    for rank, hit in enumerate(vector_hits, start=1):
        existing = merged.get(hit.chunk_id)
        if existing is None:
            existing = hit
            merged[hit.chunk_id] = existing
        existing.vector_score = hit.vector_score
        existing.merged_score += 1.0 / (fusion_constant + rank)

    for rank, hit in enumerate(keyword_hits, start=1):
        existing = merged.get(hit.chunk_id)
        if existing is None:
            existing = hit
            merged[hit.chunk_id] = existing
        existing.keyword_score = hit.keyword_score
        existing.merged_score += 1.0 / (fusion_constant + rank)

    return sorted(
        merged.values(),
        key=lambda item: (
            item.merged_score,
            item.vector_score if item.vector_score is not None else -1.0,
            item.keyword_score if item.keyword_score is not None else -1.0,
        ),
        reverse=True,
    )


async def get_total_chunk_count(workspace_id: int, document_ids: list[int] | None = None) -> int:
    async with SessionLocal() as db:
        query = select(func.count()).select_from(Chunk).where(Chunk.workspace_id == workspace_id)
        if document_ids:
            query = query.where(Chunk.document_id.in_(document_ids))
        chunk_count = await db.scalar(query)
    return int(chunk_count or 0)


def choose_vector_retrieval_mode(chunk_count: int) -> RetrievalMode:
    use_hnsw = USE_HNSW_FOR_LARGE_DATA and chunk_count >= HNSW_CHUNK_THRESHOLD
    return RetrievalMode(
        use_hnsw=use_hnsw,
        chunk_count=chunk_count,
        mode_label="hnsw_cosine" if use_hnsw else "exact_cosine",
    )


async def vector_retrieval(
    workspace_id: int,
    question_embedding: list[float],
    limit: int,
    document_ids: list[int] | None = None,
) -> tuple[list[RetrievedChunk], RetrievalMode]:
    chunk_count = await get_total_chunk_count(workspace_id, document_ids)
    retrieval_mode = choose_vector_retrieval_mode(chunk_count)

    async with SessionLocal() as db:
        if retrieval_mode.use_hnsw:
            await db.execute(text("SET LOCAL hnsw.ef_search = :ef_search"), {"ef_search": HNSW_EF_SEARCH})
        else:
            await db.execute(text("SET LOCAL enable_indexscan = off"))
            await db.execute(text("SET LOCAL enable_bitmapscan = off"))
            await db.execute(text("SET LOCAL enable_indexonlyscan = off"))

        query_text = """
            SELECT
                id AS chunk_id,
                chunk_index,
                document_id,
                workspace_id,
                filename,
                content,
                1 - (embedding <=> CAST(:embedding AS vector)) AS vector_score
            FROM chunks
            WHERE workspace_id = :workspace_id
        """
        params: dict[str, object] = {
            "workspace_id": workspace_id,
            "embedding": str(question_embedding),
            "limit": limit,
        }
        if document_ids:
            query_text += " AND document_id = ANY(CAST(:document_ids AS int[]))"
            params["document_ids"] = document_ids
        query_text += """
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """
        result = await db.execute(text(query_text), params)
        rows = result.mappings().all()

    return ([
        RetrievedChunk(
            chunk_id=row["chunk_id"],
            chunk_index=row["chunk_index"],
            document_id=row["document_id"],
            workspace_id=row["workspace_id"],
            filename=row["filename"],
            content=row["content"],
            vector_score=float(row["vector_score"]) if row["vector_score"] is not None else None,
        )
        for row in rows
    ], retrieval_mode)


async def keyword_retrieval(
    workspace_id: int,
    question: str,
    limit: int,
    document_ids: list[int] | None = None,
) -> list[RetrievedChunk]:
    async with SessionLocal() as db:
        query_text = """
            WITH query AS (
                SELECT websearch_to_tsquery('english', :question) AS q
            )
            SELECT
                c.id AS chunk_id,
                c.chunk_index,
                c.document_id,
                c.workspace_id,
                c.filename,
                c.content,
                ts_rank_cd(c.tsv, query.q) AS keyword_score
            FROM chunks c, query
            WHERE c.workspace_id = :workspace_id
              AND query.q @@ c.tsv
        """
        params: dict[str, object] = {
            "workspace_id": workspace_id,
            "question": question,
            "limit": limit,
        }
        if document_ids:
            query_text += " AND c.document_id = ANY(CAST(:document_ids AS int[]))"
            params["document_ids"] = document_ids
        query_text += """
            ORDER BY keyword_score DESC, c.id ASC
            LIMIT :limit
        """
        result = await db.execute(text(query_text), params)
        rows = result.mappings().all()

    return [
        RetrievedChunk(
            chunk_id=row["chunk_id"],
            chunk_index=row["chunk_index"],
            document_id=row["document_id"],
            workspace_id=row["workspace_id"],
            filename=row["filename"],
            content=row["content"],
            keyword_score=float(row["keyword_score"]) if row["keyword_score"] is not None else None,
        )
        for row in rows
    ]


async def build_grounded_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str], str | None]:
    if not ENABLE_GROQ_GENERATION or not GROQ_API_KEY:
        answer, filenames = build_local_grounded_answer(question, chunks)
        reason = None if ENABLE_GROQ_GENERATION else "Groq generation disabled for faster local answers"
        if ENABLE_GROQ_GENERATION and not GROQ_API_KEY:
            reason = "GROQ_API_KEY not configured"
        return answer, filenames, reason

    filenames = sorted({chunk.filename for chunk in chunks})
    if not chunks:
        return "not found in the uploaded documents", filenames, None

    context = "\n\n".join(
        [
            f"[Chunk {index}]"
            f"\nFilename: {chunk.filename}"
            f"\nContent:\n{chunk.content}"
            for index, chunk in enumerate(chunks, start=1)
        ]
    )

    system_prompt = (
        "You are a grounded retrieval assistant. "
        "Answer only from the provided context. "
        "Combine evidence from multiple chunks when needed. "
        "If the answer is not supported by the context, reply exactly: not found in the uploaded documents. "
        "Do not include filenames or citations in the answer body."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer the question using only the context."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    def _call_groq() -> tuple[str | None, str | None]:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip(), None
        except Exception as exc:
            return None, str(exc)

    answer, error = await asyncio.to_thread(_call_groq)
    if answer:
        return answer, filenames, None

    fallback, filenames = build_local_grounded_answer(question, chunks)
    return fallback, filenames, error


@app.get("/")
def home():
    return {
        "message": "Hybrid RAG API is running",
        "docs": "/docs",
        "endpoints": ["/upload", "/documents", "/query", "/ask-docs", "/health", "/ready", "/stats", "/reset"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag_chatbot"}


@app.get("/ready")
async def ready():
    workspace_id = await get_default_workspace_id()
    return {"status": "ready", "default_workspace_id": workspace_id}


@app.post("/upload")
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    x_client_app: str | None = Header(default=None),
    _: None = Depends(verify_service_api_key),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    workspace_id = await get_default_workspace_id()
    successes: list[dict[str, int | str]] = []
    errors: list[dict[str, str]] = []
    request_meta = _request_metadata(request, x_client_app)

    _log_event(
        "upload_request_received",
        request=request_meta,
        file_names=[upload.filename or "unnamed" for upload in files],
        workspace_id=workspace_id,
    )

    async with SessionLocal() as db:
        for upload in files:
            filename = upload.filename or "unnamed"
            try:
                file_bytes = await upload.read()
                if not file_bytes:
                    raise ValueError("The uploaded file is empty.")

                extracted_text = extract_text_from_upload(filename, file_bytes)
                chunks = chunk_text_paragraph_aware(extracted_text)
                if not chunks:
                    raise ValueError("No text chunks could be created from the uploaded file.")

                document = DocumentRecord(
                    workspace_id=workspace_id,
                    filename=filename,
                    file_type=Path(filename).suffix.lower(),
                    content_text=extracted_text,
                    chunk_count=len(chunks),
                )
                db.add(document)
                await db.flush()

                embeddings = await get_embeddings_batch(chunks, batch_size=EMBEDDING_BATCH_SIZE)
                chunk_rows = [
                    {
                        "workspace_id": workspace_id,
                        "document_id": document.id,
                        "filename": filename,
                        "chunk_index": index,
                        "content": chunk_text,
                        "embedding": embedding,
                    }
                    for index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings), start=1)
                ]
                await db.execute(insert(Chunk), chunk_rows)
                successes.append(
                    {
                        "filename": filename,
                        "document_id": document.id,
                        "stored": True,
                        "text_extracted_chars": len(extracted_text),
                        "chunks_created": len(chunks),
                        "embeddings_created": len(embeddings),
                        "chunks_indexed": len(chunks),
                        "vector_indexed": True,
                        "full_text_indexed": True,
                    }
                )
            except ValueError as exc:
                await db.rollback()
                errors.append({"filename": filename, "error": str(exc)})
            except Exception as exc:
                await db.rollback()
                errors.append({"filename": filename, "error": f"Unexpected ingestion error: {exc}"})
            else:
                await db.commit()

    async with SessionLocal() as db:
        document_count = await db.scalar(select(func.count()).select_from(DocumentRecord))
        chunk_count = await db.scalar(select(func.count()).select_from(Chunk))

    _log_event(
        "upload_request_completed",
        request=request_meta,
        uploaded_count=len(successes),
        failed_count=len(errors),
        document_count=int(document_count or 0),
        chunk_count=int(chunk_count or 0),
        processed_files=successes,
        errors=errors,
    )

    return {
        "uploaded_count": len(successes),
        "failed_count": len(errors),
        "processed_files": successes,
        "document_ids": [
            int(item["document_id"])
            for item in successes
            if "document_id" in item
        ],
        "documents": [
            {
                "document_id": int(item["document_id"]),
                "filename": str(item["filename"]),
                "chunks_indexed": int(item["chunks_indexed"]),
            }
            for item in successes
            if "document_id" in item
        ],
        "errors": errors,
        "document_count": int(document_count or 0),
        "chunk_count": int(chunk_count or 0),
    }


@app.get("/documents")
async def list_documents():
    async with SessionLocal() as db:
        result = await db.execute(
            select(DocumentRecord).order_by(DocumentRecord.created_at.desc(), DocumentRecord.id.desc())
        )
        documents = result.scalars().all()

    return {
        "documents": [
            {
                "id": document.id,
                "filename": document.filename,
                "file_type": document.file_type,
                "chunk_count": document.chunk_count,
                "created_at": document.created_at.isoformat(),
            }
            for document in documents
        ],
    }


@app.post("/reset")
async def reset_knowledge(_: None = Depends(verify_service_api_key)):
    async with SessionLocal() as db:
        await db.execute(text("TRUNCATE TABLE chunks, documents RESTART IDENTITY CASCADE;"))
        await db.commit()
    return {"message": "Knowledge reset successfully."}


@app.get("/stats")
async def stats():
    async with SessionLocal() as db:
        document_count = await db.scalar(select(func.count()).select_from(DocumentRecord))
        chunk_count = await db.scalar(select(func.count()).select_from(Chunk))

    return {
        "document_count": int(document_count or 0),
        "chunk_count": int(chunk_count or 0),
        "database_target": _db_target(),
        "enable_rerank": ENABLE_RERANK,
        "enable_groq_generation": ENABLE_GROQ_GENERATION,
        "use_hnsw_for_large_data": USE_HNSW_FOR_LARGE_DATA,
        "hnsw_chunk_threshold": HNSW_CHUNK_THRESHOLD,
    }


async def execute_query(
    payload: QueryRequest,
    request: Request,
    x_client_app: str | None = None,
) -> dict[str, object]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    workspace_id = await get_default_workspace_id()
    normalized_document_ids = _normalize_document_ids(payload.document_ids)
    scope_documents = await get_documents_for_scope(workspace_id, normalized_document_ids)
    scope_document_ids = [document.id for document in scope_documents]
    request_meta = _request_metadata(request, x_client_app)

    _log_event(
        "query_request_received",
        request=request_meta,
        question=question,
        requested_document_ids=normalized_document_ids,
        searched_document_ids=scope_document_ids,
        searched_filenames=[document.filename for document in scope_documents],
        debug=payload.debug,
    )

    question_embedding, keyword_hits = await asyncio.gather(
        get_embedding(question),
        keyword_retrieval(
            workspace_id,
            question,
            payload.keyword_top_k or KEYWORD_TOP_K,
            scope_document_ids if normalized_document_ids else None,
        ),
    )
    vector_hits, retrieval_mode = await vector_retrieval(
        workspace_id,
        question_embedding,
        payload.vector_top_k or VECTOR_TOP_K,
        scope_document_ids if normalized_document_ids else None,
    )
    merged_hits = merge_candidates(vector_hits, keyword_hits)

    if ENABLE_RERANK and merged_hits:
        reranked_scores = await rerank_candidates(question, [item.content for item in merged_hits])
        for candidate, score in zip(merged_hits, reranked_scores):
            candidate.rerank_score = float(score)
        merged_hits = sorted(
            merged_hits,
            key=lambda item: (
                item.rerank_score if item.rerank_score is not None else -1.0,
                item.merged_score,
            ),
            reverse=True,
        )

    final_limit = payload.final_top_k or (RERANK_TOP_K if ENABLE_RERANK else FINAL_CONTEXT_K)
    final_chunks = merged_hits[:final_limit]
    answer, filenames, groq_error = await build_grounded_answer(question, final_chunks)
    debug_payload = {
        "request": {
            "question": question,
            "requested_document_ids": normalized_document_ids or [],
        },
        "client": request_meta,
        "searched_documents": _document_debug_items(scope_documents),
        "candidate_chunks": {
            "vector": _chunk_debug_items(vector_hits, include_content=False),
            "keyword": _chunk_debug_items(keyword_hits, include_content=False),
            "merged": _chunk_debug_items(merged_hits, include_content=False),
            "selected": _chunk_debug_items(final_chunks, include_content=True),
        },
        "retrieved_context": [chunk.content for chunk in final_chunks],
    }

    _log_event(
        "query_request_completed",
        request=request_meta,
        question=question,
        answer=answer,
        searched_document_ids=scope_document_ids,
        selected_chunk_ids=[chunk.chunk_id for chunk in final_chunks],
        selected_documents=sorted({chunk.document_id for chunk in final_chunks}),
        filenames=filenames,
        vector_hits=len(vector_hits),
        keyword_hits=len(keyword_hits),
        merged_hits=len(merged_hits),
        groq_error=groq_error,
    )

    return {
        "question": question,
        "answer": answer,
        "filenames": filenames,
        "retrieval": {
            "vector_mode": retrieval_mode.mode_label,
            "chunk_count": retrieval_mode.chunk_count,
            "vector_hits": len(vector_hits),
            "keyword_hits": len(keyword_hits),
            "merged_hits": len(merged_hits),
            "hnsw_threshold": HNSW_CHUNK_THRESHOLD,
            "requested_document_ids": normalized_document_ids or [],
            "searched_document_ids": scope_document_ids,
            "searched_document_count": len(scope_documents),
        },
        "sources": [
            {
                "chunk_id": item.chunk_id,
                "chunk_index": item.chunk_index,
                "document_id": item.document_id,
                "filename": item.filename,
                "content": item.content,
                "vector_score": item.vector_score,
                "keyword_score": item.keyword_score,
                "rerank_score": item.rerank_score,
            }
            for item in final_chunks
        ],
        "groq_error": groq_error,
        "debug": debug_payload if payload.debug else None,
    }


@app.post("/query")
async def query(
    request: Request,
    payload: QueryRequest,
    x_client_app: str | None = Header(default=None),
    _: None = Depends(verify_service_api_key),
):
    return await execute_query(payload, request, x_client_app)


@app.post("/ask-docs", response_model=AskDocsResponse)
async def ask_docs(
    request: Request,
    payload: QueryRequest,
    x_client_app: str | None = Header(default=None),
    _: None = Depends(verify_service_api_key),
):
    result = await execute_query(payload, request, x_client_app)
    return AskDocsResponse(
        question=result["question"],
        answer=result["answer"],
        filenames=result["filenames"],
        found_in_documents=result["answer"].strip().lower() != "not found in the uploaded documents",
        mode=_service_mode_label(),
        groq_error=result["groq_error"],
        debug=result["debug"],
    )
