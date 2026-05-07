# Hybrid RAG Backend

[![Status](https://img.shields.io/badge/status-active%20development-2563eb)](https://github.com/)
[![Backend](https://img.shields.io/badge/backend-FastAPI%20%2B%20PostgreSQL%20%2B%20pgvector-0f172a)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/frontend-Streamlit-111827)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/database-PostgreSQL%2016%20%2B%20pgvector-0ea5e9)](https://www.postgresql.org/)
[![Embeddings](https://img.shields.io/badge/embeddings-all--MiniLM--L6--v2-7c3aed)](https://www.sbert.net/)
[![RAG](https://img.shields.io/badge/RAG-hybrid%20retrieval-9333ea)](#architecture)
[![Python](https://img.shields.io/badge/made%20with-Python-3776ab)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-not%20specified-64748b)](#summary)

A production-style hybrid Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **PostgreSQL + pgvector**, and **Streamlit**. The system ingests uploaded documents, extracts and chunks their text, stores vector embeddings and full-text indexes in PostgreSQL, and answers questions using a hybrid retrieval pipeline.

This project also exposes a clean HTTP API so it can be consumed by another application such as **EchoPanel**.

## Tech Tags

`FastAPI` `PostgreSQL` `pgvector` `Streamlit` `Sentence-Transformers` `Groq` `PyMuPDF` `python-docx` `Hybrid Retrieval` `Vector Search` `Full-Text Search` `RAG` `HNSW` `CrossEncoder` `Docker` `ngrok`

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [How the System Works](#how-the-system-works)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Configuration](#configuration)
8. [API Endpoints](#api-endpoints)
9. [EchoPanel Integration](#echopanel-integration)
10. [Run Locally](#run-locally)
11. [Expose with ngrok](#expose-with-ngrok)
12. [Current Limitations](#current-limitations)

## Overview

This backend is designed to:

- accept `.pdf`, `.txt`, and `.docx` uploads
- automatically extract and normalize text
- split documents into retrieval-friendly chunks
- generate embeddings using `all-MiniLM-L6-v2`
- store chunks and embeddings in PostgreSQL
- build PostgreSQL full-text indexes for keyword retrieval
- answer questions using hybrid retrieval:
  - vector search
  - PostgreSQL full-text search
  - merge and dedupe
  - optional reranking
  - final grounded answer generation

The backend supports both:

- a local Streamlit testing UI
- API-based integration for external apps such as EchoPanel

## Key Features

- Multi-file upload support
- Automatic ingestion on upload
- Persistent storage of old and new documents
- Hybrid retrieval: semantic + keyword search
- Optional reranking with a local CrossEncoder
- Optional answer generation via Groq
- Exact cosine retrieval for smaller corpora
- HNSW-based vector retrieval for larger corpora
- Debug-friendly API responses
- EchoPanel-ready `/ask-docs` endpoint
- Optional API-key protection using `X-API-Key`
- Optional CORS configuration for frontend integration

## Architecture

```mermaid
flowchart TD
    A["User / EchoPanel / Streamlit"] --> B["FastAPI Backend"]
    B --> C["Upload Pipeline"]
    C --> C1["Extract Text"]
    C1 --> C2["Chunk Text"]
    C2 --> C3["Generate Embeddings"]
    C3 --> D["PostgreSQL + pgvector"]

    A --> E["Ask Question"]
    E --> B
    B --> F["Embed Query"]
    B --> G["Full-Text Search"]
    F --> H["Vector Search"]
    G --> I["Merge + Dedupe"]
    H --> I
    I --> J["Optional Rerank"]
    J --> K["Grounded Answer Builder / Groq"]
    K --> A
```

## How the System Works

### 1. Document Upload

When a file is uploaded through `POST /upload`, the backend automatically:

1. stores the file contents as a document record
2. extracts raw text
3. normalizes whitespace and formatting
4. splits the text into chunks
5. generates embeddings for each chunk
6. inserts chunk rows into PostgreSQL
7. enables both:
   - vector retrieval from `embedding`
   - full-text retrieval from `tsv`

There is no separate indexing call required. Upload and indexing happen in a single ingestion flow.

### 2. Chunking

The project uses paragraph-aware character-based chunking.

Default values:

- `CHUNK_SIZE=900`
- `CHUNK_OVERLAP=120`

Meaning:

- each chunk is roughly up to 900 characters
- the next chunk repeats about 120 characters from the previous one
- the splitter prefers paragraph and sentence boundaries when possible

This helps preserve context while keeping retrieval fast and targeted.

### 3. Embeddings

Embeddings are generated using:

- `all-MiniLM-L6-v2`

The model is loaded once at startup and reused across requests.

Why embeddings are used:

- they convert text into vectors
- semantically similar text ends up close in vector space
- this allows meaning-based retrieval, even when query wording differs from document wording

### 4. Hybrid Retrieval

When a query arrives:

1. the question is embedded
2. vector search runs over stored chunk embeddings
3. PostgreSQL full-text search runs over `tsv`
4. both result sets are merged
5. duplicate chunks are removed
6. optional reranking can reorder merged hits
7. the final top chunks are passed to the answer generator

### 5. Answer Generation

Two answer paths exist:

- local grounded answer builder
- Groq answer generation

If Groq is enabled:

- the final retrieved chunks and question are sent to Groq
- the prompt forces grounded, context-only answers

If Groq is disabled:

- the backend uses its local grounded answer builder for a faster response

## Tech Stack

### Backend

- FastAPI
- Uvicorn
- SQLAlchemy Async
- asyncpg

### Database

- PostgreSQL 16
- pgvector
- PostgreSQL Full-Text Search

### AI / Retrieval

- sentence-transformers
- `all-MiniLM-L6-v2`
- CrossEncoder reranker
- Groq API

### File Processing

- PyMuPDF for PDF extraction
- python-docx for DOCX extraction

### UI

- Streamlit

### Infra / Local Dev

- Docker Compose
- ngrok

## Project Structure

```text
rag_chatbot/
  app/
    config.py
    db.py
    main.py
    models.py
    rag.py
    schemas.py
  docker-compose.yml
  requirements.txt
  setup_db.py
  seed.py
  streamlit_app.py
  .env.example
  README.md
```

### Important Files

- [app/main.py](/C:/Users/affan.khan/Desktop/rag_chatbot/app/main.py): API routes, upload flow, retrieval pipeline, answer generation
- [app/rag.py](/C:/Users/affan.khan/Desktop/rag_chatbot/app/rag.py): embedding model and optional reranker
- [app/models.py](/C:/Users/affan.khan/Desktop/rag_chatbot/app/models.py): PostgreSQL schema
- [app/config.py](/C:/Users/affan.khan/Desktop/rag_chatbot/app/config.py): environment variables and runtime settings
- [streamlit_app.py](/C:/Users/affan.khan/Desktop/rag_chatbot/streamlit_app.py): local testing UI
- [docker-compose.yml](/C:/Users/affan.khan/Desktop/rag_chatbot/docker-compose.yml): PostgreSQL + pgvector service

## Configuration

Copy `.env.example` to `.env` and adjust values as needed.

### Example

```ini
DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5433/rag_chatbot
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
SERVICE_API_KEY=
ALLOWED_ORIGINS=*
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
ENABLE_RERANK=false
ENABLE_GROQ_GENERATION=false
USE_HNSW_FOR_LARGE_DATA=true
RERANK_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
CHUNK_SIZE=900
CHUNK_OVERLAP=120
VECTOR_TOP_K=6
KEYWORD_TOP_K=6
FINAL_CONTEXT_K=4
RERANK_TOP_K=4
EMBEDDING_BATCH_SIZE=96
HNSW_CHUNK_THRESHOLD=10000
HNSW_EF_SEARCH=80
```

### Key Settings

- `ENABLE_GROQ_GENERATION`
  - enables Groq-based answer generation
- `ENABLE_RERANK`
  - enables CrossEncoder reranking
- `SERVICE_API_KEY`
  - optional shared secret for API clients
- `ALLOWED_ORIGINS`
  - CORS origin list
- `USE_HNSW_FOR_LARGE_DATA`
  - allows switching to HNSW when dataset grows
- `HNSW_CHUNK_THRESHOLD`
  - chunk-count threshold for HNSW

## API Endpoints

### Health / Service

- `GET /`
- `GET /health`
- `GET /ready`

### Upload / Storage

- `POST /upload`
- `GET /documents`
- `POST /reset`
- `GET /stats`

### Query

- `POST /query`
- `POST /ask-docs`

### Upload Response

`POST /upload` now returns EchoPanel-friendly ID fields.

Example:

```json
{
  "uploaded_count": 1,
  "failed_count": 0,
  "processed_files": [
    {
      "filename": "lahore.txt",
      "document_id": 24,
      "stored": true,
      "text_extracted_chars": 12345,
      "chunks_created": 58,
      "embeddings_created": 58,
      "chunks_indexed": 58,
      "vector_indexed": true,
      "full_text_indexed": true
    }
  ],
  "document_ids": [24],
  "documents": [
    {
      "document_id": 24,
      "filename": "lahore.txt",
      "chunks_indexed": 58
    }
  ],
  "errors": [],
  "document_count": 24,
  "chunk_count": 2174
}
```

### Query Request

Example:

```json
{
  "question": "What is Lahore's position in Pakistan in terms of population?",
  "document_ids": [24],
  "debug": true
}
```

### Query Response

Example:

```json
{
  "question": "What is Lahore's position in Pakistan in terms of population?",
  "answer": "Lahore is the second-largest city in Pakistan after Karachi.",
  "filenames": ["lahore.txt"],
  "found_in_documents": true,
  "mode": "ask_docs_local",
  "groq_error": null,
  "debug": {
    "searched_documents": [
      {
        "document_id": 24,
        "filename": "lahore.txt"
      }
    ]
  }
}
```

### Notes on Query Scope

- if `document_ids` are provided:
  - retrieval is restricted to those documents only
- if `document_ids` are not provided:
  - retrieval searches across all stored documents in the current backend scope

For app integrations such as EchoPanel, the safest pattern is to:

1. upload file
2. read `document_ids[0]`
3. send that ID back in later `/ask-docs` requests

## EchoPanel Integration

This backend is ready to be consumed by EchoPanel or any other external app.

### Recommended Request Headers

```http
X-API-Key: echopanel-demo-key
X-Client-App: EchoPanel
```

### Recommended Flow

1. EchoPanel uploads a file to `POST /upload`
2. EchoPanel reads `document_ids[0]`
3. EchoPanel stores that ID
4. EchoPanel sends questions to `POST /ask-docs` using that `document_id`

### Recommended Ask Docs Payload

```json
{
  "question": "What does the uploaded document say about revenue?",
  "document_ids": [24],
  "debug": true
}
```

### Why `document_ids` Are Recommended

Using `document_ids` avoids:

- mixing older uploads with the current file
- noisy retrieval across unrelated documents
- debugging confusion in multi-upload sessions

## Run Locally

### 1. Open the Project

```powershell
cd C:\Users\affan.khan\Desktop\rag_chatbot
```

### 2. Create and Activate a Virtual Environment

```powershell
python -m venv venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\venv\Scripts\Activate.ps1)
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```powershell
Copy-Item .env.example .env
```

Then edit `.env` with your actual values.

### 5. Start PostgreSQL + pgvector

```powershell
docker compose up -d
```

### 6. Initialize the Database

```powershell
python setup_db.py
python seed.py
```

### 7. Start the FastAPI Backend

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Available at:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 8. Start the Streamlit UI

In another terminal:

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Usually available at:

- [http://localhost:8501](http://localhost:8501)

## Expose with ngrok

If another app such as EchoPanel needs to access your local RAG backend from outside your machine, expose it through ngrok.

### 1. Start the Backend

Make sure FastAPI is already running on port `8000`.

### 2. Install ngrok

```powershell
winget install ngrok.ngrok
```

### 3. Create an ngrok Account

Before using ngrok, create a free account at:

- [https://dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup)

After signing in, open your dashboard and copy your personal auth token from:

- [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)

### 4. Add Your ngrok Auth Token

```powershell
ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
```

If `ngrok` is not recognized in PowerShell immediately after installation, close and reopen the terminal. If needed, run the installed executable using its full path.

### 5. Start the Tunnel

```powershell
ngrok http 8000
```

ngrok will return a public URL like:

```text
https://your-subdomain.ngrok-free.app
```

### 6. Use That URL in EchoPanel

Example:

```ini
RAG_API_URL=https://your-subdomain.ngrok-free.app
RAG_API_KEY=echopanel-demo-key
```

### Important Notes

- keep the FastAPI backend running
- keep Docker/PostgreSQL running
- keep ngrok running
- free ngrok URLs may change after restart
- if your LiveKit agent is on LiveKit Cloud, `localhost` alone will not work

## Current Limitations

- no OCR support
- scanned PDFs are not supported
- table extraction is limited
- current user flow behaves like a shared backend scope unless `document_ids` are used
- answer quality can vary depending on chunking, reranking, and Groq settings

## Summary

This project is a hybrid RAG backend that combines:

- document ingestion
- chunking
- embeddings
- PostgreSQL + pgvector storage
- hybrid retrieval
- optional reranking
- optional Groq-based answer generation

It works both as:

- a local RAG application with Streamlit
- a reusable backend service for external apps such as EchoPanel
