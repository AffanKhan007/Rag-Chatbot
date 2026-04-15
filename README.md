# Hybrid RAG App

This project is a simple FastAPI + PostgreSQL/pgvector + Streamlit RAG app.

## What It Does

- Upload one or many `.pdf`, `.txt`, or `.docx` files
- Extract text and chunk it
- Create embeddings and PostgreSQL full-text indexes
- Run hybrid retrieval with vector search + keyword search
- Answer questions from uploaded files

## Run

```powershell
pip install -r requirements.txt
docker compose up -d
python setup_db.py
python seed.py
uvicorn app.main:app --reload
streamlit run streamlit_app.py
```

## API

- `POST /upload`
- `GET /documents`
- `POST /query`
- `GET /stats`
- `POST /reset`

## Notes

- Workspaces were removed from the user flow
- Groq generation is disabled by default for faster answers
- Optional reranking stays behind `ENABLE_RERANK`
- Small document sets use exact cosine search by default
- Large chunk collections can switch to HNSW automatically with `USE_HNSW_FOR_LARGE_DATA=true`
- `HNSW_CHUNK_THRESHOLD` controls when the app switches from exact cosine to HNSW
