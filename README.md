# Offline Local RAG Chatbot

This is a clean, 100% offline Retrieval-Augmented Generation (RAG) system built with:
- FastAPI backend API (`app/main.py`)
- PostgreSQL + `pgvector` (Docker on `5433`)
- Async SQLAlchemy + `asyncpg`
- Local sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- Streamlit frontend (`streamlit_app.py`)

## Prerequisites
Before you start, make sure you have installed:
1. **VS Code** (Visual Studio Code)
2. **Python 3.10+**
3. **Docker Desktop** (Make sure it is open and running!)

---

## 🛠 VS Code Setup & Run Guide

### Step 1: Open the Project in VS Code
1. Open VS Code.
2. Click **File > Open Folder** and select the `rag_chatbot` folder.
3. Once opened, open the integrated terminal by pressing `Ctrl + ` ` (backtick) or clicking **Terminal > New Terminal** in the top menu.

### Step 2: Create the Virtual Environment
In your VS Code terminal, run the following commands to isolate your Python packages:
```powershell
# 1. Create the virtual environment
python -m venv venv

# 2. Activate it (Windows)
.\venv\Scripts\activate

# (If you are on Mac/Linux, use: source venv/bin/activate)

# 3. Install all AI and Backend dependencies
pip install -r requirements.txt
```

### Step 3: Start the Vector Database (Docker)
This project uses a Dockerized PostgreSQL instance pre-loaded with `pgvector`.
1. Make sure Docker Desktop is running.
2. In your VS Code terminal, run:
```powershell
# Spin up the database on port 5433
docker compose up -d
```
3. Enable the vector extension inside the database by running:
```powershell
docker compose exec -T db psql -U postgres -d rag_chatbot -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Step 4: Environment Variables
Create a file named `.env` in your project root (if it doesn't exist) and paste the following:
```ini
DATABASE_URL=postgresql+asyncpg://postgres:Affankhan0966@localhost:5433/rag_chatbot
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
```
`GROQ_API_KEY` is optional. If set, `/query` will generate a more readable grounded answer via Groq and fall back to local extraction when unavailable.
Use `.env.example` as the template for local setup.

### Step 5: Seed the AI Knowledge Base
You need to convert your text data into mathematical vectors.
1. Open `data/physics.txt` or `seed.py` and modify/add any text you want the AI to learn.
2. Run the seeder in your VS Code terminal:
```powershell
python seed.py
```
*(Note: The first time you run this, it will take a minute to download the `all-MiniLM-L6-v2` AI model to your computer).*

### Step 6: Start the FastAPI Server
Once the data is seeded, boot up your API!
```powershell
uvicorn app.main:app --reload
```

### Step 7: Run the Streamlit Frontend
Start the frontend UI (recommended) in a second terminal:
```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```
Then open: [http://localhost:8501](http://localhost:8501)

### Step 8: Test it Out (Swagger / API)
1. Open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. Upload in `/upload` (`.txt`, `.md`, `.csv`, `.json`)
3. Ask in `/query`

---

## Vector Search Notes (HNSW + Cosine)

- Search uses cosine distance (`<=>`) instead of Euclidean (`<->`).
- Embeddings are normalized (`normalize_embeddings=True`) before storage/querying.
- Current ingestion chunking in `app/main.py`:
  - `chunk_size = 250`
  - `overlap = 25` (10%)
- An HNSW index is created for ANN search:
```sql
CREATE INDEX IF NOT EXISTS knowledge_embedding_hnsw_idx
ON knowledge
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

## Supported Upload File Types

- `.txt` (plain UTF-8 text)
- `.md` (Markdown as text)
- `.csv` (rows converted into text)
- `.json` (JSON converted to formatted text)

## Quick API Examples (PowerShell)

```powershell
# Reset all indexed knowledge (recommended before new upload tests)
curl -Method POST -Uri "http://127.0.0.1:8000/reset"

# Upload a file
curl -Method POST -Uri "http://127.0.0.1:8000/upload" -Form @{ file = Get-Item ".\data\sample.txt" }

# How many chunks are indexed? (should be > 0 after upload)
curl "http://127.0.0.1:8000/stats"

# Ask a question
curl "http://127.0.0.1:8000/query?q=What%20is%20the%20main%20topic%3F"
```

## If `/query` returns "No data found"

1. Open [http://127.0.0.1:8000/stats](http://127.0.0.1:8000/stats) — `knowledge_rows` must be greater than zero after upload.
2. If it is zero, upload failed or you hit **Reset** after uploading. **Reset** (`POST /reset`) then upload again.
3. Always start the API from the project folder so `.env` loads: `.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000`
4. Keep Docker running: `docker compose up -d` (Postgres on port `5433`).

---

## GitHub Push Readiness

- Do not commit secrets. `.env` is ignored via `.gitignore`.
- Keep real keys only in local `.env`.
- Share only `.env.example` in the repository.
- Ensure both backend and frontend start before pushing:
  - `.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000`
  - `.\venv\Scripts\python.exe -m streamlit run streamlit_app.py`
- Verify API flow:
  - `POST /upload` works
  - `GET /query` returns answer
  - `GET /stats` and `POST /reset` work
