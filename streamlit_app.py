"""
Simple Streamlit frontend for Local RAG Chatbot
Talks to FastAPI backend at http://localhost:8000
Run: streamlit run streamlit_app.py
"""

import requests
import streamlit as st
import time

BACKEND = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_stats():
    try:
        r = requests.get(f"{BACKEND}/stats", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def upload_file(file_bytes, filename):
    try:
        r = requests.post(
            f"{BACKEND}/upload",
            files={"file": (filename, file_bytes)},
            timeout=60,
        )
        return r.json(), r.ok
    except Exception as e:
        return {"detail": str(e)}, False


def ask_question(question: str):
    try:
        r = requests.get(f"{BACKEND}/query", params={"q": question}, timeout=30)
        return r.json(), r.ok
    except Exception as e:
        return {"detail": str(e)}, False


def reset_knowledge():
    try:
        r = requests.post(f"{BACKEND}/reset", timeout=10)
        return r.ok
    except Exception:
        return False


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "user"|"assistant", "content": "..."}]
if "query_latencies_ms" not in st.session_state:
    st.session_state.query_latencies_ms = []
if "last_upload_latency_ms" not in st.session_state:
    st.session_state.last_upload_latency_ms = None
if "last_query_latency_ms" not in st.session_state:
    st.session_state.last_query_latency_ms = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 RAG Chatbot")

    stats = get_stats()
    if stats:
        st.metric("Chunks Indexed", stats.get("knowledge_rows", 0))
        db_target = stats.get("database_target", {})
        if db_target:
            st.caption(
                f"DB: {db_target.get('host')}:{db_target.get('port')} / {db_target.get('database')}"
            )
    else:
        st.error("⚠️ Backend offline — start uvicorn first.")

    st.divider()
    st.subheader("⏱️ Latency (Real-Time)")
    if st.session_state.last_upload_latency_ms is not None:
        st.write(f"Upload: {st.session_state.last_upload_latency_ms:.2f} ms")
    if st.session_state.last_query_latency_ms is not None:
        st.write(f"Query: {st.session_state.last_query_latency_ms:.2f} ms")

    recent = st.session_state.query_latencies_ms[-10:]
    if recent:
        avg_ms = sum(recent) / len(recent)
        sorted_recent = sorted(recent)
        p95_idx = int(round(0.95 * (len(sorted_recent) - 1)))
        p95_ms = sorted_recent[p95_idx]
        st.caption(f"Last 10 queries: {len(recent)}")
        st.caption(f"Avg: {avg_ms:.2f} ms")
        st.caption(f"P95: {p95_ms:.2f} ms")
    if st.button("Reset Latency Stats"):
        st.session_state.query_latencies_ms = []
        st.session_state.last_upload_latency_ms = None
        st.session_state.last_query_latency_ms = None
        st.rerun()

    st.divider()

    st.subheader("📂 Upload Knowledge")
    uploaded = st.file_uploader(
        "Choose a file (.txt, .md, .csv, .json)",
        type=["txt", "md", "csv", "json"],
    )
    if uploaded and st.button("Index File", type="primary"):
        with st.spinner("Indexing…"):
            start = time.perf_counter()
            result, ok = upload_file(uploaded.read(), uploaded.name)
            upload_latency_ms = (time.perf_counter() - start) * 1000
            st.session_state.last_upload_latency_ms = upload_latency_ms
        if ok and result.get("chunks_indexed"):
            st.success(
                f"Indexed {result['chunks_indexed']} chunks from `{uploaded.name}`"
                f" (total rows: {result.get('knowledge_rows', 'n/a')})"
            )
            st.info(f"Upload latency: {upload_latency_ms:.2f} ms")
            st.rerun()
        else:
            st.error(result.get("detail") or result.get("message") or str(result))

    st.divider()

    if st.button("🗑️ Clear All Knowledge"):
        if reset_knowledge():
            st.session_state.messages = []
            st.success("Knowledge cleared.")
            st.rerun()
        else:
            st.error("Reset failed.")

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("💬 Ask Your Knowledge Base")

if not stats:
    st.warning("Backend is not reachable. Make sure uvicorn is running.")
elif stats.get("knowledge_rows", 0) == 0:
    st.info("No knowledge indexed yet. Upload a file in the sidebar first.")

# Render chat history using native st.chat_message
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("source"):
            st.caption(f"📄 Source: {msg['source']}")

# Chat input
question = st.chat_input("Ask a question…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            start = time.perf_counter()
            result, ok = ask_question(question)
            query_latency_ms = (time.perf_counter() - start) * 1000
            st.session_state.last_query_latency_ms = query_latency_ms
            st.session_state.query_latencies_ms.append(query_latency_ms)
            st.session_state.query_latencies_ms = st.session_state.query_latencies_ms[-100:]

        if ok and result.get("answer"):
            answer = result["answer"]
            source = result.get("topic", "")
            source_chunk = result.get("source_chunk", "")
            st.write(answer)
            st.caption(f"⏱️ Query latency: {query_latency_ms:.2f} ms")
            if source:
                st.caption(f"📄 Source: {source}")
            if source_chunk:
                with st.expander("Show retrieved chunk"):
                    st.write(source_chunk)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "source": source}
            )
        elif ok and result.get("message"):
            msg_text = f"{result['message']} — {result.get('hint', '')}"
            st.warning(msg_text)
            st.session_state.messages.append({"role": "assistant", "content": msg_text})
        else:
            err = f"Error: {result.get('detail', 'Unknown error')}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
