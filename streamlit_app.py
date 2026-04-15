"""
Simple Streamlit frontend for the hybrid RAG app.
Run with: streamlit run streamlit_app.py
"""

import time

import requests
import streamlit as st


BACKEND = "http://localhost:8000"

st.set_page_config(page_title="Hybrid RAG", page_icon="RAG")


def get_api(path: str):
    try:
        response = requests.get(f"{BACKEND}{path}", timeout=10)
        return response.json(), response.ok
    except Exception as exc:
        return {"detail": str(exc)}, False


def post_api(path: str, json_payload=None, files=None, timeout: int = 60):
    try:
        response = requests.post(
            f"{BACKEND}{path}",
            json=json_payload,
            files=files,
            timeout=timeout,
        )
        return response.json(), response.ok
    except Exception as exc:
        return {"detail": str(exc)}, False


def get_stats():
    return get_api("/stats")


def get_documents():
    return get_api("/documents")


def upload_files(uploaded_files):
    files = [
        (
            "files",
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            ),
        )
        for uploaded_file in uploaded_files
    ]
    return post_api("/upload", files=files, timeout=180)


def ask_question(question: str):
    return post_api("/query", json_payload={"question": question}, timeout=60)


def reset_knowledge():
    return post_api("/reset", timeout=30)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query_ms" not in st.session_state:
    st.session_state.last_query_ms = None
if "last_upload_ms" not in st.session_state:
    st.session_state.last_upload_ms = None


stats, stats_ok = get_stats()
documents_payload, documents_ok = get_documents()

with st.sidebar:
    st.title("Hybrid RAG")

    if stats_ok:
        st.metric("Documents", stats.get("document_count", 0))
        st.metric("Chunks", stats.get("chunk_count", 0))
    else:
        st.error("Backend is offline. Start FastAPI first.")

    st.divider()
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload .pdf, .txt, or .docx files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Index Files", type="primary"):
        start = time.perf_counter()
        result, ok = upload_files(uploaded_files)
        st.session_state.last_upload_ms = (time.perf_counter() - start) * 1000

        if ok:
            st.success(f"Indexed {result.get('uploaded_count', 0)} file(s).")
            for item in result.get("errors", []):
                st.warning(f"{item['filename']}: {item['error']}")
            st.rerun()
        else:
            st.error(result.get("detail", "Upload failed."))

    st.divider()
    st.subheader("Recent Files")
    if documents_ok and documents_payload.get("documents"):
        for document in documents_payload["documents"][:10]:
            st.caption(f"{document['filename']} | {document['chunk_count']} chunks")
    else:
        st.caption("No files uploaded yet.")

    st.divider()
    if st.button("Reset Knowledge"):
        _, ok = reset_knowledge()
        if ok:
            st.session_state.messages = []
            st.success("Knowledge reset.")
            st.rerun()
        else:
            st.error("Reset failed.")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.last_upload_ms is not None:
        st.caption(f"Last upload: {st.session_state.last_upload_ms:.2f} ms")
    if st.session_state.last_query_ms is not None:
        st.caption(f"Last query: {st.session_state.last_query_ms:.2f} ms")


st.title("Ask Your Files")

if not stats_ok:
    st.warning("Backend is not reachable.")
elif stats.get("document_count", 0) == 0:
    st.info("Upload files first, then ask questions.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        filenames = message.get("filenames")
        if filenames:
            st.caption(f"Filenames: {', '.join(filenames)}")
        sources = message.get("sources")
        if sources:
            with st.expander("Show source chunks"):
                for source in sources:
                    st.markdown(f"**{source['filename']}**")
                    st.write(source["content"])


question = st.chat_input("Ask a question about your uploaded files")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        result, ok = ask_question(question)
        st.session_state.last_query_ms = (time.perf_counter() - start) * 1000

        if ok:
            st.write(result.get("answer", "No answer returned."))
            filenames = result.get("filenames", [])
            if filenames:
                st.caption(f"Filenames: {', '.join(filenames)}")
            retrieval = result.get("retrieval", {})
            if retrieval:
                st.caption(
                    f"Vector mode: {retrieval.get('vector_mode', 'unknown')} | "
                    f"Chunks indexed: {retrieval.get('chunk_count', 0)}"
                )
            sources = result.get("sources", [])
            if sources:
                with st.expander("Show source chunks"):
                    for source in sources:
                        st.markdown(f"**{source['filename']}**")
                        st.write(source["content"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "filenames": filenames,
                    "sources": sources,
                }
            )
        else:
            error_message = result.get("detail", "Query failed.")
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )
