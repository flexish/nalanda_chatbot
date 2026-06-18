"""Streamlit UI for multimodal RAG."""

import base64
import io
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from utils.config import DATA_FOLDER, TOP_K, VECTORSTORE_PATH
from utils.rag_graph import query_with_sources
from utils.vectorstore import MultimodalVectorStore


@st.cache_resource(show_spinner="Loading vector store...")
def get_store(vectorstore_path: str) -> MultimodalVectorStore:
    return MultimodalVectorStore.load(vectorstore_path)


def display_base64_image(b64: str, caption: str = "") -> None:
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        st.image(img, caption=caption or None, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render image: {exc}")


st.set_page_config(page_title="Multimodal RAG — Nalanda", layout="wide")
st.title("Multimodal RAG")
st.caption("Unstructured PDF extraction · MultiVectorRetriever · LangGraph · Vision LLM")

with st.sidebar:
    st.header("Settings")
    vectorstore_path = st.text_input("Vector store path", value=str(VECTORSTORE_PATH))
    data_folder = st.text_input("Data folder", value=str(DATA_FOLDER))
    top_k = st.slider("Retrieval top-k", min_value=1, max_value=10, value=TOP_K)

    st.divider()
    st.subheader("Index documents")
    index_mode = st.radio("Index scope", ["Single PDF", "Entire folder"], horizontal=True)
    pdf_files = sorted(Path(data_folder).glob("**/*.pdf")) if Path(data_folder).exists() else []
    selected_pdf = None
    if index_mode == "Single PDF" and pdf_files:
        selected_pdf = st.selectbox("PDF", pdf_files, format_func=lambda p: p.name)
    elif index_mode == "Single PDF":
        st.info(f"No PDFs found in {data_folder}")
    elif index_mode == "Entire folder":
        st.warning(
            "Indexing all PDFs uses unstructured hi_res and can take several minutes per file. "
            "Prefer **Single PDF** while testing."
        )

    if st.button("Build / update index", type="primary"):
        progress = st.empty()

        def log(msg: str) -> None:
            progress.info(msg)

        try:
            store = MultimodalVectorStore(persist_dir=Path(vectorstore_path))
            if index_mode == "Single PDF" and selected_pdf:
                from utils.ingest import ingest_pdf

                log(f"Processing {selected_pdf.name}...")
                ingested = ingest_pdf(selected_pdf)
                counts = store.add_ingested(ingested, on_progress=log)
                st.success(f"Indexed {selected_pdf.name}: {counts}")
            elif index_mode == "Entire folder":
                totals = store.index_folder(Path(data_folder), on_progress=log)
                st.success(f"Indexed folder: {totals}")
            else:
                st.error("Select a PDF or add files to the data folder.")
            get_store.clear()
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    try:
        stats = get_store(vectorstore_path).stats()
        st.metric("Summary vectors", stats["summary_vectors"])
        st.metric("Docstore entries", stats["docstore_entries"])
    except Exception:
        st.warning("Vector store not loaded yet.")

question = st.chat_input("Ask about Nalanda Mahavihara...")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                store = get_store(vectorstore_path)
                if store.stats()["summary_vectors"] == 0:
                    st.error("Index is empty. Use the sidebar to index PDFs first.")
                    st.stop()
                result = query_with_sources(store, question, top_k=top_k)
                st.markdown(result["response"])

                with st.expander("Retrieved context", expanded=False):
                    texts = result["context"].get("texts", [])
                    images = result["context"].get("images", [])

                    for i, text in enumerate(texts, 1):
                        body = text.text if hasattr(text, "text") else str(text)
                        meta = getattr(text, "metadata", {}) or {}
                        page = meta.get("page_number") if isinstance(meta, dict) else getattr(meta, "page_number", None)
                        source = meta.get("source") if isinstance(meta, dict) else getattr(meta, "source", None)
                        st.markdown(f"**Chunk {i}** · page {page}")
                        if source:
                            st.caption(Path(source).name)
                        st.text(body[:2000] + ("..." if len(body) > 2000 else ""))
                        st.divider()

                    if images:
                        st.markdown(f"**{len(images)} retrieved image(s)**")
                        cols = st.columns(min(3, len(images)))
                        for idx, img_b64 in enumerate(images):
                            with cols[idx % len(cols)]:
                                display_base64_image(img_b64, caption=f"Image {idx + 1}")
            except Exception as exc:
                st.error(f"Query failed: {exc}")
