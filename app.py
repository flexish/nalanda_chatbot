"""Streamlit UI for multimodal RAG."""

import base64
import html
import io
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from utils.config import DATA_FOLDER, TOP_K, VECTORSTORE_PATH
from utils.rag_graph import query_with_sources
from utils.vectorstore import MultimodalVectorStore


st.set_page_config(page_title="Multimodal RAG - Nalanda", layout="wide")


@st.cache_resource(show_spinner="Loading vector store...")
def get_store(vectorstore_path: str) -> MultimodalVectorStore:
    return MultimodalVectorStore.load(vectorstore_path)


def display_base64_image(b64: str, caption: str = "") -> bool:
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        st.image(img, caption=caption or None, width="content")
        return True
    except Exception:
        return False


def _short_description(text: str, max_chars: int = 260) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _init_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _chat_history_for_prompt(max_turns: int = 4, max_chars: int = 1800) -> str:
    lines: list[str] = []
    for message in st.session_state.messages[-max_turns * 2 :]:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = (message.get("content") or "").strip()
        captions = [cap for cap in message.get("captions", []) if cap]
        if not content and captions:
            content = "Images shown: " + " | ".join(captions[:4])
        if content:
            lines.append(f"{role}: {content}")
    history = "\n".join(lines)
    return history if len(history) <= max_chars else history[-max_chars:]


def _render_user_message(content: str) -> None:
    _, right = st.columns([1, 2])
    with right:
        st.markdown(
            f'<div class="chat-row user-row"><div class="chat-bubble user-bubble">{html.escape(content)}</div></div>',
            unsafe_allow_html=True,
        )


def _render_assistant_message(message: dict) -> None:
    left, _ = st.columns([2, 1])
    with left:
        content = (message.get("content") or "").strip()
        images = message.get("images", []) or []
        captions = message.get("captions", []) or []
        verified = message.get("verified")
        verification_reason = (message.get("verification_reason") or "").strip()

        if content:
            st.markdown(
                f'<div class="chat-row assistant-row"><div class="chat-bubble assistant-bubble">{html.escape(content)}</div></div>',
                unsafe_allow_html=True,
            )

        for idx, image_b64 in enumerate(images):
            caption = captions[idx] if idx < len(captions) else ""
            caption = (caption or "").strip() or _short_description(content)
            if display_base64_image(image_b64, caption=caption):
                st.markdown(
                    f'<div class="image-caption">{html.escape(caption or "Description unavailable.")}</div>',
                    unsafe_allow_html=True,
                )

        if verified is not None:
            label = "Verified" if verified else "Not verified"
            reason = f": {verification_reason}" if verification_reason else ""
            st.caption(f"{label}{reason}")


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        if message.get("role") == "user":
            _render_user_message(message.get("content", ""))
        else:
            _render_assistant_message(message)


st.markdown(
    """
    <style>
    .block-container { padding-bottom: 6rem; }
    .chat-row { display: flex; margin: 0.35rem 0 0.75rem; width: 100%; }
    .user-row { justify-content: flex-end; }
    .assistant-row { justify-content: flex-start; }
    .chat-bubble {
        border-radius: 8px;
        line-height: 1.45;
        max-width: 100%;
        padding: 0.75rem 0.9rem;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .user-bubble { background: #0f766e; color: white; }
    .assistant-bubble {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        color: #111827;
    }
    .image-caption { color: #4b5563; font-size: 0.9rem; margin: -0.35rem 0 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Multimodal RAG")
st.caption("Unstructured PDF extraction · MultiVectorRetriever · LangGraph · Vision LLM")

_init_chat_state()

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
                removed = store.remove_source(str(selected_pdf))
                if removed:
                    log(f"Replacing {removed} existing entries for {selected_pdf.name}...")
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
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    try:
        stats = get_store(vectorstore_path).stats()
        st.metric("Summary vectors", stats["summary_vectors"])
        st.metric("Docstore entries", stats["docstore_entries"])
    except Exception:
        st.warning("Vector store not loaded yet.")

_render_chat_history()

question = st.chat_input("Ask about Nalanda Mahavihara...")
if question:
    chat_history = _chat_history_for_prompt()
    st.session_state.messages.append({"role": "user", "content": question})
    _render_user_message(question)

    with st.spinner("Retrieving and generating..."):
        try:
            store = get_store(vectorstore_path)
            if store.stats()["summary_vectors"] == 0:
                assistant_message = {
                    "role": "assistant",
                    "content": "Index is empty. Use the sidebar to index PDFs first.",
                    "images": [],
                    "captions": [],
                    "verified": False,
                    "verification_reason": "empty vector store",
                }
            else:
                result = query_with_sources(
                    store,
                    question,
                    top_k=top_k,
                    chat_history=chat_history,
                )
                images = result["context"].get("images", [])
                captions = result["context"].get("image_captions", [])
                content = result.get("response", "")
                if images and not content:
                    content = f"Found {len(images)} matching image{'s' if len(images) != 1 else ''}."
                assistant_message = {
                    "role": "assistant",
                    "content": content,
                    "images": images,
                    "captions": captions,
                    "verified": result.get("verified"),
                    "verification_reason": result.get("verification_reason", ""),
                }
        except Exception as exc:
            assistant_message = {
                "role": "assistant",
                "content": f"Query failed: {exc}",
                "images": [],
                "captions": [],
                "verified": False,
                "verification_reason": "query failed",
            }

    st.session_state.messages.append(assistant_message)
    _render_assistant_message(assistant_message)
