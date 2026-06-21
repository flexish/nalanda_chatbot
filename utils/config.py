"""Application configuration from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FOLDER = Path(os.getenv("DATA_FOLDER", "./data"))
VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", "./vectorstore"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" | "anthropic" | "openrouter"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "openai")  # "openai" | "huggingface"

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "multi_modal_rag")
DOCSTORE_FILENAME = "docstore.pkl"

TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
SUMMARIZE_CONCURRENCY = int(os.getenv("SUMMARIZE_CONCURRENCY", "3"))
SUMMARIZE_TEXT = os.getenv("SUMMARIZE_TEXT", "true").lower() == "true"

# PDF chunking (unstructured partition_pdf)
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "10000"))
CHUNK_COMBINE_UNDER = int(os.getenv("CHUNK_COMBINE_UNDER", "2000"))
CHUNK_NEW_AFTER = int(os.getenv("CHUNK_NEW_AFTER", "6000"))

# Advanced RAG
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
MAX_ITERATIVE_HOPS = int(os.getenv("MAX_ITERATIVE_HOPS", "1"))

IMAGE_SUMMARY_PROMPT = os.getenv(
    "IMAGE_SUMMARY_PROMPT",
    "Describe the image in detail. For context, the image is part of documents about "
    "the Archaeological Site of Nalanda Mahavihara. Explain what the image shows in "
    "relation to the site's history, archaeology, and conservation.",
)

TEXT_SUMMARY_PROMPT = """You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}
"""

ANSWER_PROMPT_TEMPLATE = """Answer the question based only on the following context, which can include text, tables, and the below image.
Context: {context_text}
Question: {question}
"""
