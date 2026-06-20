"""
Chroma vector store + persistent docstore for MultiVectorRetriever (notebook pattern).
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings

from utils.config import (
    COLLECTION_NAME,
    DOCSTORE_FILENAME,
    EMBEDDING_BACKEND,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    TOP_K,
    VECTORSTORE_PATH,
)
from utils.ingest import IngestedPDF, ParentDocument
from utils.summarizer import summarize_images, summarize_tables, summarize_texts


class PersistentDocstore(InMemoryStore):
    """InMemoryStore that persists to disk between sessions."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            if data:
                self.mset(list(data.items()))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(dict(self.store), f)


class MultimodalVectorStore:
    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = COLLECTION_NAME):
        self.persist_dir = Path(persist_dir or VECTORSTORE_PATH)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.id_key = "doc_id"

        if EMBEDDING_BACKEND == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            embedding_fn = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            embedding_fn = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                api_key=OPENAI_API_KEY,
            )
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_fn,
            persist_directory=str(self.persist_dir),
        )
        self.docstore = PersistentDocstore(self.persist_dir / DOCSTORE_FILENAME)
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key,
            search_kwargs={"k": TOP_K},
        )

    def add_ingested(
        self,
        ingested: IngestedPDF,
        on_progress: Optional[Any] = None,
    ) -> dict[str, int]:
        """Summarize and index one PDF. Returns counts added."""
        text_summaries = summarize_texts(ingested.texts, on_progress)
        table_summaries = summarize_tables(ingested.tables, on_progress)
        image_summaries = summarize_images(ingested.images, on_progress)

        counts = {"texts": 0, "tables": 0, "images": 0}

        if text_summaries:
            counts["texts"] = self._add_parents(ingested.texts, text_summaries, ingested.source)
        if table_summaries:
            counts["tables"] = self._add_parents(ingested.tables, table_summaries, ingested.source)
        if image_summaries:
            counts["images"] = self._add_image_parents(ingested.images, image_summaries, ingested.source)

        self.docstore.save()
        return counts

    def remove_source(self, source: str) -> int:
        """Remove any vector-store and docstore entries associated with a source file."""
        ids: list[str] = []
        try:
            matches = self.vectorstore.get(where={"source": source})
            ids = list(matches.get("ids", []) or [])
        except Exception:
            ids = []

        if ids:
            try:
                self.vectorstore.delete(ids=ids)
            except Exception:
                try:
                    self.vectorstore._collection.delete(ids=ids)
                except Exception:
                    pass

            for doc_id in ids:
                self.docstore.store.pop(doc_id, None)
            self.docstore.save()

        return len(ids)

    def _add_parents(
        self,
        parents: Sequence[ParentDocument],
        summaries: Sequence[str],
        source: str,
    ) -> int:
        doc_ids = [str(uuid.uuid4()) for _ in parents]
        summary_docs = [
            Document(page_content=summary, metadata={self.id_key: doc_ids[i], "source": source})
            for i, summary in enumerate(summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_docs)
        self.retriever.docstore.mset(list(zip(doc_ids, list(parents))))
        return len(parents)

    def _add_image_parents(
        self,
        images_b64: Sequence[str],
        summaries: Sequence[str],
        source: str,
    ) -> int:
        doc_ids = [str(uuid.uuid4()) for _ in images_b64]
        summary_docs = [
            Document(page_content=summary, metadata={self.id_key: doc_ids[i], "source": source})
            for i, summary in enumerate(summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_docs)
        image_parents = [
            ParentDocument(text=image_b64, metadata={"kind": "image", self.id_key: doc_ids[i], "source": source}, kind="image")
            for i, image_b64 in enumerate(images_b64)
        ]
        self.retriever.docstore.mset(list(zip(doc_ids, image_parents)))
        return len(images_b64)

    def index_folder(
        self,
        folder: Path,
        on_progress: Optional[Any] = None,
    ) -> dict[str, int]:
        from utils.ingest import ingest_folder

        totals = {"texts": 0, "tables": 0, "images": 0, "pdfs": 0}
        for ingested in ingest_folder(folder):
            if on_progress:
                on_progress(f"Indexing {Path(ingested.source).name}...")
            counts = self.add_ingested(ingested, on_progress)
            totals["pdfs"] += 1
            for k in ("texts", "tables", "images"):
                totals[k] += counts[k]
        return totals

    def stats(self) -> dict[str, Any]:
        collection = self.vectorstore._collection
        count = collection.count() if collection else 0
        return {
            "collection": self.collection_name,
            "summary_vectors": count,
            "docstore_entries": len(self.docstore.store),
            "persist_dir": str(self.persist_dir),
        }

    @classmethod
    def load(cls, persist_dir: Optional[Path] = None) -> "MultimodalVectorStore":
        return cls(persist_dir=persist_dir)
