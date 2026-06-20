"""
PDF ingestion via unstructured partition_pdf (notebook pattern).
Extracts CompositeElement text chunks, tables, and embedded images (base64).
"""

from __future__ import annotations

import os
import unstructured_pytesseract

# Fix the Tesseract trailing slash path resolution and binary binding on Windows
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
unstructured_pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List

from utils.tesseract_setup import configure_tesseract

import re


def _clean_text(text: str) -> str:
    """Remove common PDF/OCR artifacts from extracted text."""
    # Fix hyphenated line-breaks: "discov-\nery" → "discovery"
    text = re.sub(r"-\s*\n\s*([a-z])", r"\1", text)
    # Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize repeated spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip form-feed characters (page breaks embedded in text)
    text = text.replace("\x0c", "")
    return text.strip()


@dataclass
class ParentDocument:
    """Serializable parent document stored in the docstore."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str = "text"  # text | table | image


@dataclass
class IngestedPDF:
    source: str
    texts: List[ParentDocument]
    tables: List[ParentDocument]
    images: List[ParentDocument]


def _element_metadata(el: Any) -> dict[str, Any]:
    meta = getattr(el, "metadata", None)
    if meta is None:
        return {}
    if hasattr(meta, "to_dict"):
        return meta.to_dict()
    return dict(meta) if isinstance(meta, dict) else {}


def partition_pdf_file(pdf_path: str | Path) -> list:
    configure_tesseract()
    from unstructured.partition.pdf import partition_pdf
    from utils.config import CHUNK_MAX_CHARS, CHUNK_COMBINE_UNDER, CHUNK_NEW_AFTER

    return partition_pdf(
        filename=str(pdf_path),
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=CHUNK_MAX_CHARS,
        combine_text_under_n_chars=CHUNK_COMBINE_UNDER,
        new_after_n_chars=CHUNK_NEW_AFTER,
    )


def extract_from_chunks(chunks: list, source: str) -> tuple[list[ParentDocument], list[ParentDocument], list[ParentDocument]]:
    texts: list[ParentDocument] = []
    tables: list[ParentDocument] = []
    images: list[ParentDocument] = []

    for chunk in chunks:
        chunk_type = type(chunk).__name__
        if chunk_type == "Table":
            tables.append(
                ParentDocument(
                    text=_clean_text(getattr(chunk, "text", "") or ""),
                    metadata={**_element_metadata(chunk), "source": source, "kind": "table"},
                    kind="table",
                )
            )
        elif chunk_type == "CompositeElement":
            texts.append(
                ParentDocument(
                    text=_clean_text(getattr(chunk, "text", "") or ""),
                    metadata={**_element_metadata(chunk), "source": source, "kind": "text"},
                    kind="text",
                )
            )
            orig = getattr(chunk.metadata, "orig_elements", None) or []
            for el in orig:
                if type(el).__name__ == "Image":
                    b64 = getattr(el.metadata, "image_base64", None)
                    if b64:
                        image_meta = _element_metadata(el)
                        images.append(
                            ParentDocument(
                                text=b64,
                                metadata={**image_meta, "source": source, "kind": "image"},
                                kind="image",
                            )
                        )

    return texts, tables, images


def ingest_pdf(pdf_path: str | Path) -> IngestedPDF:
    pdf_path = Path(pdf_path)
    chunks = partition_pdf_file(pdf_path)
    texts, tables, images = extract_from_chunks(chunks, source=str(pdf_path))
    return IngestedPDF(source=str(pdf_path), texts=texts, tables=tables, images=images)


def iter_pdfs(folder: str | Path) -> Iterator[Path]:
    folder = Path(folder)
    if not folder.exists():
        return
    seen: set[Path] = set()
    for pattern in ("*.pdf", "**/*.pdf"):
        for path in folder.glob(pattern):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def ingest_folder(folder: str | Path) -> list[IngestedPDF]:
    results: list[IngestedPDF] = []
    for pdf in sorted(iter_pdfs(folder), key=lambda p: p.name.lower()):
        print(f"[ingest] {pdf.name}")
        try:
            results.append(ingest_pdf(pdf))
            doc = results[-1]
            print(f"         -> {len(doc.texts)} text, {len(doc.tables)} tables, {len(doc.images)} images")
        except Exception as e:
            print(f"   ⚠️ ERROR processing {pdf.name}: {e}. Skipping file.")
            continue
    return results
