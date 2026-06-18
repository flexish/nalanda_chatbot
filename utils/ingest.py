"""
PDF ingestion via unstructured partition_pdf (notebook pattern).
Extracts CompositeElement text chunks, tables, and embedded images (base64).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List

from utils.tesseract_setup import configure_tesseract


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
    images: List[str]  # base64 strings


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

    return partition_pdf(
        filename=str(pdf_path),
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )


def extract_from_chunks(chunks: list, source: str) -> tuple[list[ParentDocument], list[ParentDocument], list[str]]:
    texts: list[ParentDocument] = []
    tables: list[ParentDocument] = []
    images: list[str] = []

    for chunk in chunks:
        chunk_type = type(chunk).__name__
        if chunk_type == "Table":
            tables.append(
                ParentDocument(
                    text=getattr(chunk, "text", "") or "",
                    metadata={**_element_metadata(chunk), "source": source, "kind": "table"},
                    kind="table",
                )
            )
        elif chunk_type == "CompositeElement":
            texts.append(
                ParentDocument(
                    text=getattr(chunk, "text", "") or "",
                    metadata={**_element_metadata(chunk), "source": source, "kind": "text"},
                    kind="text",
                )
            )
            orig = getattr(chunk.metadata, "orig_elements", None) or []
            for el in orig:
                if type(el).__name__ == "Image":
                    b64 = getattr(el.metadata, "image_base64", None)
                    if b64:
                        images.append(b64)

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
        results.append(ingest_pdf(pdf))
        doc = results[-1]
        print(f"         -> {len(doc.texts)} text, {len(doc.tables)} tables, {len(doc.images)} images")
    return results
