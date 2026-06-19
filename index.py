"""CLI: index PDFs from DATA_FOLDER into the vector store."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from utils.config import DATA_FOLDER, VECTORSTORE_PATH
from utils.vectorstore import MultimodalVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Index PDFs for multimodal RAG")
    parser.add_argument(
        "--folder",
        type=Path,
        default=DATA_FOLDER,
        help="Folder containing PDF files",
    )
    parser.add_argument(
        "--vectorstore",
        type=Path,
        default=VECTORSTORE_PATH,
        help="Path to persist Chroma + docstore",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Index a single PDF instead of the whole folder",
    )
    args = parser.parse_args()

    store = MultimodalVectorStore(persist_dir=args.vectorstore)

    if args.pdf:
        from utils.ingest import ingest_pdf

        print(f"Indexing {args.pdf}...")
        ingested = ingest_pdf(args.pdf)
        counts = store.add_ingested(ingested, on_progress=print)
        print(f"Done: {counts}")
    else:
        folder = args.folder
        if not folder.exists():
            print(f"Data folder not found: {folder}", file=sys.stderr)
            return 1
        print(f"Indexing PDFs in {folder} -> {args.vectorstore}")
        
        # --- RESUME CAPABILITY IMPLEMENTATION ---
        # 1. Look up files already processed using the store's metadata tracking
        try:
            # Most Chroma/LangChain wrappers let you extract all existing document metadata
            existing_data = store.chroma_vectorstore.get(include=["metadatas"])
            indexed_sources = {meta.get("source") for meta in existing_data.get("metadatas", []) if meta and "source" in meta}
        except Exception:
            try:
                # Secondary fallback depending on how your internal vectorstore class is named
                existing_data = store.db.get(include=["metadatas"])
                indexed_sources = {meta.get("source") for meta in existing_data.get("metadatas", []) if meta and "source" in meta}
            except Exception:
                indexed_sources = set()

        # 2. Dynamically clean and run individual files instead of processing the entire folder at once
        from utils.ingest import iter_pdfs, ingest_pdf
        
        pdf_files = sorted(iter_pdfs(folder), key=lambda p: p.name.lower())
        totals = {"text": 0, "table": 0, "image": 0}
        
        for pdf in pdf_files:
            if str(pdf) in indexed_sources:
                print(f"[skip] {pdf.name} is already indexed in the vector store.")
                continue
                
            print(f"[ingest] Processing remaining file: {pdf.name}")
            try:
                ingested = ingest_pdf(pdf)
                counts = store.add_ingested(ingested, on_progress=print)
                for k, v in counts.items():
                    totals[k] = totals.get(k, 0) + v
            except Exception as e:
                print(f"   ⚠️ ERROR processing {pdf.name}: {e}. Skipping file.")
                continue
                
        print(f"Done: {totals}")
        # --- END RESUME CAPABILITY ---

    print("Store stats:", store.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
