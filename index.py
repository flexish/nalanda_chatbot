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
        totals = store.index_folder(folder, on_progress=print)
        print(f"Done: {totals}")

    print("Store stats:", store.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
