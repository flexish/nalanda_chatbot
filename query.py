"""CLI: query the multimodal RAG pipeline."""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from utils.config import VECTORSTORE_PATH
from utils.rag_graph import query_with_sources
from utils.vectorstore import MultimodalVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Query multimodal RAG")
    parser.add_argument("question", nargs="?", default=None, help="Question to ask")
    parser.add_argument("--vectorstore", type=str, default=str(VECTORSTORE_PATH))
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    question = args.question
    if not question:
        question = input("Question: ").strip()
    if not question:
        print("No question provided.", file=sys.stderr)
        return 1

    store = MultimodalVectorStore.load(args.vectorstore)
    stats = store.stats()
    if stats["summary_vectors"] == 0:
        print("Vector store is empty. Run: python index.py", file=sys.stderr)
        return 1

    result = query_with_sources(store, question, top_k=args.top_k)
    print("\nAnswer:\n", result["response"])

    texts = result["context"].get("texts", [])
    images = result["context"].get("images", [])
    if texts or images:
        print(f"\n--- Sources: {len(texts)} text/table chunks, {len(images)} images ---")
        for i, text in enumerate(texts, 1):
            meta = getattr(text, "metadata", {}) or {}
            if not isinstance(meta, dict):
                meta = vars(meta) if hasattr(meta, "__dict__") else {}
            page = meta.get("page_number")
            snippet = (text.text if hasattr(text, "text") else str(text))[:200]
            print(f"  [{i}] page={page} {snippet}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
