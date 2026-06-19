"""CLI: query the multimodal RAG pipeline."""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from utils.config import VECTORSTORE_PATH
from utils.rag_graph import query_with_sources
from utils.vectorstore import MultimodalVectorStore


def _short_description(text: str, max_chars: int = 260) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Query multimodal RAG")
    parser.add_argument("question", nargs="?", default=None, help="Question to ask")
    parser.add_argument("--vectorstore", type=str, default=str(VECTORSTORE_PATH))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved source snippets")
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
    print(result["response"])

    captions = result.get("context", {}).get("image_captions", [])
    images = result.get("context", {}).get("images", [])
    if images and captions:
        print(f"\n[Image description] {captions[0]}")
    elif images:
        fallback = _short_description(result.get("response", ""))
        print(f"\n[Image description] {fallback or 'Description unavailable.'}")

    if args.show_sources:
        texts = result["context"].get("texts", [])
        if texts or images:
            print(f"\n--- Sources: {len(texts)} text/table chunks, {len(images)} images ---")
            for i, text in enumerate(texts, 1):
                meta = getattr(text, "metadata", {}) or {}
                if not isinstance(meta, dict):
                    meta = vars(meta) if hasattr(meta, "__dict__") else {}
                page = meta.get("page_number")
                snippet = (text.text if hasattr(text, "text") else str(text))[:200]
                print(f"  [{i}] page={page} {snippet}...")
            for i, img in enumerate(images, 1):
                cap = captions[i - 1] if i - 1 < len(captions) else ""
                print(f"  [img {i}] caption: {cap}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
