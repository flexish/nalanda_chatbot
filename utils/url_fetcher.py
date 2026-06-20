"""Fetch and extract clean text from admin-configured URLs for use as RAG context."""
from __future__ import annotations

import re
import time
from typing import Optional

from utils.ingest import ParentDocument

_TIMEOUT = 10
_MAX_CHARS = 8000
_CACHE_TTL = 3600          # cache fetched content for 1 hour
_STRIP_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]

# url → (fetch_timestamp, ParentDocument)
_cache: dict[str, tuple[float, ParentDocument]] = {}


def fetch_url_content(url: str) -> Optional[ParentDocument]:
    """Fetch a URL and return its text as a ParentDocument. Returns None on failure."""
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NalandaRAG/1.0)"}
        resp = requests.get(url.strip(), timeout=_TIMEOUT, headers=headers)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "").lower()

        if "text/html" in content_type:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.content, "html.parser")
            for tag in soup(_STRIP_TAGS):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        elif "text/plain" in content_type:
            text = resp.text
        else:
            return None

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()

        if not text:
            return None

        return ParentDocument(
            text=text[:_MAX_CHARS],
            metadata={"source": url, "kind": "text", "origin": "url"},
            kind="text",
        )
    except Exception:
        return None


def fetch_urls(urls: list[str]) -> list[ParentDocument]:
    """Fetch multiple URLs concurrently, return only successful results."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results: list[ParentDocument] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_url_content, u): u for u in urls if u.strip()}
        for fut in as_completed(futures):
            doc = fut.result()
            if doc:
                results.append(doc)
    return results


def fetch_urls_cached(urls: list[str]) -> list[ParentDocument]:
    """Fetch URLs with a 1-hour in-process cache. Stale entries are re-fetched."""
    now = time.time()
    results: list[ParentDocument] = []
    to_fetch: list[str] = []

    for url in urls:
        entry = _cache.get(url)
        if entry and now - entry[0] < _CACHE_TTL:
            results.append(entry[1])
        else:
            to_fetch.append(url)

    if to_fetch:
        fresh = fetch_urls(to_fetch)
        for doc in fresh:
            _cache[doc.metadata["source"]] = (now, doc)
        results.extend(fresh)

    return results


def invalidate_cache(url: str | None = None) -> None:
    """Invalidate the cache for a specific URL, or clear all if url is None."""
    if url:
        _cache.pop(url, None)
    else:
        _cache.clear()
