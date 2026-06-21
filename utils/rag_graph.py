"""
LangGraph orchestration for multimodal RAG (notebook retrieve → parse → answer flow).
"""

from __future__ import annotations

from base64 import b64decode
import io
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from PIL import Image

from utils.config import (
    ANSWER_PROMPT_TEMPLATE,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    ENABLE_RERANKING,
    ENABLE_WEB_SEARCH,
    LLM_PROVIDER,
    MAX_ITERATIVE_HOPS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    RERANKER_MODEL,
    TOP_K,
    WEB_SEARCH_MAX_RESULTS,
)
from utils.ingest import ParentDocument
from utils.vectorstore import MultimodalVectorStore


def _build_llm() -> Any:
    if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY)
    if LLM_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        return ChatOpenAI(model=OPENROUTER_MODEL, api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)


# ── Cross-encoder cache (loaded once per process) ─────────────────────────────
_reranker_cache: dict[str, Any] = {}


def _get_reranker():
    if RERANKER_MODEL not in _reranker_cache:
        from sentence_transformers import CrossEncoder
        _reranker_cache[RERANKER_MODEL] = CrossEncoder(RERANKER_MODEL)
    return _reranker_cache[RERANKER_MODEL]


# ── Query rewriting + Multi-Query + HyDE ──────────────────────────────────────

_PRONOUN_WORDS = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "their",
                  "this", "that", "these", "those", "who", "whom"}


def _resolve_pronouns(llm: Any, question: str, chat_history: str) -> str:
    """Replace pronouns/vague references with the specific entity from chat history.

    Runs before retrieval so the vectorstore fetches the right documents.
    """
    words = set(question.lower().split())
    if not chat_history or not (words & _PRONOUN_WORDS):
        return question
    prompt = (
        f"Conversation history:\n{chat_history}\n\n"
        f"Question: {question}\n\n"
        "If the question contains pronouns (he, she, it, they, his, her) or vague references "
        "(this person, the above, that place) that refer to something in the conversation history, "
        "rewrite the question replacing those references with the specific entity from the history.\n"
        "If no pronoun resolution is needed, return the question unchanged.\n"
        "Return only the rewritten question, nothing else."
    )
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        resolved = (result.content if hasattr(result, "content") else str(result)).strip()
        # Reject if LLM returns a long explanation instead of just the question
        if not resolved or len(resolved) >= len(question) * 3:
            return question
        # Strip trailing punctuation the LLM may add
        return resolved.rstrip(".,!?;:")
    except Exception:
        return question


def _rewrite_and_expand_queries(llm: Any, question: str, chat_history: str) -> list[str]:
    """Use LLM to generate rewritten query, alternative phrasing, and HyDE snippet."""
    history_ctx = f"Chat history:\n{chat_history}\n\n" if chat_history else ""
    prompt = (
        f"{history_ctx}"
        "You are a search query optimizer for a RAG system about Nalanda Mahavihara.\n\n"
        f"User question: {question}\n\n"
        "Generate 3 search inputs:\n"
        "1. QUERY1: Rewrite the question as an optimized semantic search query (resolve pronouns using history if needed)\n"
        "2. QUERY2: An alternative phrasing that might retrieve different relevant chunks\n"
        "3. HYDE: A 1-2 sentence hypothetical answer snippet that would appear in a real document about this topic\n\n"
        "Reply in this exact format:\n"
        "QUERY1: <rewritten query>\n"
        "QUERY2: <alternative phrasing>\n"
        "HYDE: <hypothetical document snippet>"
    )
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        text = result.content if hasattr(result, "content") else str(result)
        queries: list[str] = [question]
        for line in text.splitlines():
            line = line.strip()
            for prefix in ("QUERY1:", "QUERY2:", "HYDE:"):
                if line.startswith(prefix):
                    q = line[len(prefix):].strip()
                    if q and q not in queries:
                        queries.append(q)
        return queries
    except Exception:
        return [question]


# ── Re-ranking ────────────────────────────────────────────────────────────────

def _rerank_docs(question: str, docs: list[Any], top_n: int | None = None) -> list[Any]:
    """Re-rank docs using a cross-encoder. Falls back silently if unavailable."""
    if not docs or not ENABLE_RERANKING:
        return docs
    try:
        reranker = _get_reranker()
        pairs = [(question, _doc_content(doc)) for doc in docs if _doc_content(doc)]
        if not pairs:
            return docs
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        result = [doc for _, doc in ranked]
        return result[:top_n] if top_n else result
    except Exception:
        return docs[:top_n] if top_n else docs


# ── CRAG: retrieval quality check + web search fallback ───────────────────────

def _retrieval_quality_score(docs: list[Any]) -> float:
    """0-1 score based on total retrieved text length."""
    if not docs:
        return 0.0
    total = sum(len(_doc_content(doc)) for doc in docs)
    return min(1.0, total / 800)


def _web_image_search(query: str, max_results: int = 4) -> list[tuple[str, str]]:
    """Fetch images from the web via DuckDuckGo. Returns (base64, caption) tuples."""
    if not ENABLE_WEB_SEARCH:
        return []
    try:
        import requests as _requests
        from base64 import b64encode
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.images(query, max_results=max_results * 3))
        results: list[tuple[str, str]] = []
        for r in raw:
            if len(results) >= max_results:
                break
            url = r.get("thumbnail") or r.get("image", "")
            if not url:
                continue
            try:
                resp = _requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200 and resp.content:
                    b64 = b64encode(resp.content).decode()
                    if _is_renderable_image(b64):
                        caption = (r.get("title") or f"Image from {r.get('source', 'web')}").strip()
                        results.append((b64, caption))
            except Exception:
                continue
        return results
    except Exception:
        return []


def _web_search(query: str) -> list[ParentDocument]:
    """Web search returning results as ParentDocuments (uses ddgs / duckduckgo_search)."""
    if not ENABLE_WEB_SEARCH:
        return []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS))
        docs = []
        for r in results:
            snippet = f"{r.get('title', '')}: {r.get('body', '')}".strip()
            if snippet:
                docs.append(ParentDocument(
                    text=snippet,
                    metadata={"source": "web", "kind": "text", "url": r.get("href", "")},
                    kind="text",
                ))
        return docs
    except Exception:
        return []


class RAGState(TypedDict, total=False):
    question: str
    chat_history: str
    retrieval_question: str
    retrieved_docs: list[Any]
    context: dict[str, list[Any]]
    mode: str
    response: str
    verified: bool
    verification_reason: str
    hops: int
    web_docs: list[Any]  # admin-configured URL content injected directly into LLM prompt
    web_searched: bool   # True once the internet-fallback node has run


IMAGE_QUERY_HINTS = (
    "show",
    "image",
    "photo",
    "picture",
    "figure",
    "view",
    "display",
    "see",
    "look",
    "draw",
)

FOLLOWUP_HINTS = (
    "it",
    "its",
    "that",
    "this",
    "they",
    "them",
    "those",
    "these",
    "same",
    "above",
    "previous",
    "more",
    "also",
)

MIN_IMAGE_RELEVANCE = 0.22
MIN_IMAGE_SCORE = 220
MIN_IMAGE_TOKEN_OVERLAP = 1

IMAGE_STOPWORDS = {
    "about",
    "also",
    "an",
    "and",
    "are",
    "around",
    "as",
    "be",
    "can",
    "did",
    "do",
    "does",
    "draw",
    "display",
    "figure",
    "find",
    "for",
    "from",
    "give",
    "has",
    "have",
    "image",
    "in",
    "is",
    "it",
    "look",
    "me",
    "of",
    "on",
    "photo",
    "picture",
    "please",
    "see",
    "show",
    "tell",
    "the",
    "this",
    "that",
    "to",
    "view",
    "was",
    "what",
    "when",
    "where",
    "who",
    "with",
    "would",
}

IMAGE_ENTITY_STOPWORDS = IMAGE_STOPWORDS | {
    "about",
    "describe",
    "details",
    "explain",
    "image",
    "picture",
    "photo",
    "shown",
    "show",
    "sketch",
    "statue",
    "sculpture",
    "subject",
    "thing",
    "view",
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _is_image_request(question: str) -> bool:
    lowered = question.lower()
    return any(hint in lowered for hint in IMAGE_QUERY_HINTS)


def _looks_like_followup(question: str) -> bool:
    tokens = _tokenize(question)
    return len(tokens) <= 8 or bool(tokens & set(FOLLOWUP_HINTS))


def _retrieval_question(question: str, chat_history: str = "") -> str:
    if chat_history and _looks_like_followup(question):
        return f"{chat_history}\nCurrent question: {question}"
    return question


def _requested_image_limit(question: str) -> int | None:
    lowered = question.lower()
    patterns = (
        r"\b(?:only|show|display|give|fetch|get|return|need|want|just|exactly)\s+(\d{1,2})\s+(?:image|images|photo|photos|picture|pictures|figure|figures)\b",
        r"\b(\d{1,2})\s+(?:image|images|photo|photos|picture|pictures|figure|figures)\b",
        r"\b(?:only|show|display|give|fetch|get|return|need|want|just|exactly)\s+(" + "|".join(NUMBER_WORDS) + r")\s+(?:image|images|photo|photos|picture|pictures|figure|figures)\b",
        r"\b(" + "|".join(NUMBER_WORDS) + r")\s+(?:image|images|photo|photos|picture|pictures|figure|figures)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        raw = match.group(1)
        value = int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw)
        if value:
            return max(1, min(value, 20))
    return None


def _wants_all_images(question: str) -> bool:
    return bool(re.search(r"\b(all|every)\s+(?:image|images|photo|photos|picture|pictures|figure|figures)\b", question.lower()))


def _limit_images(
    images: list[tuple[str, str]],
    question: str,
    default_limit: int | None = 6,
) -> list[tuple[str, str]]:
    limit = _requested_image_limit(question)
    if limit:
        return images[:limit]
    if default_limit and not _wants_all_images(question):
        return images[:default_limit]
    return images


def _apply_requested_image_limit(context: dict[str, list[Any]], question: str) -> dict[str, list[Any]]:
    limit = _requested_image_limit(question)
    if not limit:
        return context

    images = context.get("images", []) or []
    captions = context.get("image_captions", []) or []
    if len(images) <= limit:
        return context

    limited_context = dict(context)
    limited_context["images"] = images[:limit]
    limited_context["image_captions"] = captions[:limit]
    return limited_context


def _query_without_image_noise(question: str) -> str:
    cleaned = question.lower()
    cleaned = re.sub(r"\b(show\s+me|show|display|see|look\s+at|give\s+me|find|fetch|get)\b", " ", cleaned)
    cleaned = re.sub(r"\b(image\s+of|photo\s+of|picture\s+of|photograph\s+of|image|photo|picture|figure|photograph)\b", " ", cleaned)
    # Strip leftover leading articles / "me" after verb removal
    cleaned = re.sub(r"^\s*(me\b\s*)?(a\b\s*|an\b\s*|the\b\s*)?", "", cleaned)
    # Strip possessives and trailing punctuation left by pronoun resolution
    cleaned = re.sub(r"'s\b", "", cleaned)
    cleaned = re.sub(r"[.,!?;:]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or question


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _topic_tokens(text: str) -> set[str]:
    return {token for token in _tokenize(text) if token not in IMAGE_STOPWORDS}


def _entity_tokens(text: str) -> set[str]:
    return {token for token in _tokenize(text) if token not in IMAGE_ENTITY_STOPWORDS and len(token) > 5}


def _normalize_entity_aliases(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\bavlokiteshwar\b", "avalokiteshwar", normalized)
    normalized = re.sub(r"\bavalokiteswar\b", "avalokiteshwar", normalized)
    normalized = re.sub(r"\bavalokiteswara\b", "avalokiteshwar", normalized)
    normalized = re.sub(r"\bavalokiteshwar\b", "avalokiteshwar", normalized)
    normalized = re.sub(r"\bavalokitesvara\b", "avalokiteshwar", normalized)
    normalized = re.sub(r"\bavalokiteshvara\b", "avalokiteshwar", normalized)
    return normalized


def _score_summary(summary: str, question: str) -> int:
    summary_tokens = _topic_tokens(summary)
    question_tokens = _topic_tokens(question)
    return len(summary_tokens & question_tokens)


def _build_image_description(summary_text: str) -> str:
    summary_text = (summary_text or "").strip()
    if not summary_text:
        return ""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary_text) if s.strip()]
    if not sentences:
        return summary_text[:240]

    description_parts = [sentences[0]]
    credit_pattern = re.compile(r"\b(taken by|photo by|photographed by|image by|credited to|attribution to|copyright|courtesy of|photo credit)\b", re.I)
    for sentence in sentences[1:]:
        if credit_pattern.search(sentence):
            description_parts.append(sentence)
            break

    description = " ".join(description_parts).strip()
    if len(description) <= 320:
        return description
    return description[:320].rsplit(" ", 1)[0] + "..."


def _doc_content(doc: Any) -> str:
    return (getattr(doc, "text", "") or "").strip()


def _doc_metadata(doc: Any) -> dict[str, Any]:
    meta = getattr(doc, "metadata", {}) or {}
    return meta if isinstance(meta, dict) else {}


def _doc_kind(doc: Any) -> str:
    return getattr(doc, "kind", "") or str(_doc_metadata(doc).get("kind", ""))


def _doc_page_key(doc: Any) -> tuple[str, Any]:
    meta = _doc_metadata(doc)
    return str(meta.get("source", "")), meta.get("page_number")


def _score_context_text(doc: Any, question: str) -> int:
    text = _normalize_entity_aliases(_doc_content(doc))
    question_norm = _normalize_entity_aliases(question)
    question_entities = _entity_tokens(question_norm)
    text_tokens = _topic_tokens(text)
    question_tokens = _topic_tokens(question_norm)
    score = len(text_tokens & question_tokens)
    if question_entities:
        overlap = len(text_tokens & question_entities)
        score += overlap * 5
        if overlap:
            score += 10
    return score


def _select_image_from_context(docs: list[Any], question: str) -> tuple[str, str] | None:
    text_docs = [doc for doc in docs if _doc_kind(doc) != "image"]
    image_docs = [doc for doc in docs if _doc_kind(doc) == "image"]
    if not image_docs:
        return None

    best_text_doc = max(text_docs, key=lambda doc: _score_context_text(doc, question), default=None)
    if best_text_doc is None:
        return None

    best_text_score = _score_context_text(best_text_doc, question)
    if best_text_score <= 0:
        return None

    target_source, target_page = _doc_page_key(best_text_doc)
    matching_images = [doc for doc in image_docs if _doc_page_key(doc) == (target_source, target_page)]
    if not matching_images and target_source:
        matching_images = [doc for doc in image_docs if _doc_page_key(doc)[0] == target_source]
    if not matching_images:
        matching_images = image_docs

    chosen_image = matching_images[0]
    chosen_image_b64 = _doc_content(chosen_image)
    if not isinstance(chosen_image_b64, str) or not _is_renderable_image(chosen_image_b64):
        return None

    description = _build_image_description(_doc_content(best_text_doc))
    if not description:
        description = _build_image_description(_doc_content(chosen_image))
    return chosen_image_b64, description


def _context_text_blob(context: dict[str, list[Any]]) -> str:
    parts: list[str] = []
    for item in context.get("texts", []):
        if hasattr(item, "text"):
            parts.append(str(item.text))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _self_rag_verification(
    llm: Any,
    question: str,
    answer: str,
    context: dict[str, list[Any]],
    image_caption: str,
) -> dict[str, Any]:
    context_text = _context_text_blob(context)
    has_image = bool(context.get("images"))

    verification_prompt = (
        "You are a quality checker for a RAG chatbot about Nalanda Mahavihara.\n\n"
        f"Retrieved context from PDFs:\n{context_text[:1500] or '(none)'}\n\n"
        f"User question: {question}\n"
        f"Generated answer: {answer}\n"
        + (f"Image caption: {image_caption}\n" if has_image and image_caption else "")
        + "\nTasks:\n"
        "1. Is the answer grounded in (supported by) the retrieved context? Reply yes or no.\n"
        + ("2. Is the attached image relevant to the question? Reply yes or no.\n" if has_image else "")
        + "\nReply in this exact format:\n"
        "GROUNDED: yes/no\n"
        + ("KEEP_IMAGE: yes/no\n" if has_image else "")
        + "REASON: one short sentence"
    )

    try:
        result = llm.invoke([HumanMessage(content=verification_prompt)])
        text = result.content if hasattr(result, "content") else str(result)

        grounded = True
        keep_image = True
        reason = ""

        for line in text.splitlines():
            line = line.strip()
            lower = line.lower()
            if lower.startswith("grounded:"):
                grounded = "yes" in lower
            elif lower.startswith("keep_image:"):
                keep_image = "yes" in lower
            elif lower.startswith("reason:"):
                reason = line.split(":", 1)[-1].strip()

        return {"grounded": grounded, "keep_image": keep_image, "reason": reason}

    except Exception:
        # Token-matching fallback if LLM call fails
        normalized_question = _normalize_entity_aliases(question)
        normalized_caption = _normalize_entity_aliases(image_caption or "")
        question_entities = _entity_tokens(normalized_question)
        caption_entities = _entity_tokens(normalized_caption)
        context_tokens = _topic_tokens(_context_text_blob(context))
        answer_tokens = _topic_tokens(answer)

        grounded = bool(answer_tokens & context_tokens) or not answer.strip()
        keep_image = False
        reason = ""

        if context.get("images"):
            overlap = question_entities & caption_entities
            if overlap and (
                len(overlap) >= 2
                or "avalokiteshwar" in overlap
                or "bodhisattva" in overlap
                or "avalokitesvara" in overlap
            ):
                keep_image = True
                reason = "caption shares core entity with question"
            elif not question_entities and len(answer_tokens & context_tokens) >= 2:
                keep_image = True
                reason = "broad question with strong context overlap"

        return {"grounded": grounded, "keep_image": keep_image, "reason": reason}


def _is_renderable_image(b64_data: str) -> bool:
    try:
        raw = b64decode(b64_data, validate=True)
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
        return True
    except Exception:
        return False


def _select_best_image(
    store: MultimodalVectorStore,
    question: str,
    k: int,
    allowed_sources: set[str] | None = None,
    allow_weak: bool = False,
) -> tuple[str, str] | None:
    normalized_question = _normalize_entity_aliases(question)
    search_terms = [normalized_question]
    fallback_question = _query_without_image_noise(question)
    if fallback_question not in search_terms:
        search_terms.append(_normalize_entity_aliases(fallback_question))

    candidates: list[tuple[int, str, str]] = []
    seen_doc_ids: set[str] = set()
    search_k = max(k, 10)
    question_entities = _entity_tokens(normalized_question)

    for search_term in search_terms:
        try:
            ranked_docs = store.vectorstore.similarity_search_with_relevance_scores(search_term, k=search_k)
        except Exception:
            continue
        for summary_doc, relevance in ranked_docs:
            doc_id = summary_doc.metadata.get("doc_id") if hasattr(summary_doc, "metadata") else None
            source = summary_doc.metadata.get("source") if hasattr(summary_doc, "metadata") else None
            if allowed_sources and source and source not in allowed_sources:
                continue
            if allowed_sources and not source:
                continue
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            parent = store.docstore.mget([doc_id])[0]
            if hasattr(parent, "kind") and getattr(parent, "kind", None) != "image":
                continue
            image_b64 = parent.text if hasattr(parent, "text") else parent
            if not isinstance(image_b64, str) or not _is_renderable_image(image_b64):
                continue
            summary_text = (getattr(summary_doc, "page_content", "") or "").strip()
            if not summary_text:
                summary_text = str(summary_doc.metadata.get("summary", ""))
            summary_entities = _entity_tokens(_normalize_entity_aliases(summary_text))
            entity_overlap = bool(question_entities and (question_entities & summary_entities))
            if question_entities and not entity_overlap:
                if "avalokiteshwar" not in normalized_question and "bodhisattva" not in normalized_question:
                    continue

            caption = _build_image_description(summary_text)
            summary_score = _score_summary(summary_text, search_term)
            combined_score = int((relevance or 0) * 1000) + summary_score
            # allow weaker matches when explicitly requested (fallback)
            if allow_weak:
                min_relevance = 0.01
                min_score = 0
            else:
                min_relevance = 0.08 if entity_overlap else MIN_IMAGE_RELEVANCE
                min_score = 120 if entity_overlap else MIN_IMAGE_SCORE
            if (relevance or 0) < min_relevance:
                continue
            if combined_score < min_score:
                continue
            if summary_score < MIN_IMAGE_TOKEN_OVERLAP and not _is_image_request(question):
                continue
            candidates.append((combined_score, image_b64, caption))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    return best[1], best[2]


def _select_images_by_keyword(
    store: MultimodalVectorStore,
    question: str,
    allowed_sources: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Return all renderable images whose summary/source matches the key terms."""
    question_norm = _normalize_entity_aliases(_query_without_image_noise(question))
    question_tokens = _topic_tokens(question_norm)
    if not question_tokens:
        return []

    try:
        data = store.vectorstore.get(include=["documents", "metadatas"])
    except Exception:
        return []

    candidates: list[tuple[int, str, str]] = []
    seen_doc_ids: set[str] = set()
    ids = data.get("ids", []) or []
    documents = data.get("documents", []) or []
    metadatas = data.get("metadatas", []) or []

    for i in range(len(ids)):
        meta = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
        if meta.get("kind") != "image":
            continue

        source = str(meta.get("source", ""))
        if allowed_sources and source not in allowed_sources:
            continue

        summary = documents[i] if i < len(documents) and documents[i] else ""
        searchable = _normalize_entity_aliases(f"{summary} {source}")
        overlap = _topic_tokens(searchable) & question_tokens
        if not overlap:
            continue

        doc_id = meta.get("doc_id")
        if not doc_id or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)

        parent = store.docstore.mget([doc_id])[0]
        image_b64 = parent.text if hasattr(parent, "text") else parent
        if not isinstance(image_b64, str) or not _is_renderable_image(image_b64):
            continue

        score = len(overlap) * 10 + (20 if "avalokiteshwar" in overlap else 0)
        candidates.append((score, image_b64, _build_image_description(summary)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(image_b64, caption) for _, image_b64, caption in candidates]


def _caption_from_text_doc(doc: Any, question: str) -> str:
    text = _doc_content(doc)
    if not text:
        return ""

    tokens = _topic_tokens(_normalize_entity_aliases(_query_without_image_noise(question)))
    lowered = _normalize_entity_aliases(text)
    positions = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    if not positions:
        return _build_image_description(text)

    anchor = min(positions)
    prefix = text[max(0, anchor - 180) : anchor]
    annexure_pos = prefix.lower().rfind("annexure")
    excavation_pos = prefix.lower().rfind("excavation")
    local_anchor = max(annexure_pos, excavation_pos)
    if local_anchor >= 0:
        start = max(0, anchor - len(prefix) + local_anchor)
    else:
        start = max(0, anchor - 60)
    end = min(len(text), max(positions) + 260)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    return snippet[:320].rsplit(" ", 1)[0] + ("..." if len(snippet) > 320 else "")


def _select_images_near_text_docs(
    store: MultimodalVectorStore,
    text_docs: list[Any],
    question: str,
    page_window: int = 2,
) -> list[tuple[str, str]]:
    candidates: list[tuple[int, str, str]] = []
    seen: set[str] = set()

    # Topic tokens from the question — used to reject off-topic nearby images
    question_tokens = _topic_tokens(_query_without_image_noise(question))

    for text_doc in text_docs:
        text_meta = _doc_metadata(text_doc)
        source = str(text_meta.get("source", ""))
        page = text_meta.get("page_number")
        if not source or page is None:
            continue

        try:
            page_number = int(page)
        except Exception:
            continue

        text_caption = _caption_from_text_doc(text_doc, question)
        for doc_id, parent in store.docstore.store.items():
            if doc_id in seen or _doc_kind(parent) != "image":
                continue

            image_meta = _doc_metadata(parent)
            if str(image_meta.get("source", "")) != source:
                continue

            try:
                image_page = int(image_meta.get("page_number"))
            except Exception:
                continue

            distance = abs(image_page - page_number)
            if distance > page_window:
                continue

            image_b64 = _doc_content(parent)
            if not image_b64 or not _is_renderable_image(image_b64):
                continue

            # Reject images whose own summary has no topic overlap with the query.
            # Use ONLY the image's own metadata summary (not the text doc's caption) so that
            # e.g. a Bodhisattva statue is not returned for "Khilji" just because both are
            # in the same PDF. If the image has no summary, skip it when query is specific.
            image_summary = str(image_meta.get("summary", "")).strip()
            if question_tokens:
                if not image_summary:
                    continue  # no image summary → can't verify relevance → skip
                image_tokens = _topic_tokens(_normalize_entity_aliases(image_summary))
                if not (question_tokens & image_tokens):
                    continue

            seen.add(doc_id)
            caption = text_caption or f"Image from page {image_page} of {source}."
            candidates.append((100 - distance, image_b64, caption))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(image_b64, caption) for _, image_b64, caption in candidates]


def _select_best_text_docs(store: MultimodalVectorStore, question: str, k: int) -> list[Any]:
    normalized_question = _normalize_entity_aliases(question)
    search_terms = [normalized_question]
    fallback_question = _query_without_image_noise(question)
    if fallback_question not in search_terms:
        search_terms.append(_normalize_entity_aliases(fallback_question))

    candidates: list[tuple[int, Any]] = []
    seen_doc_ids: set[str] = set()
    search_k = max(k, 10)

    for search_term in search_terms:
        try:
            ranked_docs = store.vectorstore.similarity_search_with_relevance_scores(search_term, k=search_k)
        except Exception:
            continue
        for summary_doc, relevance in ranked_docs:
            doc_id = summary_doc.metadata.get("doc_id") if hasattr(summary_doc, "metadata") else None
            if not doc_id or doc_id in seen_doc_ids:
                continue
            parent = store.docstore.mget([doc_id])[0]
            if hasattr(parent, "kind") and getattr(parent, "kind", None) == "image":
                continue
            if parent is None:
                continue
            seen_doc_ids.add(doc_id)

            summary_text = (getattr(summary_doc, "page_content", "") or "").strip()
            if not summary_text:
                summary_text = _doc_content(parent)
            text_score = _score_context_text(parent, question)
            if text_score <= 0:
                continue
            combined_score = int((relevance or 0) * 1000) + text_score
            if combined_score <= 0:
                continue
            candidates.append((combined_score, parent))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[Any] = []
    for _, parent in candidates[: max(1, min(3, k))]:
        selected.append(parent)
    return selected


def _select_text_docs_by_keyword(store: MultimodalVectorStore, question: str, limit: int = 2) -> list[Any]:
    """Prefer parent chunks that literally contain the user's key terms."""
    question_norm = _normalize_entity_aliases(_query_without_image_noise(question))
    query_tokens = _topic_tokens(question_norm)
    if not query_tokens:
        return []

    candidates: list[tuple[int, Any]] = []
    for parent in store.docstore.store.values():
        if _doc_kind(parent) == "image":
            continue

        text = _doc_content(parent)
        if not text:
            continue

        text_tokens = _topic_tokens(_normalize_entity_aliases(text))
        overlap = text_tokens & query_tokens
        if not overlap:
            continue

        score = len(overlap) * 10
        text_norm = _normalize_entity_aliases(text)
        query_norm = _normalize_entity_aliases(_query_without_image_noise(question))
        if query_norm and query_norm in text_norm:
            score += 30
        for token in query_tokens:
            score += min(text_norm.count(token), 3) * 5
        if all(token in text_tokens for token in query_tokens):
            score += 25
        if "sarai" in overlap:
            score += 20
        if "mound" in overlap:
            score += 10
        candidates.append((score, parent))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [parent for _, parent in candidates[:limit]]


def _merge_docs(primary: list[Any], secondary: list[Any], limit: int) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for doc in primary + secondary:
        meta = _doc_metadata(doc)
        key = str(meta.get("doc_id") or id(doc))
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= limit:
            break
    return merged


def parse_docs(docs: list[Any]) -> dict[str, list[Any]]:
    """Split base64-encoded images and text/table parent documents."""
    images: list[str] = []
    texts: list[Any] = []
    for doc in docs:
        if hasattr(doc, "kind") and getattr(doc, "kind", None) == "image":
            image_b64 = getattr(doc, "text", "") or ""
            if isinstance(image_b64, str) and _is_renderable_image(image_b64):
                images.append(image_b64)
            continue
        if isinstance(doc, str):
            try:
                b64decode(doc, validate=True)
                images.append(doc)
                continue
            except Exception:
                pass
        texts.append(doc)
    return {"images": images, "texts": texts}


def build_multimodal_messages(
    context: dict[str, list[Any]],
    question: str,
    chat_history: str = "",
    web_docs: list[Any] | None = None,
    web_searched: bool = False,
) -> list[HumanMessage]:
    context_text = ""
    for text_element in context.get("texts", []):
        if hasattr(text_element, "text"):
            context_text += text_element.text
        elif isinstance(text_element, str):
            context_text += text_element

    prompt_template = ANSWER_PROMPT_TEMPLATE.format(
        context_text=context_text,
        question=question,
    )

    if web_searched:
        prompt_template = (
            "Note: The local documents did not contain sufficient information for this question. "
            "The following context was retrieved from the internet — use it to answer as best you can "
            "and mention that your answer is based on web sources.\n\n"
        ) + prompt_template

    # Inject admin-configured URL content directly into the prompt (ChatGPT/Claude browsing style).
    # URL content bypasses vector search — it is always included and the LLM reasons over it directly.
    if web_docs:
        web_sections: list[str] = []
        for doc in web_docs:
            src = (_doc_metadata(doc).get("source") or "web")
            text = (_doc_content(doc) or "").strip()
            if text:
                web_sections.append(f"[{src}]\n{text}")
        if web_sections:
            prompt_template += (
                "\n\n---\nAdditional web sources (admin-configured URLs):\n\n"
                + "\n\n".join(web_sections)
            )

    if chat_history:
        prompt_template = (
            "Conversation history for resolving follow-up references only:\n"
            f"{chat_history}\n\n"
            f"{prompt_template}"
        )
    prompt_content: list[dict[str, Any]] = [{"type": "text", "text": prompt_template}]
    for image in context.get("images", []):
        prompt_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
        )
    return [HumanMessage(content=prompt_content)]


def _run_retrieve(
    store: MultimodalVectorStore,
    question: str,
    chat_history: str,
    k: int,
    llm: Any = None,
) -> dict[str, Any]:
    # 1. Query rewriting + multi-query + HyDE (via LLM)
    if llm:
        queries = _rewrite_and_expand_queries(llm, question, chat_history)
    else:
        queries = [_retrieval_question(question, chat_history)]
    # queries = [original, QUERY1, QUERY2, HYDE]; use QUERY1 as the primary retrieval question
    retrieval_question = queries[1] if len(queries) > 1 else queries[0]

    # Image fast-path: keyword scan — skip HyDE (last query) to avoid false positives
    # from verbose hypothetical passages that contain generic terms like "buddhist"
    if _is_image_request(question):
        image_scan_queries = queries[:-1] if len(queries) > 2 else queries
        for q in image_scan_queries:
            keyword_images = _select_images_by_keyword(store, q)
            if keyword_images:
                limited_images = _limit_images(keyword_images, question)
                images = [image_b64 for image_b64, _ in limited_images]
                captions = [caption for _, caption in limited_images]
                return {
                    "retrieved_docs": images,
                    "context": {"images": images, "texts": [], "image_captions": captions},
                    "mode": "image",
                    "retrieval_question": retrieval_question,
                }

    # For pure image requests: skip text retrieval, reranking, and CRAG text search.
    # Go straight to image search → web image fallback. This avoids ~1-2s of wasted work.
    if _is_image_request(question):
        best = _select_best_image(store, retrieval_question, k)
        if not best:
            image_scan_queries = queries[:-1] if len(queries) > 2 else queries
            for q in image_scan_queries:
                keyword_images = _select_images_by_keyword(store, q)
                if keyword_images:
                    limited_images = _limit_images(keyword_images, question)
                    images = [image_b64 for image_b64, _ in limited_images]
                    captions = [caption for _, caption in limited_images]
                    return {
                        "retrieved_docs": images,
                        "context": {"images": images, "texts": [], "image_captions": captions},
                        "mode": "image",
                        "retrieval_question": retrieval_question,
                    }
        if best:
            image_b64, caption = best
            return {
                "retrieved_docs": [image_b64],
                "context": {"images": [image_b64], "texts": [], "image_captions": [caption]},
                "mode": "image",
                "retrieval_question": retrieval_question,
            }
        # No local image found — go to web
        if ENABLE_WEB_SEARCH:
            web_images = _web_image_search(retrieval_question)
            if web_images:
                images = [b64 for b64, _ in web_images]
                captions = [cap for _, cap in web_images]
                return {
                    "retrieved_docs": images,
                    "context": {"images": images, "texts": [], "image_captions": captions},
                    "mode": "image",
                    "retrieval_question": retrieval_question,
                    "web_searched": True,
                }
        return {
            "retrieved_docs": [],
            "context": {"images": [], "texts": [], "image_captions": []},
            "mode": "image",
            "retrieval_question": retrieval_question,
        }

    # 2. Multi-query text retrieval: keyword + semantic for every query variant
    all_text_docs: list[Any] = []
    seen_ids: set[str] = set()
    for query in queries:
        keyword_docs = _select_text_docs_by_keyword(store, query, limit=2)
        semantic_docs = _select_best_text_docs(store, query, k)
        for doc in keyword_docs + semantic_docs:
            doc_id = str(_doc_metadata(doc).get("doc_id") or id(doc))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_text_docs.append(doc)

    # 3. Re-rank the merged pool with a cross-encoder
    text_docs = _rerank_docs(question, all_text_docs, top_n=max(3, k))

    # 4. CRAG: if retrieval quality is poor, augment with web search
    if _retrieval_quality_score(text_docs) < 0.3:
        web_docs = _web_search(retrieval_question)
        for doc in web_docs:
            doc_id = str(id(doc))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                text_docs.append(doc)

    context = parse_docs(text_docs)
    text_sources = {
        str(getattr(doc, "metadata", {}).get("source", ""))
        for doc in text_docs
        if isinstance(getattr(doc, "metadata", {}), dict) and getattr(doc, "metadata", {}).get("source")
    }

    best = _select_best_image(store, retrieval_question, k, allowed_sources=text_sources or None)
    if not best:
        image_scan_queries = queries[:-1] if len(queries) > 2 else queries
        for q in image_scan_queries:
            keyword_images = _select_images_by_keyword(store, q, allowed_sources=text_sources or None)
            if keyword_images:
                limited_images = _limit_images(keyword_images, question)
                images = [image_b64 for image_b64, _ in limited_images]
                captions = [caption for _, caption in limited_images]
                return {
                    "retrieved_docs": images,
                    "context": {"images": images, "texts": [], "image_captions": captions},
                    "mode": "image",
                    "retrieval_question": retrieval_question,
                }
    # Augment text answer with a relevant local image if available
    if best:
        image_b64, caption = best
        context = {
            "images": [image_b64],
            "texts": context.get("texts", []),
            "image_captions": [caption] if caption else [],
        }
    else:
        context["images"] = []

    return {
        "retrieved_docs": text_docs,
        "context": context,
        "mode": "text",
        "retrieval_question": retrieval_question,
    }


def create_rag_graph(store: MultimodalVectorStore, top_k: int | None = None):
    """Build and compile a LangGraph RAG pipeline."""
    k = top_k or TOP_K
    retriever = store.retriever
    retriever.search_kwargs = {"k": max(k, 8)}
    llm = _build_llm()

    def retrieve_node(state: RAGState) -> dict[str, Any]:
        return _run_retrieve(store, state["question"], state.get("chat_history", ""), k, llm)

    def generate_node(state: RAGState) -> dict[str, str]:
        # If this was an image request but nothing was found, skip the LLM entirely
        if state.get("mode") == "image" and not state.get("context", {}).get("images"):
            return {"response": "No images matching your query were found in the indexed documents or on the web."}
        messages = build_multimodal_messages(
            state["context"],
            state["question"],
            state.get("chat_history", ""),
            web_docs=state.get("web_docs") or None,
            web_searched=state.get("web_searched", False),
        )
        result = llm.invoke(messages)
        content = result.content if hasattr(result, "content") else str(result)
        return {"response": content}

    def verify_node(state: RAGState) -> dict[str, Any]:
        context = dict(state.get("context", {}))
        images = context.get("images", []) or []
        captions = context.get("image_captions", []) or []
        answer = state.get("response", "")

        context_text = _context_text_blob(context)
        if not context_text.strip() and not images:
            if state.get("web_searched"):
                msg = "I searched both the indexed documents and the web but couldn't find relevant information about your question."
                reason = "no results in documents or web"
            else:
                msg = "I could not find relevant information in the indexed documents."
                reason = "no retrieved PDF text context"
            return {
                "verified": False,
                "verification_reason": reason,
                "context": context,
                "response": msg,
            }

        if state.get("mode") == "image":
            return {"verified": True, "verification_reason": "explicit image request", "context": context}

        verification = _self_rag_verification(
            llm,
            state["question"],
            answer,
            context,
            captions[0] if captions else "",
        )

        if not verification.get("keep_image", True):
            context["images"] = []
            context["image_captions"] = []

        if not verification.get("grounded", True):
            return {
                "verified": False,
                "verification_reason": verification.get("reason", ""),
                "context": context,
                "response": f"{answer}\n\n_Note: This answer may not be fully supported by the indexed documents._",
            }

        return {
            "verified": True,
            "verification_reason": verification.get("reason", ""),
            "context": context,
        }

    def iterate_node(state: RAGState) -> dict[str, Any]:
        """Generate a refined query from the unverified answer, retrieve again, merge context."""
        answer = state.get("response", "")
        question = state["question"]
        refine_prompt = (
            f"The following RAG answer was flagged as unverified:\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Generate a single refined search query to find better supporting evidence. "
            "Reply with ONLY the search query, nothing else."
        )
        try:
            res = llm.invoke([HumanMessage(content=refine_prompt)])
            refined_query = (res.content if hasattr(res, "content") else str(res)).strip()
        except Exception:
            refined_query = question

        new_retrieved = _run_retrieve(store, refined_query, question, k, llm)
        old_ctx = state.get("context", {})
        new_ctx = new_retrieved["context"]

        old_ids = {str(_doc_metadata(d).get("doc_id") or id(d)) for d in old_ctx.get("texts", [])}
        extra = [d for d in new_ctx.get("texts", []) if str(_doc_metadata(d).get("doc_id") or id(d)) not in old_ids]
        merged_texts = (old_ctx.get("texts", []) + extra)[:k]

        return {
            "context": {
                "texts": merged_texts,
                "images": old_ctx.get("images") or new_ctx.get("images", []),
                "image_captions": old_ctx.get("image_captions") or new_ctx.get("image_captions", []),
            },
            "hops": state.get("hops", 0) + 1,
            "mode": state.get("mode", "text"),
        }

    def web_search_node(state: RAGState) -> dict[str, Any]:
        query = state.get("retrieval_question") or state["question"]
        web_results = _web_search(query)
        old_ctx = state.get("context", {})
        merged_texts = (old_ctx.get("texts", []) + web_results)
        return {
            "context": {
                "texts": merged_texts,
                "images": old_ctx.get("images", []),
                "image_captions": old_ctx.get("image_captions", []),
            },
            "web_searched": True,
        }

    def should_continue(state: RAGState) -> str:
        if state.get("mode") == "image":
            return "end"
        if not state.get("verified", True):
            if not state.get("web_searched") and ENABLE_WEB_SEARCH:
                return "web_search"
            # Don't iterate after web search — it already tried the best fallback
            if not state.get("web_searched") and state.get("hops", 0) < MAX_ITERATIVE_HOPS:
                return "iterate"
        return "end"

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("iterate", iterate_node)
    graph.add_node("web_search", web_search_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify", should_continue, {"web_search": "web_search", "iterate": "iterate", "end": END}
    )
    graph.add_edge("web_search", "generate")
    graph.add_edge("iterate", "generate")
    return graph.compile()


def query_with_sources(
    store: MultimodalVectorStore,
    question: str,
    top_k: int | None = None,
    chat_history: str = "",
    web_docs: list[Any] | None = None,
) -> dict[str, Any]:
    """Invoke graph and return response + parsed context (notebook chain_with_sources shape)."""
    import asyncio, threading
    llm = _build_llm()

    # Synchronous domain check (run the async helper in a new event loop)
    def _sync_domain_check() -> bool:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_is_nalanda_domain(llm, question, chat_history))
        finally:
            loop.close()

    if not _sync_domain_check():
        return {
            "question": question,
            "retrieval_question": question,
            "response": _OUT_OF_DOMAIN_REPLY,
            "context": {"texts": [], "images": [], "image_captions": []},
            "mode": "text",
            "verified": True,
            "verification_reason": "out of domain",
            "web_searched": False,
        }

    graph = create_rag_graph(store, top_k=top_k)
    result = graph.invoke({
        "question": question,
        "chat_history": chat_history,
        "web_docs": web_docs or [],
    })
    context = _apply_requested_image_limit(result.get("context", {"texts": [], "images": []}), question)
    return {
        "question": question,
        "retrieval_question": result.get("retrieval_question", question),
        "response": result.get("response", ""),
        "context": context,
        "mode": result.get("mode", "text"),
        "verified": result.get("verified", True),
        "verification_reason": result.get("verification_reason", ""),
        "web_searched": result.get("web_searched", False),
    }


_OUT_OF_DOMAIN_REPLY = (
    "I'm the Nalanda Mahavihara assistant. I can only answer questions about "
    "Nalanda Mahavihara — its history, architecture, scholars, Buddhist heritage, "
    "excavations, UNESCO designation, and related topics. "
    "Please ask me something within that domain!"
)


async def _is_nalanda_domain(llm: Any, question: str, chat_history: str) -> bool:
    """Return True if the question is within the Nalanda Mahavihara knowledge domain."""
    import asyncio
    history_ctx = f"Conversation history:\n{chat_history}\n\n" if chat_history else ""
    prompt = (
        f"{history_ctx}"
        "You are a domain filter for a chatbot dedicated to Nalanda Mahavihara "
        "(the ancient Buddhist university in Bihar, India).\n\n"
        "Answer YES if the question — even without mentioning 'Nalanda' by name — "
        "could reasonably be answered by a Nalanda Mahavihara expert. This includes:\n"
        "- History, ruins, architecture, excavations, or UNESCO status of Nalanda\n"
        "- Buddhist monasteries, learning centers, or heritage in ancient India\n"
        "- People connected to Nalanda: scholars (Nagarjuna, Aryabhata, Xuanzang...), "
        "founders (Kumaragupta), destroyers (Bakhtiyar Khilji), patrons\n"
        "- Dynasties associated with Nalanda (Gupta, Pala, Harsha, etc.)\n"
        "- Ancient Indian education, religion, or history relevant to Nalanda's era\n"
        "- Image or photo requests for any of the above topics\n"
        "- Implicit references: 'the ancient Buddhist university', 'the place Khilji burned', "
        "'the university in Bihar', 'the site discovered by archaeologists' etc.\n"
        "- Any follow-up question that continues a Nalanda-related conversation in history\n\n"
        "Answer NO only if the question is clearly unrelated to Nalanda or its domain "
        "(e.g. cricket, cooking, modern technology, other countries' history unrelated to Nalanda).\n\n"
        "IMPORTANT: Image or photo requests about Nalanda, its ruins, scholars, or Buddhist heritage "
        "are ALWAYS YES — never reject a request just because it asks for an image.\n\n"
        "When in doubt, answer YES.\n\n"
        f"Question: {question}\n\n"
        "Reply ONLY with YES or NO."
    )
    try:
        result = await asyncio.to_thread(llm.invoke, [HumanMessage(content=prompt)])
        text = (result.content if hasattr(result, "content") else str(result)).strip().upper()
        return text.startswith("YES")
    except Exception:
        return True  # fail open: if check errors, allow the query


async def astream_rag_response(
    store: MultimodalVectorStore,
    question: str,
    top_k: int | None = None,
    chat_history: str = "",
    web_docs: list[Any] | None = None,
):
    """Async generator: yields {type:'token', content:'...'} chunks then a final {type:'done', ...} event."""
    import asyncio

    k = top_k or TOP_K
    llm = _build_llm()

    # Resolve pronouns first so domain guard and retrieval both see the concrete entity
    if chat_history:
        question = await asyncio.to_thread(_resolve_pronouns, llm, question, chat_history)

    # Domain guard: reject questions outside Nalanda Mahavihara scope
    if not await _is_nalanda_domain(llm, question, chat_history):
        yield {
            "type": "done",
            "answer": _OUT_OF_DOMAIN_REPLY,
            "images": [], "captions": [], "mode": "text",
            "verified": True, "verification_reason": "out of domain",
            "web_searched": False,
        }
        return

    retrieved = await asyncio.to_thread(_run_retrieve, store, question, chat_history, k, llm)
    context = retrieved["context"]
    mode = retrieved.get("mode", "text")
    pre_verified = retrieved.get("verified")
    pre_reason = retrieved.get("verification_reason", "")

    web_searched = retrieved.get("web_searched", False)
    if not _context_text_blob(context).strip() and not context.get("images"):
        if ENABLE_WEB_SEARCH:
            query = retrieved.get("retrieval_question") or question
            web_results = await asyncio.to_thread(_web_search, query)
            if web_results:
                context = {"texts": web_results, "images": [], "image_captions": []}
                web_searched = True
            else:
                yield {
                    "type": "done",
                    "answer": "I searched both the indexed documents and the web but couldn't find relevant information about your question.",
                    "images": [], "captions": [], "mode": mode,
                    "verified": False, "verification_reason": "no results in documents or web",
                    "web_searched": False,
                }
                return
        else:
            yield {
                "type": "done",
                "answer": "I could not find relevant information in the indexed documents.",
                "images": [], "captions": [], "mode": mode,
                "verified": False, "verification_reason": "no retrieved PDF text context",
                "web_searched": False,
            }
            return

    # Image request with no images found — don't call the LLM
    if mode == "image" and not context.get("images"):
        yield {
            "type": "done",
            "answer": "No images matching your query were found in the indexed documents or on the web.",
            "images": [], "captions": [], "mode": mode,
            "verified": False, "verification_reason": "no images found",
            "web_searched": web_searched,
        }
        return

    # Cross-check: verify LOCAL images match the query before showing them.
    # Web images (web_searched=True) already used the specific subject as their search query
    # so they are trusted and skip this check.
    if context.get("images") and not web_searched:
        import asyncio
        subject = _query_without_image_noise(question).strip() or question

        async def _check_image(b64: str) -> bool:
            check_prompt = (
                f"Does this image SPECIFICALLY and CLEARLY depict '{subject}'?\n"
                f"Reply YES only if the image is a direct, clear match for this subject.\n"
                f"Reply NO if the image is only loosely related, generic, or shows something else.\n"
                f"Reply only YES or NO."
            )
            try:
                check_msg = HumanMessage(content=[
                    {"type": "text", "text": check_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ])
                result = await asyncio.to_thread(llm.invoke, [check_msg])
                text = (result.content if hasattr(result, "content") else str(result)).strip().upper()
                return "YES" in text
            except Exception:
                return True  # fail open

        image_matches = await _check_image(context["images"][0])

        if not image_matches:
            if mode == "image":
                # Try web fallback before giving up
                retrieval_q = retrieved.get("retrieval_question") or question
                web_imgs = await asyncio.to_thread(_web_image_search, retrieval_q)
                if web_imgs:
                    context = dict(context)
                    context["images"] = [b64 for b64, _ in web_imgs]
                    context["image_captions"] = [cap for _, cap in web_imgs]
                    web_searched = True
                    # Web image was searched with the specific subject — trust it, no re-check
                else:
                    # No web image either — give up
                    context = dict(context)
                    context["images"] = []
                    context["image_captions"] = []
                    yield {
                        "type": "done",
                        "answer": f"No relevant images of '{subject}' were found in the knowledge base or on the web.",
                        "images": [], "captions": [], "mode": mode,
                        "verified": False, "verification_reason": "image mismatch, no web fallback",
                        "web_searched": False,
                    }
                    return
            else:
                # Text mode: drop the irrelevant image, continue with text answer
                context = dict(context)
                context["images"] = []
                context["image_captions"] = []

    messages = build_multimodal_messages(context, question, chat_history, web_docs=web_docs, web_searched=web_searched)
    full_response = ""

    async for chunk in llm.astream(messages):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full_response += token
            yield {"type": "token", "content": token}

    images = list(context.get("images", []) or [])
    captions = list(context.get("image_captions", []) or [])

    if pre_verified is not None:
        verified = pre_verified
        reason = pre_reason
    elif mode == "image":
        verified = True
        reason = "explicit image request"
    else:
        verification = await asyncio.to_thread(
            _self_rag_verification, llm, question, full_response, context,
            captions[0] if captions else "",
        )
        if not verification.get("keep_image", True):
            images = []
            captions = []
        verified = bool(verification.get("grounded", True))
        reason = verification.get("reason", "")

        # Internet fallback: if answer is ungrounded, search the web and regenerate
        if not verified and ENABLE_WEB_SEARCH:
            query = retrieved.get("retrieval_question") or question
            web_results = await asyncio.to_thread(_web_search, query)
            if web_results:
                web_searched = True
                context = dict(context)
                context["texts"] = (context.get("texts", []) + web_results)
                web_messages = build_multimodal_messages(
                    context, question, chat_history, web_docs=web_docs, web_searched=True
                )
                full_response = ""
                async for chunk in llm.astream(web_messages):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        full_response += token
                        yield {"type": "token", "content": token}
                verified = True
                reason = "answer sourced from web search"

        if not verified:
            full_response += "\n\n_Note: This answer may not be fully supported by the indexed documents._"

    final_context = _apply_requested_image_limit({"images": images, "image_captions": captions}, question)

    yield {
        "type": "done",
        "answer": full_response,
        "images": final_context.get("images", []),
        "captions": final_context.get("image_captions", []),
        "mode": mode,
        "verified": verified,
        "verification_reason": reason,
        "web_searched": web_searched,
    }
