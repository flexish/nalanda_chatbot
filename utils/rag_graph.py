"""
LangGraph orchestration for multimodal RAG (notebook retrieve → parse → answer flow).
"""

from __future__ import annotations

from base64 import b64decode
import io
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from PIL import Image

from utils.config import (
    ANSWER_PROMPT_TEMPLATE,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
)
from utils.vectorstore import MultimodalVectorStore


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
    cleaned = re.sub(r"\b(show|display|see|look at|give me|find|fetch)\b", " ", cleaned)
    cleaned = re.sub(r"\b(image|photo|picture|figure|picture of|image of|photo of)\b", " ", cleaned)
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
    llm: ChatOpenAI,
    question: str,
    answer: str,
    context: dict[str, list[Any]],
    image_caption: str,
) -> dict[str, Any]:
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


def create_rag_graph(store: MultimodalVectorStore, top_k: int | None = None):
    """Build and compile a LangGraph RAG pipeline."""
    k = top_k or TOP_K
    retriever = store.retriever
    retriever.search_kwargs = {"k": max(k, 8)}
    if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY)
    else:
        llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)

    def retrieve_node(state: RAGState) -> dict[str, Any]:
        question = state["question"]
        chat_history = state.get("chat_history", "")
        retrieval_question = _retrieval_question(question, chat_history)
        mode = "text"

        if _is_image_request(question):
            keyword_images = _select_images_by_keyword(store, retrieval_question)
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

        keyword_limit = 8 if _is_image_request(question) else 2
        keyword_text_docs = _select_text_docs_by_keyword(store, retrieval_question, limit=keyword_limit)
        semantic_text_docs = _select_best_text_docs(store, retrieval_question, k)
        text_docs = _merge_docs(keyword_text_docs, semantic_text_docs, limit=max(3, k))
        context = parse_docs(text_docs)

        text_sources = {
            str(getattr(doc, "metadata", {}).get("source", ""))
            for doc in text_docs
            if isinstance(getattr(doc, "metadata", {}), dict) and getattr(doc, "metadata", {}).get("source")
        }

        if _is_image_request(question):
            nearby_images = _select_images_near_text_docs(store, text_docs, retrieval_question)
            if nearby_images:
                limited_images = _limit_images(nearby_images, question)
                images = [image_b64 for image_b64, _ in limited_images]
                captions = [caption for _, caption in limited_images]
                return {
                    "retrieved_docs": images,
                    "context": {"images": images, "texts": text_docs, "image_captions": captions},
                    "mode": "image",
                    "retrieval_question": retrieval_question,
                    "verified": True,
                    "verification_reason": "image matched by nearby PDF text/page label",
                }

        best = _select_best_image(store, retrieval_question, k, allowed_sources=text_sources or None)
        # If user explicitly asked to show an image and no strict match found,
        # try a relaxed/fallback search to prefer showing something rather than nothing.
        if not best and _is_image_request(question):
            best = _select_best_image(store, retrieval_question, k, allowed_sources=text_sources or None, allow_weak=True)
        if not best and _is_image_request(question):
            keyword_images = _select_images_by_keyword(store, retrieval_question, allowed_sources=text_sources or None)
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
            if not context.get("texts"):
                context["texts"] = [caption] if caption else []
            context = {
                "images": [image_b64],
                "texts": context.get("texts", []),
                "image_captions": [caption] if caption else [],
            }
        else:
            context["images"] = []

        if _is_image_request(question):
            if best:
                image_b64, caption = best
                mode = "image"
                return {
                    "retrieved_docs": [image_b64],
                    "context": {"images": [image_b64], "texts": [], "image_captions": [caption]},
                    "mode": mode,
                }

        return {
            "retrieved_docs": text_docs,
            "context": context,
            "mode": mode,
            "retrieval_question": retrieval_question,
        }

    def generate_node(state: RAGState) -> dict[str, str]:
        if state.get("mode") == "image" and state.get("context", {}).get("images"):
            # For image mode we skip LLM text generation — response will be handled by UI using image + caption
            return {"response": ""}
        messages = build_multimodal_messages(
            state["context"],
            state["question"],
            state.get("chat_history", ""),
        )
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm
        result = chain.invoke({})
        content = result.content if hasattr(result, "content") else str(result)
        return {"response": content}

    def verify_node(state: RAGState) -> dict[str, Any]:
        context = dict(state.get("context", {}))
        images = context.get("images", []) or []
        captions = context.get("image_captions", []) or []
        answer = state.get("response", "")
        if not images:
            context_tokens = _topic_tokens(_context_text_blob(context))
            answer_tokens = _topic_tokens(answer)
            if not context_tokens:
                return {
                    "verified": False,
                    "verification_reason": "no retrieved PDF text context",
                    "context": context,
                    "response": "I could not verify this from the indexed PDF context.",
                }
            if answer.strip() and not (answer_tokens & context_tokens):
                return {
                    "verified": False,
                    "verification_reason": "answer has weak overlap with retrieved PDF context",
                    "context": context,
                    "response": (
                        f"{answer}\n\n"
                        "Verification note: this answer has weak overlap with the retrieved PDF context."
                    ),
                }
            return {"verified": True, "verification_reason": "grounded in retrieved PDF text", "context": context}

        if state.get("mode") == "image":
            # Explicit image requests should keep the selected knowledge-base image.
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
        return {
            "verified": bool(verification.get("grounded", True)),
            "verification_reason": verification.get("reason", ""),
            "context": context,
        }

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


def query_with_sources(
    store: MultimodalVectorStore,
    question: str,
    top_k: int | None = None,
    chat_history: str = "",
) -> dict[str, Any]:
    """Invoke graph and return response + parsed context (notebook chain_with_sources shape)."""
    graph = create_rag_graph(store, top_k=top_k)
    result = graph.invoke({"question": question, "chat_history": chat_history})
    context = _apply_requested_image_limit(result.get("context", {"texts": [], "images": []}), question)
    return {
        "question": question,
        "retrieval_question": result.get("retrieval_question", question),
        "response": result.get("response", ""),
        "context": context,
        "mode": result.get("mode", "text"),
        "verified": result.get("verified", True),
        "verification_reason": result.get("verification_reason", ""),
    }
