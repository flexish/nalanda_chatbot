"""Generate summaries for text, tables, and images (notebook pattern)."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    IMAGE_SUMMARY_PROMPT,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    SUMMARIZE_CONCURRENCY,
    SUMMARIZE_TEXT,
    TEXT_SUMMARY_PROMPT,
)
from utils.ingest import ParentDocument


def _text_summarizer():
    prompt = ChatPromptTemplate.from_template(TEXT_SUMMARY_PROMPT)
    if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY)
    elif LLM_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        model = ChatOpenAI(temperature=0.5, model=OPENROUTER_MODEL, api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    else:
        model = ChatOpenAI(temperature=0.5, model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    return {"element": lambda x: x} | prompt | model | StrOutputParser()


def _image_model():
    if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY)
    elif LLM_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        return ChatOpenAI(model=OPENROUTER_MODEL, api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    elif OPENAI_API_KEY:
        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    return None


def _build_image_message(b64: str):
    from langchain_core.messages import HumanMessage
    if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        content = [
            {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
        ]
    else:
        content = [
            {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
    return [HumanMessage(content=content)]


def summarize_texts(
    texts: List[ParentDocument],
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[str]:
    if not texts:
        return []
    if not SUMMARIZE_TEXT:
        return [t.text[:500] for t in texts]
    if on_progress:
        on_progress(f"Summarizing {len(texts)} text chunks...")
    elements = [t.text for t in texts]
    return _text_summarizer().batch(elements, {"max_concurrency": SUMMARIZE_CONCURRENCY})


def summarize_tables(
    tables: List[ParentDocument],
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[str]:
    if not tables:
        return []
    if not SUMMARIZE_TEXT:
        return [(t.metadata.get("text_as_html") or t.text)[:500] for t in tables]
    if on_progress:
        on_progress(f"Summarizing {len(tables)} tables...")
    elements = [
        t.metadata.get("text_as_html") or t.text
        for t in tables
    ]
    return _text_summarizer().batch(elements, {"max_concurrency": SUMMARIZE_CONCURRENCY})


def summarize_images(
    images_b64: Sequence[str | ParentDocument],
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[str]:
    if not images_b64:
        return []
    model = _image_model()
    if model is None:
        if on_progress:
            on_progress(f"Skipping {len(images_b64)} image(s) — no vision-capable provider configured.")
        return ["Image extracted from document (vision summarization unavailable)."] * len(images_b64)
    if on_progress:
        on_progress(f"Summarizing {len(images_b64)} images (vision model)...")
    chain = model | StrOutputParser()
    payloads = [
        image.text if isinstance(image, ParentDocument) else image
        for image in images_b64
    ]
    messages = [_build_image_message(b64) for b64 in payloads]
    try:
        return chain.batch(messages, {"max_concurrency": 2})
    except Exception as e:
        if on_progress:
            on_progress(f"Vision summarization failed ({e}). Storing images without captions.")
        return ["Image extracted from document (vision summarization unavailable)."] * len(images_b64)
