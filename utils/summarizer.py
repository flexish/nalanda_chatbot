"""Generate summaries for text, tables, and images (notebook pattern)."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    IMAGE_SUMMARY_PROMPT,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SUMMARIZE_CONCURRENCY,
    TEXT_SUMMARY_PROMPT,
)
from utils.ingest import ParentDocument


def _text_summarizer():
    prompt = ChatPromptTemplate.from_template(TEXT_SUMMARY_PROMPT)
    if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY)
    elif GROQ_API_KEY:
        from langchain_groq import ChatGroq
        model = ChatGroq(temperature=0.5, model=GROQ_MODEL, api_key=GROQ_API_KEY)
    else:
        model = ChatOpenAI(temperature=0.5, model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    return {"element": lambda x: x} | prompt | model | StrOutputParser()


def _image_summarizer():
    messages = [
        (
            "user",
            [
                {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY)
    else:
        model = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    return prompt | model | StrOutputParser()


def summarize_texts(
    texts: List[ParentDocument],
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[str]:
    if not texts:
        return []
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
    if on_progress:
        on_progress(f"Summarizing {len(images_b64)} images (vision model)...")
    chain = _image_summarizer()
    image_payloads = [
        image.text if isinstance(image, ParentDocument) else image
        for image in images_b64
    ]
    return chain.batch(image_payloads, {"max_concurrency": 2})
