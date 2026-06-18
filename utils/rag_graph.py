"""
LangGraph orchestration for multimodal RAG (notebook retrieve → parse → answer flow).
"""

from __future__ import annotations

from base64 import b64decode
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from utils.config import ANSWER_PROMPT_TEMPLATE, OPENAI_API_KEY, OPENAI_MODEL, TOP_K
from utils.vectorstore import MultimodalVectorStore


class RAGState(TypedDict, total=False):
    question: str
    retrieved_docs: list[Any]
    context: dict[str, list[Any]]
    response: str


def parse_docs(docs: list[Any]) -> dict[str, list[Any]]:
    """Split base64-encoded images and text/table parent documents."""
    images: list[str] = []
    texts: list[Any] = []
    for doc in docs:
        if isinstance(doc, str):
            try:
                b64decode(doc, validate=True)
                images.append(doc)
                continue
            except Exception:
                pass
        texts.append(doc)
    return {"images": images, "texts": texts}


def build_multimodal_messages(context: dict[str, list[Any]], question: str) -> list[HumanMessage]:
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
    retriever.search_kwargs = {"k": k}
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)

    def retrieve_node(state: RAGState) -> dict[str, Any]:
        docs = retriever.invoke(state["question"])
        return {"retrieved_docs": docs, "context": parse_docs(docs)}

    def generate_node(state: RAGState) -> dict[str, str]:
        messages = build_multimodal_messages(state["context"], state["question"])
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm
        result = chain.invoke({})
        content = result.content if hasattr(result, "content") else str(result)
        return {"response": content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def query_with_sources(store: MultimodalVectorStore, question: str, top_k: int | None = None) -> dict[str, Any]:
    """Invoke graph and return response + parsed context (notebook chain_with_sources shape)."""
    graph = create_rag_graph(store, top_k=top_k)
    result = graph.invoke({"question": question})
    return {
        "question": question,
        "response": result.get("response", ""),
        "context": result.get("context", {"texts": [], "images": []}),
    }
