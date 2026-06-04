from typing import Any

from qdrant_client import models

from src.core.document_processing.text_processor import extract_entities
from src.core.grading.graders import (
    grade_documents_batch,
    route_and_rewrite,
)
from src.core.llm import get_llm
from src.core.state import AgentState
from src.core.tools import get_vector_store_tool, get_web_search_tool
from src.prompts import GENERATION_SYSTEM_PROMPT, GENERATION_USER_PROMPT
from src.utils.logger import logger


def router_node(state: AgentState) -> dict[str, Any]:
    question = state.get("question", "")
    result = route_and_rewrite(question)

    web_search = result.datasource == "websearch"

    logger.info(
        "route_decision",
        destination=result.datasource,
        query=result.rewritten_query,
        web_search=web_search,
    )

    return {
        "web_search": web_search,
        "web_search_done": False,
        "web_fallback_needed": False,
        "question": result.rewritten_query,
        "raw_documents": None,
        "docs_retrieved_total": None,
    }


def _entity_filter(entities: list[str]) -> models.Filter | None:
    """Filter to chunks whose stored entities overlap the query's entities."""
    if not entities:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.entities",
                match=models.MatchAny(any=entities),
            )
        ]
    )


def retrieve_node(state: AgentState) -> dict[str, Any]:
    question = state.get("question", "")
    top_k = state.get("top_k") or 5
    vector_store = get_vector_store_tool()

    # Entity-aware retrieval: pull named entities from the query and use them to
    # filter on the entities extracted at ingestion. Falls back to unfiltered
    # hybrid search if the query has no entities or the filter matches nothing.
    query_entities = extract_entities(question)
    entity_filter = _entity_filter(query_entities)

    # Hybrid search: dense + BM25 sparse, fused server-side by Qdrant (RRF).
    results = vector_store.similarity_search_with_score(question, k=top_k, filter=entity_filter)

    entity_fallback = bool(entity_filter) and not results
    if entity_fallback:
        results = vector_store.similarity_search_with_score(question, k=top_k)

    doc_items = []
    scores = []

    for doc, score in results:
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        if not content.strip():
            continue
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        doc_items.append(
            {
                "content": content,
                "filename": metadata.get("filename", "unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "chunk_length": metadata.get("chunk_length", len(content)),
                "source": "vectorstore",
            }
        )
        scores.append(float(score))

    empty_filtered = len(results) - len(doc_items)
    docs_retrieved_total = len(doc_items)

    score_stats = {}
    if scores:
        score_stats = {
            "score_min": round(min(scores), 4),
            "score_max": round(max(scores), 4),
            "score_mean": round(sum(scores) / len(scores), 4),
        }

    logger.info(
        "docs_retrieved",
        count=docs_retrieved_total,
        empty_filtered=empty_filtered,
        query_entities=query_entities or None,
        entity_filtered=bool(entity_filter) and not entity_fallback,
        entity_fallback=entity_fallback,
        **score_stats,
    )

    return {"raw_documents": doc_items, "docs_retrieved_total": docs_retrieved_total}


def web_search_node(state: AgentState) -> dict[str, Any]:
    question = state.get("question", "")
    web_search = get_web_search_tool()

    web_docs: list[dict] = []
    try:
        result = web_search.invoke(question)
        content = str(result)
        web_docs = [
            {
                "content": content,
                "filename": "web",
                "chunk_index": 0,
                "chunk_length": len(content),
                "source": "web",
            }
        ]
        logger.info("web_search_complete", count=len(web_docs))
    except Exception as e:
        logger.error("web_search_failed", error=str(e))

    return {
        "raw_documents": web_docs,
        "docs_retrieved_total": len(web_docs),
        "web_search_done": True,
    }


def grade_documents_node(state: AgentState) -> dict[str, Any]:
    question = state.get("question", "")
    documents = state.get("raw_documents", [])

    if not documents:
        logger.info("docs_graded", total=0, relevant=0)
        return {"documents": []}

    contents = [doc["content"] for doc in documents]
    scores = grade_documents_batch(question, contents)
    filtered_docs = [doc for doc, score in zip(documents, scores) if score == "yes"]

    logger.info("docs_graded", total=len(documents), relevant=len(filtered_docs))

    if len(filtered_docs) < 2 and not state.get("web_search_done", False):
        logger.info("web_fallback_triggered", relevant_docs=len(filtered_docs))
        return {"documents": filtered_docs, "raw_documents": None, "web_fallback_needed": True}

    if state.get("web_search_done", False):
        existing_docs = state.get("documents", [])
        merged = existing_docs + filtered_docs
        logger.info(
            "fallback_grading_complete",
            existing=len(existing_docs),
            new_web=len(filtered_docs),
            total=len(merged),
        )
        return {"documents": merged, "web_fallback_needed": False}

    return {"documents": filtered_docs, "web_fallback_needed": False}


def generate_node(state: AgentState) -> dict[str, Any]:
    question = state.get("question", "")
    documents = state.get("documents", [])
    chat_history = state.get("chat_history", [])

    context_parts = []
    for doc in documents:
        label = (
            "[Web Search]"
            if doc.get("source") == "web"
            else f"[Document: {doc.get('filename', 'unknown')}]"
        )
        context_parts.append(f"{label}\n{doc['content']}")
    context = "\n\n".join(context_parts)

    llm = get_llm(state.get("model"), temperature=0.7)

    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT.format(context=context)},
        *chat_history,
        {"role": "user", "content": GENERATION_USER_PROMPT.format(question=question)},
    ]

    response = llm.invoke(messages)
    generation = response.content if isinstance(response.content, str) else str(response.content)

    # Empty generations are a probabilistic failure; one retry usually fixes it.
    if not generation.strip():
        logger.info("generation_empty_retry")
        response = llm.invoke(messages)
        generation = (
            response.content if isinstance(response.content, str) else str(response.content)
        )

    logger.info("answer_generated", chars=len(generation))

    updated_history = chat_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": generation},
    ]

    return {
        "generation": generation,
        "chat_history": updated_history,
    }
