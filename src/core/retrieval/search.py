import random
import time
from collections.abc import Sequence
from typing import Any

import httpx
from opentelemetry import trace
from qdrant_client import models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.config import Settings, get_settings
from src.core.nlp import extract_entities
from src.core.retrieval.rerank import rerank
from src.core.vectorstore import get_retrieval_vector_store
from src.utils.logger import logger

_tracer = trace.get_tracer(__name__)
_TRANSIENT_QDRANT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def _entity_filter(entities: list[str]) -> models.Filter | None:
    """Match chunks whose stored entities overlap the query's entities."""
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


def _merge_results(
    primary: Sequence[tuple[Any, float]], supplemental: Sequence[tuple[Any, float]]
) -> list[tuple[Any, float]]:
    """Append entity matches without allowing duplicate chunks into the reranker."""
    merged = []
    seen = set()
    for doc, score in [*primary, *supplemental]:
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        chunk_id = metadata.get("chunk_id")
        key = chunk_id if chunk_id is not None else id(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append((doc, score))
    return merged


def _fetch_k(top_k: int, settings: Settings) -> int:
    """Candidate pool size to retrieve before reranking: a multiple of the user's top_k,
    capped. Reranking off → fetch exactly top_k (the original behaviour)."""
    if not settings.RERANK_ENABLED:
        return top_k
    return max(top_k, min(top_k * settings.RERANK_MULTIPLIER, settings.RERANK_FETCH_CAP))


def _is_transient_qdrant_error(error: Exception) -> bool:
    if isinstance(error, ResourceExhaustedResponse):
        return True
    if isinstance(error, ResponseHandlingException):
        return isinstance(
            error.source,
            (httpx.TimeoutException, httpx.NetworkError, httpx.ProxyError),
        )
    return (
        isinstance(error, UnexpectedResponse)
        and error.status_code in _TRANSIENT_QDRANT_STATUS_CODES
    )


def _qdrant_search(vector_store: Any, question: str, **kwargs: Any) -> list[tuple[Any, float]]:
    for attempt in range(1, 3):
        try:
            return vector_store.similarity_search_with_score(question, **kwargs)
        except Exception as error:
            if attempt == 2 or not _is_transient_qdrant_error(error):
                raise
            status_code = error.status_code if isinstance(error, UnexpectedResponse) else None
            logger.warning(
                "qdrant_query_retry",
                attempt=attempt,
                failure_type=type(error).__name__,
                status_code=status_code,
            )
            time.sleep(random.uniform(0.05, 0.25))
    raise AssertionError("unreachable")


def hybrid_search(question: str, top_k: int) -> list[dict]:
    """Retrieve documents for a query: hybrid candidates plus entity matches, then rerank.

    Hybrid = dense + BM25 sparse, fused server-side by Qdrant (RRF). We pull a wide candidate
    pool (fetch_k) so the reranker has room to promote buried hits. Entity matches supplement
    that pool but never exclude it, then the reranker keeps the user's top_k. Returns documents
    in final ranked order.
    """
    settings = get_settings()
    vector_store = get_retrieval_vector_store()
    fetch_k = _fetch_k(top_k, settings)

    query_entities = extract_entities(question)
    entity_filter = _entity_filter(query_entities)

    logger.info(
        "retrieval_start",
        top_k=top_k,
        fetch_k=fetch_k,
        query_entity_count=len(query_entities),
        entity_supplement_active=bool(entity_filter) and settings.RERANK_ENABLED,
    )

    with _tracer.start_as_current_span("retrieval.hybrid_search") as span:
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.fetch_k", fetch_k)
        span.set_attribute(
            "retrieval.entity_supplement_active", bool(entity_filter) and settings.RERANK_ENABLED
        )

        with _tracer.start_as_current_span("retrieval.qdrant_query"):
            primary_results = _qdrant_search(vector_store, question, k=fetch_k)

        entity_results = []
        if entity_filter and settings.RERANK_ENABLED:
            with _tracer.start_as_current_span("retrieval.qdrant_entity_supplement"):
                entity_results = _qdrant_search(
                    vector_store, question, k=top_k, filter=entity_filter
                )
        results = _merge_results(primary_results, entity_results)

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
                    "document_id": metadata.get("document_id"),
                    "filename": metadata.get("filename", "unknown"),
                    "chunk_id": metadata.get("chunk_id"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "chunk_length": metadata.get("chunk_length", len(content)),
                    "page": metadata.get("page"),
                    "source": "document",
                }
            )
            scores.append(float(score))

        blank_results = len(results) - len(doc_items)

        score_stats = {}
        if scores:
            score_stats = {
                "score_min": round(min(scores), 4),
                "score_max": round(max(scores), 4),
                "score_mean": round(sum(scores) / len(scores), 4),
            }

        span.set_attribute("retrieval.docs_found", len(doc_items))
        span.set_attribute("retrieval.entity_supplement_count", len(entity_results))

        logger.info(
            "docs_retrieved",
            count=len(doc_items),
            fetch_k=fetch_k,
            blank_results=blank_results,
            query_entity_count=len(query_entities),
            entity_supplement_count=len(entity_results),
            **score_stats,
        )

        # Rerank the candidate pool with a cross-encoder and keep the user's top_k. Disabled →
        # fetch_k already equals top_k, so the trim is a defensive no-op preserving the contract.
        if settings.RERANK_ENABLED and doc_items:
            return rerank(question, doc_items, top_k)
        return doc_items[:top_k]
