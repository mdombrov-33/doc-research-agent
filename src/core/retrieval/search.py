from opentelemetry import trace
from qdrant_client import models

from src.config import Settings, get_settings
from src.core.nlp import extract_entities
from src.core.retrieval.rerank import rerank
from src.core.vectorstore import get_vector_store
from src.utils.logger import logger

_tracer = trace.get_tracer(__name__)


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


def _fetch_k(top_k: int, settings: Settings) -> int:
    """Candidate pool size to retrieve before reranking: a multiple of the user's top_k,
    capped. Reranking off → fetch exactly top_k (the original behaviour)."""
    if not settings.RERANK_ENABLED:
        return top_k
    return max(top_k, min(top_k * settings.RERANK_MULTIPLIER, settings.RERANK_FETCH_CAP))


def hybrid_search(question: str, top_k: int) -> list[dict]:
    """Retrieve documents for a query: entity-filtered hybrid search, then cross-encoder rerank.

    Hybrid = dense + BM25 sparse, fused server-side by Qdrant (RRF). We pull a wide candidate
    pool (fetch_k) so the reranker has room to promote buried hits, then keep the user's top_k.
    Returns documents in final ranked order.
    """
    settings = get_settings()
    vector_store = get_vector_store()
    fetch_k = _fetch_k(top_k, settings)

    query_entities = extract_entities(question)
    entity_filter = _entity_filter(query_entities)

    logger.info(
        "retrieval_start",
        question=question,
        top_k=top_k,
        fetch_k=fetch_k,
        query_entities=query_entities or None,
        entity_filter_active=entity_filter is not None,
    )

    with _tracer.start_as_current_span("retrieval.hybrid_search") as span:
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.fetch_k", fetch_k)
        span.set_attribute("retrieval.entity_filter_active", entity_filter is not None)

        with _tracer.start_as_current_span("retrieval.qdrant_query"):
            results = vector_store.similarity_search_with_score(
                question, k=fetch_k, filter=entity_filter
            )

        entity_fallback = bool(entity_filter) and not results
        if entity_fallback:
            logger.info("entity_filter_fallback", reason="zero_results")
            with _tracer.start_as_current_span("retrieval.qdrant_query_fallback"):
                results = vector_store.similarity_search_with_score(question, k=fetch_k)
        elif entity_filter and len(results) < fetch_k:
            logger.warning(
                "entity_filter_narrow",
                results_returned=len(results),
                fetch_k=fetch_k,
                query_entities=query_entities,
            )

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

        empty_filtered = len(results) - len(doc_items)

        score_stats = {}
        if scores:
            score_stats = {
                "score_min": round(min(scores), 4),
                "score_max": round(max(scores), 4),
                "score_mean": round(sum(scores) / len(scores), 4),
            }

        span.set_attribute("retrieval.docs_found", len(doc_items))
        span.set_attribute("retrieval.entity_fallback", entity_fallback)

        logger.info(
            "docs_retrieved",
            count=len(doc_items),
            fetch_k=fetch_k,
            empty_filtered=empty_filtered,
            query_entities=query_entities or None,
            entity_filtered=bool(entity_filter) and not entity_fallback,
            entity_fallback=entity_fallback,
            **score_stats,
        )

        # Rerank the candidate pool with a cross-encoder and keep the user's top_k. Disabled →
        # fetch_k already equals top_k, so the trim is a defensive no-op preserving the contract.
        if settings.RERANK_ENABLED and doc_items:
            return rerank(question, doc_items, top_k)
        return doc_items[:top_k]
