import random
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import httpx
from opentelemetry import trace
from qdrant_client import models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.config import get_settings
from src.core.evidence_observability import evidence_log_fields, text_log_fields
from src.core.nlp import extract_entities
from src.core.retrieval.rerank import rerank
from src.core.vectorstore import get_retrieval_vector_store
from src.utils.logger import logger

_tracer = trace.get_tracer(__name__)
_TRANSIENT_QDRANT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_MAX_SEARCH_QUERIES = 2
_RRF_RANK_CONSTANT = 60
_BRANCH_RANKS_KEY = "_retrieval_branch_ranks"


@dataclass(frozen=True)
class RetrievalMetrics:
    query_count: int
    query_shape: str
    candidates: int
    evidence: int
    reranker_calls: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[dict]
    metrics: RetrievalMetrics


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


def _chunk_identity(doc: dict) -> str:
    """Return a stable identity for deterministic cross-query deduplication."""
    chunk_id = doc.get("chunk_id")
    if chunk_id is not None:
        return f"chunk:{chunk_id}"
    document_id = doc.get("document_id")
    chunk_index = doc.get("chunk_index")
    if document_id is not None and chunk_index is not None:
        return f"document:{document_id}:{chunk_index}"
    content_hash = sha256(str(doc.get("content", "")).encode("utf-8")).hexdigest()
    return f"content:{content_hash}"


def _merge_results(
    primary: Sequence[tuple[Any, float]], supplemental: Sequence[tuple[Any, float]]
) -> list[tuple[Any, float]]:
    """Append entity matches without duplicate chunks within one query branch."""
    merged = []
    seen = set()
    for doc, score in [*primary, *supplemental]:
        key = _raw_document_identity(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append((doc, score))
    return merged


def _raw_document_identity(doc: Any) -> str:
    metadata = doc.metadata if hasattr(doc, "metadata") else {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id is not None:
        return f"chunk:{chunk_id}"
    content = doc.page_content if hasattr(doc, "page_content") else str(doc)
    return f"content:{sha256(content.encode('utf-8')).hexdigest()}"


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


def _document_item(doc: Any) -> dict | None:
    content = doc.page_content if hasattr(doc, "page_content") else str(doc)
    if not content.strip():
        return None
    metadata = doc.metadata if hasattr(doc, "metadata") else {}
    return {
        "content": content,
        "document_id": metadata.get("document_id"),
        "filename": metadata.get("filename", "unknown"),
        "chunk_id": metadata.get("chunk_id"),
        "chunk_index": metadata.get("chunk_index", 0),
        "chunk_length": metadata.get("chunk_length", len(content)),
        "page": metadata.get("page"),
        "source": "document",
    }


def _collect_candidates(question: str, candidate_budget: int, entity_budget: int) -> list[dict]:
    """Collect one branch's hybrid and entity candidates without reranking."""
    vector_store = get_retrieval_vector_store()
    query_entities = extract_entities(question)
    entity_filter = _entity_filter(query_entities)

    logger.info(
        "retrieval_branch_start",
        candidate_budget=candidate_budget,
        query_entity_count=len(query_entities),
        entity_supplement_active=bool(entity_filter),
        **text_log_fields(question, field="query"),
    )

    with _tracer.start_as_current_span("retrieval.candidate_branch") as span:
        span.set_attribute("retrieval.candidate_budget", candidate_budget)
        span.set_attribute("retrieval.entity_supplement_active", bool(entity_filter))

        with _tracer.start_as_current_span("retrieval.qdrant_query"):
            primary_results = _qdrant_search(vector_store, question, k=candidate_budget)

        entity_results: list[tuple[Any, float]] = []
        if entity_filter:
            with _tracer.start_as_current_span("retrieval.qdrant_entity_supplement"):
                entity_results = _qdrant_search(
                    vector_store, question, k=entity_budget, filter=entity_filter
                )

        results = _merge_results(primary_results, entity_results)
        primary_identities = {_raw_document_identity(doc) for doc, _score in primary_results}
        entity_identities = {_raw_document_identity(doc) for doc, _score in entity_results}
        query_sha256 = text_log_fields(question, field="query")["query_sha256"]
        doc_items = []
        scores = []
        for candidate_rank, (doc, score) in enumerate(results, start=1):
            item = _document_item(doc)
            if item is None:
                continue
            identity = _raw_document_identity(doc)
            channels = []
            if identity in primary_identities:
                channels.append("hybrid")
            if identity in entity_identities:
                channels.append("entity")
            numeric_score = float(score)
            doc_items.append(item)
            scores.append(numeric_score)
            logger.info(
                "retrieval_branch_candidate",
                query_sha256=query_sha256,
                candidate_rank=candidate_rank,
                retrieval_score=round(numeric_score, 4),
                retrieval_channels=channels,
                **evidence_log_fields(item),
            )

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
            "retrieval_branch_complete",
            count=len(doc_items),
            candidate_budget=candidate_budget,
            blank_results=blank_results,
            query_entity_count=len(query_entities),
            entity_supplement_count=len(entity_results),
            **score_stats,
        )
        return doc_items


def _merge_candidate_branches(branches: Sequence[Sequence[dict]], budget: int) -> list[dict]:
    """Fuse branch ranks with RRF while retaining provenance for final evidence coverage."""
    by_identity: dict[str, dict] = {}
    rank_scores: dict[str, float] = {}
    best_ranks: dict[str, int] = {}
    branch_ranks: dict[str, dict[str, int]] = {}

    for branch_index, branch in enumerate(branches, start=1):
        branch_seen: set[str] = set()
        for rank, doc in enumerate(branch, start=1):
            identity = _chunk_identity(doc)
            if identity in branch_seen:
                continue
            branch_seen.add(identity)
            by_identity.setdefault(identity, doc)
            rank_scores[identity] = rank_scores.get(identity, 0.0) + 1 / (_RRF_RANK_CONSTANT + rank)
            best_ranks[identity] = min(best_ranks.get(identity, rank), rank)
            branch_ranks.setdefault(identity, {})[str(branch_index)] = rank

    ranked_identities = sorted(
        by_identity,
        key=lambda identity: (-rank_scores[identity], best_ranks[identity], identity),
    )
    selected_identities = ranked_identities[:budget]
    for candidate_rank, identity in enumerate(selected_identities, start=1):
        logger.info(
            "retrieval_candidate_selected",
            candidate_rank=candidate_rank,
            rrf_score=round(rank_scores[identity], 6),
            branch_ranks=branch_ranks[identity],
            **evidence_log_fields(by_identity[identity], include_preview=False),
        )
    return [
        {
            **by_identity[identity],
            _BRANCH_RANKS_KEY: branch_ranks[identity],
        }
        for identity in selected_identities
    ]


def retrieve_evidence(search_queries: Sequence[str], rerank_query: str) -> RetrievalResult:
    """Execute one bounded retrieval plan and return one coverage-aware evidence set."""
    queries = list(dict.fromkeys(query.strip() for query in search_queries if query.strip()))
    if not 1 <= len(queries) <= _MAX_SEARCH_QUERIES:
        raise ValueError("retrieval requires one or two non-empty search queries")

    settings = get_settings()
    candidate_budget = settings.RETRIEVAL_CANDIDATE_BUDGET
    evidence_budget = settings.RETRIEVAL_EVIDENCE_BUDGET
    started_at = time.monotonic()

    with _tracer.start_as_current_span("retrieval.plan") as span:
        span.set_attribute("retrieval.query_count", len(queries))
        span.set_attribute("retrieval.candidate_budget", candidate_budget)
        span.set_attribute("retrieval.evidence_budget", evidence_budget)

        if len(queries) == 1:
            branches = [_collect_candidates(queries[0], candidate_budget, evidence_budget)]
        else:
            with ThreadPoolExecutor(max_workers=len(queries)) as executor:
                futures = [
                    executor.submit(
                        copy_context().run,
                        _collect_candidates,
                        query,
                        candidate_budget,
                        evidence_budget,
                    )
                    for query in queries
                ]
                branches = [future.result() for future in futures]

        candidates = _merge_candidate_branches(branches, candidate_budget)
        logger.info(
            "candidate_retrieval_complete",
            query_count=len(queries),
            candidates=len(candidates),
            duration_ms=round((time.monotonic() - started_at) * 1000, 1),
        )

        documents = (
            rerank(
                rerank_query,
                candidates,
                evidence_budget,
                branch_count=len(queries),
            )
            if candidates
            else []
        )
        metrics = RetrievalMetrics(
            query_count=len(queries),
            query_shape="multipart" if len(queries) == 2 else "single",
            candidates=len(candidates),
            evidence=len(documents),
            reranker_calls=1 if candidates else 0,
        )
        span.set_attribute("retrieval.candidates", metrics.candidates)
        span.set_attribute("retrieval.evidence", metrics.evidence)
        span.set_attribute("retrieval.reranker_calls", metrics.reranker_calls)
        logger.info("retrieval_plan_complete", **metrics.as_dict())
        return RetrievalResult(documents=documents, metrics=metrics)
