import time
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder
from opentelemetry import trace

from src.config import get_settings
from src.core.evidence_observability import evidence_log_fields
from src.utils.logger import logger

_tracer = trace.get_tracer(__name__)
_BRANCH_RANKS_KEY = "_retrieval_branch_ranks"

type RankedDocument = tuple[dict, float, int]


@lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str) -> TextCrossEncoder:
    """Load the cross-encoder once. First call downloads the ONNX model (~80MB) and caches it."""
    logger.info("cross_encoder_loading", model=model_name)
    encoder = TextCrossEncoder(model_name=model_name)
    logger.info("cross_encoder_loaded", model=model_name)
    return encoder


def warmup() -> None:
    """Prime the mandatory cross-encoder so the first query does not pay model-load latency."""
    settings = get_settings()
    _get_cross_encoder(settings.RERANK_MODEL)


def _select_evidence(
    ranked: list[RankedDocument],
    limit: int,
    branch_count: int,
) -> tuple[list[RankedDocument], dict[int, list[str]], int]:
    if branch_count <= 1:
        top = ranked[:limit]
        return top, {original_rank: ["global_rerank"] for _, _, original_rank in top}, 0

    coverage_per_branch = max(1, limit // branch_count - 1)
    branch_shortlists: dict[str, list[RankedDocument]] = {}
    for branch_index in range(1, branch_count + 1):
        branch_key = str(branch_index)
        branch_shortlists[branch_key] = sorted(
            (triple for triple in ranked if branch_key in triple[0].get(_BRANCH_RANKS_KEY, {})),
            key=lambda triple: triple[0][_BRANCH_RANKS_KEY][branch_key],
        )[:coverage_per_branch]

    selected_ranks: set[int] = set()
    selection_reasons: dict[int, list[str]] = {}
    for depth in range(coverage_per_branch):
        for branch_key, shortlist in branch_shortlists.items():
            if depth >= len(shortlist):
                continue
            original_rank = shortlist[depth][2]
            if original_rank not in selected_ranks and len(selected_ranks) >= limit:
                continue
            selected_ranks.add(original_rank)
            selection_reasons.setdefault(original_rank, []).append(f"branch_coverage:{branch_key}")

    for _, _, original_rank in ranked:
        if len(selected_ranks) >= limit:
            break
        if original_rank in selected_ranks:
            continue
        selected_ranks.add(original_rank)
        selection_reasons[original_rank] = ["global_rerank"]

    top = [triple for triple in ranked if triple[2] in selected_ranks]
    coverage_selected = sum(
        any(reason.startswith("branch_coverage:") for reason in reasons)
        for reasons in selection_reasons.values()
    )
    return top, selection_reasons, coverage_selected


def rerank(
    query: str,
    documents: list[dict],
    limit: int,
    *,
    branch_count: int = 1,
) -> list[dict]:
    """Reorder retrieved documents by cross-encoder relevance and bound the evidence set.

    Each document is a dict carrying a "content" field. Unlike bi-encoder retrieval, the
    cross-encoder reads the query and each document together, so the ordering reflects the
    actual query-document relationship rather than embedding proximity alone.
    """
    if not documents:
        return documents

    settings = get_settings()
    started_at = time.monotonic()
    with _tracer.start_as_current_span("retrieval.rerank") as span:
        span.set_attribute("rerank.model", settings.RERANK_MODEL)
        span.set_attribute("rerank.candidates", len(documents))
        span.set_attribute("rerank.evidence_budget", limit)

        encoder = _get_cross_encoder(settings.RERANK_MODEL)
        scores = list(encoder.rerank(query, [doc["content"] for doc in documents]))

        # Pair each doc with its score and original retrieval rank, then sort by score, best first.
        ranked_all = sorted(
            zip(documents, scores, range(len(documents))),
            key=lambda triple: triple[1],
            reverse=True,
        )

        # Calibrated relevance floor: drop chunks the cross-encoder scored below the threshold
        # before they reach the evidence pool. Everything below dies here; if that empties the
        # pool, assessment sees no evidence and the graph falls back or abstains (intended).
        floor = settings.RERANK_SCORE_FLOOR
        ranked = ranked_all
        dropped = 0
        if floor is not None:
            kept = [triple for triple in ranked_all if triple[1] >= floor]
            dropped = len(ranked_all) - len(kept)
            ranked = kept

        top, selection_reasons, coverage_selected = _select_evidence(
            ranked,
            limit,
            branch_count,
        )
        selected_candidate_ranks = {original_rank for _, _, original_rank in top}
        for rerank_rank, (doc, score, original_rank) in enumerate(ranked_all, start=1):
            logger.info(
                "rerank_candidate_scored",
                rerank_rank=rerank_rank,
                candidate_rank=original_rank + 1,
                rerank_score=round(float(score), 4),
                floor_passed=floor is None or score >= floor,
                selected=original_rank in selected_candidate_ranks,
                selection_reasons=selection_reasons.get(original_rank, []),
                **evidence_log_fields(doc, include_preview=False),
            )

        if floor is not None:
            if not kept:
                duration_ms = round((time.monotonic() - started_at) * 1000, 1)
                span.set_attribute("rerank.floor_excluded_all", True)
                span.set_attribute("rerank.duration_ms", duration_ms)
                logger.info(
                    "documents_reranked",
                    model=settings.RERANK_MODEL,
                    candidates=len(documents),
                    returned=0,
                    floor=floor,
                    floor_dropped=dropped,
                    duration_ms=duration_ms,
                )
                return []
            span.set_attribute("rerank.floor_dropped", dropped)

        # "promoted" = docs moved into the final evidence slice from below its raw cutoff.
        promoted = sum(1 for _, _, original_rank in top if original_rank >= limit)

        span.set_attribute("rerank.promoted", promoted)
        span.set_attribute("rerank.branch_count", branch_count)
        span.set_attribute("rerank.branch_coverage_selected", coverage_selected)
        duration_ms = round((time.monotonic() - started_at) * 1000, 1)
        span.set_attribute("rerank.duration_ms", duration_ms)

        logger.info(
            "documents_reranked",
            model=settings.RERANK_MODEL,
            candidates=len(documents),
            returned=len(top),
            promoted=promoted,
            branch_count=branch_count,
            branch_coverage_selected=coverage_selected,
            duration_ms=duration_ms,
            score_max=round(top[0][1], 4),
            score_min=round(top[-1][1], 4),
        )

        return [doc for doc, _, _ in top]
