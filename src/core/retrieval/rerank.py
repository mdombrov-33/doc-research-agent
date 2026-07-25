import time
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder
from opentelemetry import trace

from src.config import get_settings
from src.utils.logger import logger

_tracer = trace.get_tracer(__name__)


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


def rerank(query: str, documents: list[dict], limit: int) -> list[dict]:
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
        ranked = sorted(
            zip(documents, scores, range(len(documents))),
            key=lambda triple: triple[1],
            reverse=True,
        )

        # Calibrated relevance floor: drop chunks the cross-encoder scored below the threshold
        # before they reach the evidence pool. Everything below dies here; if that empties the
        # pool, assessment sees no evidence and the graph falls back or abstains (intended).
        floor = settings.RERANK_SCORE_FLOOR
        if floor is not None:
            kept = [triple for triple in ranked if triple[1] >= floor]
            dropped = len(ranked) - len(kept)
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
            ranked = kept

        top = ranked[:limit]

        # "promoted" = docs moved into the final evidence slice from below its raw cutoff.
        promoted = sum(1 for _, _, original_rank in top if original_rank >= limit)

        span.set_attribute("rerank.promoted", promoted)
        duration_ms = round((time.monotonic() - started_at) * 1000, 1)
        span.set_attribute("rerank.duration_ms", duration_ms)

        logger.info(
            "documents_reranked",
            model=settings.RERANK_MODEL,
            candidates=len(documents),
            returned=len(top),
            promoted=promoted,
            duration_ms=duration_ms,
            score_max=round(top[0][1], 4),
            score_min=round(top[-1][1], 4),
        )

        return [doc for doc, _, _ in top]
