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
    """Prime the cross-encoder at startup so the first query doesn't pay the ~2s model load.
    No-op when reranking is disabled, since rerank() is the only caller."""
    settings = get_settings()
    if settings.RERANK_ENABLED:
        _get_cross_encoder(settings.RERANK_MODEL)


def rerank(query: str, documents: list[dict], top_k: int) -> list[dict]:
    """Reorder retrieved documents by cross-encoder relevance and keep the top_k.

    Each document is a dict carrying a "content" field. Unlike bi-encoder retrieval, the
    cross-encoder reads the query and each document together, so the ordering reflects the
    actual query-document relationship rather than embedding proximity alone.
    """
    if not documents:
        return documents

    settings = get_settings()
    with _tracer.start_as_current_span("retrieval.rerank") as span:
        span.set_attribute("rerank.model", settings.RERANK_MODEL)
        span.set_attribute("rerank.candidates", len(documents))
        span.set_attribute("rerank.top_k", top_k)

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
                span.set_attribute("rerank.floor_excluded_all", True)
                logger.info(
                    "documents_reranked",
                    model=settings.RERANK_MODEL,
                    candidates=len(documents),
                    returned=0,
                    floor=floor,
                    floor_dropped=dropped,
                )
                return []
            span.set_attribute("rerank.floor_dropped", dropped)
            ranked = kept

        top = ranked[:top_k]

        # "promoted" = docs the reranker moved into the top_k slice from positions top_k..fetch_k,
        # i.e. hits that raw hybrid ordering would have cut.
        promoted = sum(1 for _, _, original_rank in top if original_rank >= top_k)

        span.set_attribute("rerank.promoted", promoted)

        logger.info(
            "documents_reranked",
            model=settings.RERANK_MODEL,
            candidates=len(documents),
            returned=len(top),
            promoted=promoted,
            score_max=round(top[0][1], 4),
            score_min=round(top[-1][1], 4),
        )

        return [doc for doc, _, _ in top]
