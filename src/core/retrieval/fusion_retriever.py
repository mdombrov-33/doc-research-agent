import numpy as np

from src.core.exceptions import FusionRetrievalError
from src.core.retrieval.bm25_indexer import BM25Indexer
from src.utils.logger import logger


class FusionRetriever:
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.bm25_indexer = BM25Indexer()

    def fuse_results(
        self, documents: list[str], vector_scores: list[float], query: str
    ) -> list[tuple[int, float]]:
        if not documents:
            return []

        vec_array = np.array(vector_scores, dtype=np.float64)
        vec_min = float(np.min(vec_array))
        vec_max = float(np.max(vec_array))

        if vec_max > vec_min:
            try:
                vector_normalized = ((vec_array - vec_min) / (vec_max - vec_min)).tolist()
            except Exception as e:
                logger.error("vector_normalization_failed", error=str(e))
                vector_normalized = [1.0] * len(vector_scores)
        else:
            vector_normalized = [1.0] * len(vector_scores)

        try:
            self.bm25_indexer.build_index(documents)
        except Exception as e:
            logger.error("bm25_index_build_failed", error=str(e), exc_info=True)
            raise FusionRetrievalError(f"BM25 index build failed: {e}") from e

        try:
            bm25_scores = self.bm25_indexer.get_scores(query)
        except Exception as e:
            logger.error("bm25_scoring_failed", error=str(e), exc_info=True)
            raise FusionRetrievalError(f"BM25 scoring failed: {e}") from e

        try:
            bm25_array = np.array(bm25_scores, dtype=np.float64)
            bm25_min = float(np.min(bm25_array))
            bm25_max = float(np.max(bm25_array))
        except Exception as e:
            logger.error("bm25_array_conversion_failed", error=str(e), exc_info=True)
            raise FusionRetrievalError(f"BM25 array conversion failed: {e}") from e

        if bm25_max > bm25_min:
            bm25_normalized = ((bm25_array - bm25_min) / (bm25_max - bm25_min)).tolist()
        else:
            bm25_normalized = [0.5] * len(bm25_scores)

        fused_scores = [
            (i, self.alpha * vec + (1 - self.alpha) * bm25)
            for i, (vec, bm25) in enumerate(zip(vector_normalized, bm25_normalized))
        ]
        fused_scores.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            "fusion_complete",
            doc_count=len(documents),
            alpha=self.alpha,
            top_score=round(fused_scores[0][1], 4) if fused_scores else None,
        )

        return fused_scores
