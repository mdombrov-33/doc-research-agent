import numpy as np

from src.core.retrieval.bm25_indexer import BM25Indexer
from src.utils.logger import logger


class FusionRetriever:
    """Handles fusion retrieval combining vector and BM25 search."""

    def __init__(self, alpha: float = 0.6):
        """
        Initialize with fusion weight.

        Args:
            alpha: Weight for vector scores (1-alpha for BM25).
                   0.6 = 60% vector, 40% BM25 (balanced for names + semantics)
        """
        self.alpha = alpha
        self.bm25_indexer = BM25Indexer()

    def fuse_results(
        self, documents: list[str], vector_scores: list[float], query: str
    ) -> list[tuple[int, float]]:
        """
        Fuse vector and BM25 scores for retrieved documents.

        Args:
            documents: List of document text strings
            vector_scores: List of vector similarity scores (0-1, higher is better)
            query: Search query string

        Returns:
            List of tuples (doc_index, fused_score) sorted by score descending
        """
        if not documents:
            logger.warning("No documents to fuse")
            return []

        # Normalize vector scores to 0-1 range
        vec_array = np.array(vector_scores, dtype=np.float64)
        vec_min = float(np.min(vec_array))
        vec_max = float(np.max(vec_array))

        if vec_max > vec_min:
            try:
                vector_normalized = ((vec_array - vec_min) / (vec_max - vec_min)).tolist()
            except Exception as e:
                logger.error(f"Vector normalization failed: {e}")
                vector_normalized = [1.0] * len(vector_scores)
        else:
            vector_normalized = [1.0] * len(vector_scores)  # All equal, use 1.0
            logger.info(f"All vector scores equal ({vec_max:.4f}), using 1.0")

        logger.info(
            f"Vector scores: min={vec_min:.4f}, max={vec_max:.4f}, "
            f"normalized={vector_normalized[:3]}"
        )

        try:
            self.bm25_indexer.build_index(documents)
            logger.info("BM25 index built successfully")
        except Exception as e:
            logger.error(f"BM25 index build failed: {e}", exc_info=True)
            raise

        try:
            bm25_scores = self.bm25_indexer.get_scores(query)
            logger.info(f"BM25 scores retrieved: {len(bm25_scores)} scores")
        except Exception as e:
            logger.error(f"BM25 scoring failed: {e}", exc_info=True)
            raise

        # Normalize BM25 scores to 0-1 range with safe handling
        try:
            bm25_array = np.array(bm25_scores, dtype=np.float64)
            bm25_min = float(np.min(bm25_array))
            bm25_max = float(np.max(bm25_array))
        except Exception as e:
            logger.error(f"BM25 array conversion failed: {e}", exc_info=True)
            raise

        if bm25_max > bm25_min:
            bm25_normalized = ((bm25_array - bm25_min) / (bm25_max - bm25_min)).tolist()
        else:
            # All equal - use 0.5 as neutral score
            bm25_normalized = [0.5] * len(bm25_scores)
            logger.info(f"All BM25 scores equal ({bm25_max:.4f}), using 0.5")

        logger.info(
            f"BM25 scores: min={bm25_min:.4f}, max={bm25_max:.4f}, normalized={bm25_normalized[:3]}"
        )

        fused_scores = []
        for i, (vec_score, bm25_score) in enumerate(zip(vector_normalized, bm25_normalized)):
            fused = self.alpha * vec_score + (1 - self.alpha) * bm25_score
            fused_scores.append((i, fused))
            if i < 3:  # Log first 3 for debugging
                logger.info(
                    f"Doc {i}: vector={vec_score:.4f}, bm25={bm25_score:.4f}, fused={fused:.4f}"
                )

        fused_scores.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"Fused {len(documents)} results with alpha={self.alpha} "
            f"(vector={self.alpha:.0%}, bm25={1 - self.alpha:.0%})"
        )

        return fused_scores
