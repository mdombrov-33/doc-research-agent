from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.core.retrieval.tokenizer import tokenize
from src.utils.logger import logger


class BM25Indexer:
    """Handles BM25 indexing and scoring for keyword search."""

    def __init__(self):
        self.index: BM25Okapi | None = None
        self.documents: list[str] = []

    def build_index(self, documents: list[str]) -> None:
        """
        Build BM25 index from document texts.

        Args:
            documents: List of document text strings
        """
        if not documents:
            logger.warning("No documents provided to build BM25 index")
            return

        self.documents = documents
        tokenized_docs = [tokenize(doc) for doc in documents]

        # Debug: check if any docs have tokens
        empty_count = sum(1 for doc in tokenized_docs if not doc)
        logger.info(
            f"Tokenized {len(documents)} docs: {empty_count} empty, {len(documents) - empty_count} with tokens"  # noqa: E501
        )

        if empty_count == len(documents):
            logger.error("ALL documents tokenized to empty! First doc preview:")
            logger.error(f"Doc 0 (first 200 chars): {documents[0][:200]}")
            logger.error(f"Doc 0 tokens: {tokenized_docs[0]}")
            # Don't raise, just log and continue - will cause division by zero but we'll see the debug info  # noqa: E501
        elif empty_count > 0:
            logger.warning(f"{empty_count}/{len(documents)} documents have no tokens")

        self.index = BM25Okapi(tokenized_docs)
        logger.info(f"Built BM25 index for {len(documents)} documents")

    def get_scores(self, query: str) -> list[float]:
        """
        Get BM25 scores for a query against indexed documents.

        Args:
            query: Search query string

        Returns:
            List of BM25 scores for each document
        """
        if not self.index:
            logger.warning("BM25 index not built, returning zero scores")
            return [0.0] * len(self.documents)

        query_tokens = tokenize(query)

        if not query_tokens:
            logger.warning("Query tokenization resulted in empty tokens")
            return [0.0] * len(self.documents)

        scores = self.index.get_scores(query_tokens)
        return scores.tolist()
