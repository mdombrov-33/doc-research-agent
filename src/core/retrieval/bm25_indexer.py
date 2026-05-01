from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.core.retrieval.tokenizer import tokenize
from src.utils.logger import logger


class BM25Indexer:
    def __init__(self):
        self.index: BM25Okapi | None = None
        self.documents: list[str] = []

    def build_index(self, documents: list[str]) -> None:
        if not documents:
            return

        self.documents = documents
        tokenized_docs = [tokenize(doc) for doc in documents]

        empty_count = sum(1 for doc in tokenized_docs if not doc)

        if empty_count == len(documents):
            logger.error(
                "all_docs_tokenized_empty",
                doc_count=len(documents),
                first_doc_preview=documents[0][:200],
            )
        elif empty_count > 0:
            logger.info("partial_tokenization_empty", empty=empty_count, total=len(documents))

        self.index = BM25Okapi(tokenized_docs)
        logger.debug("bm25_index_built", doc_count=len(documents))

    def get_scores(self, query: str) -> list[float]:
        if not self.index:
            return [0.0] * len(self.documents)

        query_tokens = tokenize(query)
        if not query_tokens:
            return [0.0] * len(self.documents)

        scores = self.index.get_scores(query_tokens)
        return scores.tolist()
