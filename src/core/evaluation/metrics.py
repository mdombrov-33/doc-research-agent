from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class QueryEvaluation:
    question: str
    retrieval_precision: float
    docs_retrieved: int
    docs_relevant: int
    web_search_triggered: bool
    latency_ms: float


class EvaluationTracker:
    def __init__(self, db_path: str | None = None):
        self.total_queries = 0
        self.web_search_triggered = 0
        self.total_docs_retrieved = 0
        self.total_docs_relevant = 0
        self.total_latency_ms = 0.0
        self._lock = Lock()
        self._db: Any = None

        if db_path:
            from src.core.evaluation.db import EvalDB

            self._db = EvalDB(db_path)
            saved = self._db.load()
            if saved:
                self.total_queries = saved["total_queries"]
                self.web_search_triggered = saved["web_search_triggered"]
                self.total_docs_retrieved = saved["total_docs_retrieved"]
                self.total_docs_relevant = saved["total_docs_relevant"]
                self.total_latency_ms = saved["total_latency_ms"]

    def record(self, evaluation: QueryEvaluation) -> None:
        with self._lock:
            self.total_queries += 1

            if evaluation.web_search_triggered:
                self.web_search_triggered += 1

            self.total_docs_retrieved += evaluation.docs_retrieved
            self.total_docs_relevant += evaluation.docs_relevant
            self.total_latency_ms += evaluation.latency_ms

            if self._db:
                self._db.flush(
                    self.total_queries,
                    self.web_search_triggered,
                    self.total_docs_retrieved,
                    self.total_docs_relevant,
                    self.total_latency_ms,
                )

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            if self.total_queries == 0:
                return {
                    "total_queries": 0,
                    "web_search_rate": 0.0,
                    "avg_docs_retrieved": 0.0,
                    "avg_docs_relevant": 0.0,
                    "avg_retrieval_precision": 0.0,
                    "avg_latency_ms": 0.0,
                }

            return {
                "total_queries": self.total_queries,
                "web_search_rate": self.web_search_triggered / self.total_queries,
                "avg_docs_retrieved": self.total_docs_retrieved / self.total_queries,
                "avg_docs_relevant": self.total_docs_relevant / self.total_queries,
                "avg_retrieval_precision": (
                    self.total_docs_relevant / self.total_docs_retrieved
                    if self.total_docs_retrieved > 0
                    else 0.0
                ),
                "avg_latency_ms": self.total_latency_ms / self.total_queries,
            }
