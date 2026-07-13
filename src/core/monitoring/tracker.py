from dataclasses import dataclass
from threading import Lock
from typing import Any

from src.core.monitoring.db import MetricsDB


@dataclass
class QueryMetrics:
    question: str
    sources_retrieved: int
    web_search_triggered: bool
    latency_ms: float


class MetricsTracker:
    """Online/runtime telemetry — aggregates live query stats during serving.

    Distinct from the offline retrieval evaluation in evals/, which scores
    retrieval quality against a golden set.
    """

    def __init__(self, db_path: str | None = None):
        self.total_queries = 0
        self.web_search_triggered = 0
        self.total_sources_retrieved = 0
        self.total_latency_ms = 0.0
        self._lock = Lock()
        self._db: Any = None

        if db_path:
            self._db = MetricsDB(db_path)
            saved = self._db.load()
            if saved:
                self.total_queries = saved["total_queries"]
                self.web_search_triggered = saved["web_search_triggered"]
                self.total_sources_retrieved = saved["total_sources_retrieved"]
                self.total_latency_ms = saved["total_latency_ms"]

    def record(self, evaluation: QueryMetrics) -> None:
        with self._lock:
            self.total_queries += 1

            if evaluation.web_search_triggered:
                self.web_search_triggered += 1

            self.total_sources_retrieved += evaluation.sources_retrieved
            self.total_latency_ms += evaluation.latency_ms

            if self._db:
                self._db.flush(
                    self.total_queries,
                    self.web_search_triggered,
                    self.total_sources_retrieved,
                    self.total_latency_ms,
                )

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            if self.total_queries == 0:
                return {
                    "total_queries": 0,
                    "web_search_rate": 0.0,
                    "avg_sources_retrieved": 0.0,
                    "avg_latency_ms": 0.0,
                }

            return {
                "total_queries": self.total_queries,
                "web_search_rate": self.web_search_triggered / self.total_queries,
                "avg_sources_retrieved": self.total_sources_retrieved / self.total_queries,
                "avg_latency_ms": self.total_latency_ms / self.total_queries,
            }
