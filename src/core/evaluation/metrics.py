from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class QueryEvaluation:
    question: str
    retrieval_precision: float
    docs_retrieved: int
    docs_relevant: int
    hallucination_check: str
    quality_check: str
    web_search_triggered: bool
    generation_attempts: int
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "retrieval_precision": self.retrieval_precision,
            "docs_retrieved": self.docs_retrieved,
            "docs_relevant": self.docs_relevant,
            "hallucination_check": self.hallucination_check,
            "quality_check": self.quality_check,
            "web_search_triggered": self.web_search_triggered,
            "generation_attempts": self.generation_attempts,
            "latency_ms": self.latency_ms,
        }


class EvaluationTracker:
    def __init__(self):
        self.total_queries = 0
        self.hallucination_passed = 0
        self.quality_passed = 0
        self.web_search_triggered = 0
        self.total_docs_retrieved = 0
        self.total_docs_relevant = 0
        self.total_latency_ms = 0.0
        self.total_generation_attempts = 0
        self._lock = Lock()

    def record(self, evaluation: QueryEvaluation) -> None:
        with self._lock:
            self.total_queries += 1

            if evaluation.hallucination_check == "yes":
                self.hallucination_passed += 1

            if evaluation.quality_check == "yes":
                self.quality_passed += 1

            if evaluation.web_search_triggered:
                self.web_search_triggered += 1

            self.total_docs_retrieved += evaluation.docs_retrieved
            self.total_docs_relevant += evaluation.docs_relevant
            self.total_latency_ms += evaluation.latency_ms
            self.total_generation_attempts += evaluation.generation_attempts

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            if self.total_queries == 0:
                return {
                    "total_queries": 0,
                    "hallucination_pass_rate": 0.0,
                    "quality_pass_rate": 0.0,
                    "web_search_rate": 0.0,
                    "avg_docs_retrieved": 0.0,
                    "avg_docs_relevant": 0.0,
                    "avg_retrieval_precision": 0.0,
                    "avg_latency_ms": 0.0,
                    "avg_generation_attempts": 0.0,
                }

            return {
                "total_queries": self.total_queries,
                "hallucination_pass_rate": self.hallucination_passed / self.total_queries,
                "quality_pass_rate": self.quality_passed / self.total_queries,
                "web_search_rate": self.web_search_triggered / self.total_queries,
                "avg_docs_retrieved": self.total_docs_retrieved / self.total_queries,
                "avg_docs_relevant": self.total_docs_relevant / self.total_queries,
                "avg_retrieval_precision": (
                    self.total_docs_relevant / self.total_docs_retrieved
                    if self.total_docs_retrieved > 0
                    else 0.0
                ),
                "avg_latency_ms": self.total_latency_ms / self.total_queries,
                "avg_generation_attempts": self.total_generation_attempts / self.total_queries,
            }


_tracker_instance: EvaluationTracker | None = None
_tracker_lock = Lock()


def get_evaluation_tracker() -> EvaluationTracker:
    global _tracker_instance
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = EvaluationTracker()
    return _tracker_instance
