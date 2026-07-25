from dataclasses import dataclass
from threading import Lock
from typing import Any

from src.config import Settings
from src.core.agent.outcomes import FinalOutcome
from src.core.monitoring.db import (
    MetricsStore,
    PostgresMetricsDB,
    SqliteMetricsDB,
    _event_stats_from_rows,
)


@dataclass
class QueryMetrics:
    sources_retrieved: int
    web_search_triggered: bool
    latency_ms: float
    time_to_first_token_ms: float | None
    outcome: FinalOutcome
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reported_cost: float | None = None


class MetricsTracker:
    """Online/runtime telemetry — aggregates live query stats during serving.

    Distinct from the offline retrieval evaluation in evals/, which scores
    retrieval quality against a golden set.
    """

    def __init__(self, store: MetricsStore | None = None):
        self.total_queries = 0
        self.web_search_triggered = 0
        self.total_sources_retrieved = 0
        self.total_latency_ms = 0.0
        self.total_time_to_first_token_ms = 0.0
        self.time_to_first_token_samples = 0
        self.document_answers = 0
        self.web_answers = 0
        self.abstentions = 0
        self._events: list[QueryMetrics] = []
        self._lock = Lock()
        self._store = store

        if self._store:
            saved = self._store.load()
            if saved:
                self.total_queries = saved["total_queries"]
                self.web_search_triggered = saved["web_search_triggered"]
                self.total_sources_retrieved = saved["total_sources_retrieved"]
                self.total_latency_ms = saved["total_latency_ms"]
                self.total_time_to_first_token_ms = saved["total_time_to_first_token_ms"]
                self.time_to_first_token_samples = saved["time_to_first_token_samples"]
                self.document_answers = saved["document_answers"]
                self.web_answers = saved["web_answers"]
                self.abstentions = saved["abstentions"]

    def record(self, evaluation: QueryMetrics) -> None:
        with self._lock:
            if self._store:
                self._store.record(
                    evaluation.sources_retrieved,
                    evaluation.web_search_triggered,
                    evaluation.latency_ms,
                    evaluation.time_to_first_token_ms,
                    evaluation.outcome,
                )
                self._store.record_event(
                    evaluation.model,
                    evaluation.input_tokens,
                    evaluation.output_tokens,
                    evaluation.reported_cost,
                    evaluation.latency_ms,
                    evaluation.time_to_first_token_ms,
                    evaluation.outcome,
                )
                return

            self.total_queries += 1

            if evaluation.web_search_triggered:
                self.web_search_triggered += 1

            self.total_sources_retrieved += evaluation.sources_retrieved
            self.total_latency_ms += evaluation.latency_ms
            if evaluation.time_to_first_token_ms is not None:
                self.total_time_to_first_token_ms += evaluation.time_to_first_token_ms
                self.time_to_first_token_samples += 1
            if evaluation.outcome == "document_answer":
                self.document_answers += 1
            elif evaluation.outcome == "web_answer":
                self.web_answers += 1
            else:
                self.abstentions += 1
            self._events.append(evaluation)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            if self._store:
                saved = self._store.load()
                stats = _stats_from_totals(saved) if saved else _empty_stats()
                stats.update(self._store.load_event_stats())
                return stats

            stats = _stats_from_totals(
                {
                    "total_queries": self.total_queries,
                    "web_search_triggered": self.web_search_triggered,
                    "total_sources_retrieved": self.total_sources_retrieved,
                    "total_latency_ms": self.total_latency_ms,
                    "total_time_to_first_token_ms": self.total_time_to_first_token_ms,
                    "time_to_first_token_samples": self.time_to_first_token_samples,
                    "document_answers": self.document_answers,
                    "web_answers": self.web_answers,
                    "abstentions": self.abstentions,
                }
            )
            stats.update(_event_stats_from_metrics(self._events))
            return stats

    def close(self) -> None:
        if self._store:
            self._store.close()

    @classmethod
    def from_settings(cls, settings: Settings) -> "MetricsTracker":
        if settings.METRICS_BACKEND == "postgres":
            if not settings.DATABASE_URL:
                raise ValueError("DATABASE_URL is required for Postgres metrics")
            return cls(PostgresMetricsDB(settings.DATABASE_URL))
        return cls(SqliteMetricsDB(settings.metrics_db_path))


def _stats_from_totals(totals: dict[str, int | float]) -> dict[str, Any]:
    total_queries = totals["total_queries"]
    if total_queries == 0:
        return _empty_stats()

    return {
        "total_queries": total_queries,
        "web_search_rate": totals["web_search_triggered"] / total_queries,
        "avg_sources_retrieved": totals["total_sources_retrieved"] / total_queries,
        "avg_latency_ms": totals["total_latency_ms"] / total_queries,
        "avg_time_to_first_token_ms": (
            totals["total_time_to_first_token_ms"] / totals["time_to_first_token_samples"]
            if totals["time_to_first_token_samples"]
            else 0.0
        ),
        "document_answer_rate": totals["document_answers"] / total_queries,
        "web_answer_rate": totals["web_answers"] / total_queries,
        "abstention_rate": totals["abstentions"] / total_queries,
    }


def _empty_stats() -> dict[str, Any]:
    return {
        "total_queries": 0,
        "web_search_rate": 0.0,
        "avg_sources_retrieved": 0.0,
        "avg_latency_ms": 0.0,
        "avg_time_to_first_token_ms": 0.0,
        "document_answer_rate": 0.0,
        "web_answer_rate": 0.0,
        "abstention_rate": 0.0,
    }


def _event_stats_from_metrics(events: list[QueryMetrics]) -> dict[str, Any]:
    """Aggregate the in-memory event list the same way the DB backends aggregate rows."""
    averages = (
        _mean(e.input_tokens for e in events),
        _mean(e.output_tokens for e in events),
        _mean(e.reported_cost for e in events),
    )
    model_rows: list[tuple] = []
    for model in sorted({e.model for e in events if e.model is not None}):
        group = [e for e in events if e.model == model]
        model_rows.append(
            (
                model,
                len(group),
                _sum(e.input_tokens for e in group),
                _sum(e.output_tokens for e in group),
                _sum(e.reported_cost for e in group),
            )
        )
    return _event_stats_from_rows(averages, model_rows)


def _mean(values: Any) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _sum(values: Any) -> float | int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None
