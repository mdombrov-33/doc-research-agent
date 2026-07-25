import sqlite3
from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.core.agent.outcomes import FinalOutcome
from src.core.monitoring import db
from src.core.monitoring.db import SqliteMetricsDB
from src.core.monitoring.tracker import MetricsTracker, QueryMetrics


def _make_eval(
    sources_retrieved: int = 5,
    web_search_triggered: bool = False,
    latency_ms: float = 1000.0,
    time_to_first_token_ms: float | None = None,
    outcome: FinalOutcome = "document_answer",
    model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    reported_cost: float | None = None,
) -> QueryMetrics:
    return QueryMetrics(
        sources_retrieved=sources_retrieved,
        web_search_triggered=web_search_triggered,
        latency_ms=latency_ms,
        time_to_first_token_ms=time_to_first_token_ms,
        outcome=outcome,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reported_cost=reported_cost,
    )


def test_empty_tracker():
    tracker = MetricsTracker()
    stats = tracker.get_stats()
    assert stats["total_queries"] == 0
    assert stats["avg_sources_retrieved"] == 0.0
    assert stats["web_search_rate"] == 0.0


def test_single_query_stats():
    tracker = MetricsTracker()
    tracker.record(_make_eval(sources_retrieved=5, latency_ms=2000.0))
    stats = tracker.get_stats()
    assert stats["total_queries"] == 1
    assert stats["avg_sources_retrieved"] == 5.0
    assert stats["avg_latency_ms"] == 2000.0
    assert stats["avg_time_to_first_token_ms"] == 0.0
    assert stats["document_answer_rate"] == 1.0
    assert "avg_retrieval_precision" not in stats


def test_legacy_database_migrates_source_total(tmp_path):
    db_path = tmp_path / "metrics.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE monitoring_stats ("
            "id INTEGER PRIMARY KEY, total_queries INTEGER, "
            "web_search_triggered INTEGER, total_docs_retrieved INTEGER, "
            "total_docs_relevant INTEGER, total_latency_ms REAL)"
        )
        conn.execute("INSERT INTO monitoring_stats VALUES (1, 2, 1, 6, 5, 3000.0)")

    stats = MetricsTracker(SqliteMetricsDB(str(db_path))).get_stats()

    assert stats == {
        "total_queries": 2,
        "web_search_rate": 0.5,
        "avg_sources_retrieved": 3.0,
        "avg_latency_ms": 1500.0,
        "avg_time_to_first_token_ms": 0.0,
        "document_answer_rate": 0.0,
        "web_answer_rate": 0.0,
        "abstention_rate": 0.0,
        "avg_input_tokens": 0.0,
        "avg_output_tokens": 0.0,
        "avg_cost_per_query": None,
        "models": {},
    }


def test_sqlite_metrics_persist_recorded_totals(tmp_path):
    db_path = tmp_path / "metrics.db"
    tracker = MetricsTracker(SqliteMetricsDB(str(db_path)))
    tracker.record(
        _make_eval(
            sources_retrieved=4,
            web_search_triggered=True,
            latency_ms=500.0,
            time_to_first_token_ms=125.0,
        )
    )

    stats = MetricsTracker(SqliteMetricsDB(str(db_path))).get_stats()

    assert stats == {
        "total_queries": 1,
        "web_search_rate": 1.0,
        "avg_sources_retrieved": 4.0,
        "avg_latency_ms": 500.0,
        "avg_time_to_first_token_ms": 125.0,
        "document_answer_rate": 1.0,
        "web_answer_rate": 0.0,
        "abstention_rate": 0.0,
        "avg_input_tokens": 0.0,
        "avg_output_tokens": 0.0,
        "avg_cost_per_query": None,
        "models": {},
    }


def test_avg_sources_retrieved_across_multiple_queries():
    tracker = MetricsTracker()
    tracker.record(_make_eval(sources_retrieved=4))
    tracker.record(_make_eval(sources_retrieved=2))
    stats = tracker.get_stats()
    assert stats["avg_sources_retrieved"] == 3.0


def test_web_search_rate():
    tracker = MetricsTracker()
    tracker.record(_make_eval(web_search_triggered=True))
    tracker.record(_make_eval(web_search_triggered=True))
    tracker.record(_make_eval(web_search_triggered=False))
    stats = tracker.get_stats()
    assert stats["web_search_rate"] == 2 / 3


def test_outcome_rates():
    tracker = MetricsTracker()
    tracker.record(_make_eval(outcome="document_answer"))
    tracker.record(_make_eval(outcome="web_answer"))
    tracker.record(_make_eval(outcome="abstained"))

    assert tracker.get_stats()["document_answer_rate"] == 1 / 3
    assert tracker.get_stats()["web_answer_rate"] == 1 / 3
    assert tracker.get_stats()["abstention_rate"] == 1 / 3


def test_avg_latency():
    tracker = MetricsTracker()
    tracker.record(_make_eval(latency_ms=1000.0))
    tracker.record(_make_eval(latency_ms=3000.0))
    stats = tracker.get_stats()
    assert stats["avg_latency_ms"] == 2000.0


def test_avg_time_to_first_token_uses_visible_token_samples_only():
    tracker = MetricsTracker()
    tracker.record(_make_eval(time_to_first_token_ms=None))
    tracker.record(_make_eval(time_to_first_token_ms=1000.0))
    tracker.record(_make_eval(time_to_first_token_ms=2000.0))

    assert tracker.get_stats()["avg_time_to_first_token_ms"] == 1500.0


def test_postgres_metrics_record_uses_atomic_increment(monkeypatch):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    monkeypatch.setattr(db.psycopg, "connect", lambda url, autocommit: connection)

    store = db.PostgresMetricsDB("postgresql://example")
    store.record(
        sources_retrieved=3,
        web_search_triggered=True,
        latency_ms=250.0,
        time_to_first_token_ms=100.0,
        outcome="document_answer",
    )

    assert cursor.execute.call_args_list[-1].args[1] == (
        True,
        3,
        250.0,
        100.0,
        True,
        True,
        False,
        False,
    )
    record_sql = cursor.execute.call_args_list[-1].args[0]
    assert "total_queries = monitoring_stats.total_queries + excluded.total_queries" in record_sql


def test_postgres_metrics_configuration_requires_database_url():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        MetricsTracker.from_settings(Settings(METRICS_BACKEND="postgres"))


def test_event_stats_average_tokens_and_cost_across_queries():
    tracker = MetricsTracker()
    tracker.record(
        _make_eval(model="a/x", input_tokens=100, output_tokens=20, reported_cost=0.002)
    )
    tracker.record(
        _make_eval(model="a/x", input_tokens=200, output_tokens=40, reported_cost=0.004)
    )

    stats = tracker.get_stats()
    assert stats["avg_input_tokens"] == 150.0
    assert stats["avg_output_tokens"] == 30.0
    assert stats["avg_cost_per_query"] == 0.003


def test_event_stats_per_model_breakdown():
    tracker = MetricsTracker()
    tracker.record(_make_eval(model="a/x", input_tokens=100, output_tokens=20))
    tracker.record(_make_eval(model="b/y", input_tokens=300, output_tokens=60))

    models = tracker.get_stats()["models"]
    assert models == {
        "a/x": {"queries": 1, "input_tokens": 100, "output_tokens": 20, "reported_cost": None},
        "b/y": {"queries": 1, "input_tokens": 300, "output_tokens": 60, "reported_cost": None},
    }


def test_event_stats_cost_is_none_when_provider_never_reports_it():
    tracker = MetricsTracker()
    tracker.record(_make_eval(model="a/x", input_tokens=100, output_tokens=20))

    assert tracker.get_stats()["avg_cost_per_query"] is None


def test_sqlite_event_stats_persist_across_instances(tmp_path):
    db_path = tmp_path / "metrics.db"
    tracker = MetricsTracker(SqliteMetricsDB(str(db_path)))
    tracker.record(
        _make_eval(model="a/x", input_tokens=100, output_tokens=20, reported_cost=0.002)
    )

    stats = MetricsTracker(SqliteMetricsDB(str(db_path))).get_stats()

    assert stats["avg_input_tokens"] == 100.0
    assert stats["avg_output_tokens"] == 20.0
    assert stats["avg_cost_per_query"] == 0.002
    assert stats["models"] == {
        "a/x": {"queries": 1, "input_tokens": 100, "output_tokens": 20, "reported_cost": 0.002}
    }


def test_postgres_record_event_persists_query_row(monkeypatch):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    monkeypatch.setattr(db.psycopg, "connect", lambda url, autocommit: connection)

    store = db.PostgresMetricsDB("postgresql://example")
    store.record_event(
        model="a/x",
        input_tokens=100,
        output_tokens=20,
        reported_cost=0.002,
        latency_ms=250.0,
        time_to_first_token_ms=100.0,
        outcome="document_answer",
    )

    assert cursor.execute.call_args_list[-1].args[1] == (
        "a/x",
        100,
        20,
        0.002,
        250.0,
        100.0,
        "document_answer",
    )
    assert "INSERT INTO query_events" in cursor.execute.call_args_list[-1].args[0]
