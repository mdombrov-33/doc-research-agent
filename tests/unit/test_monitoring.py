import sqlite3
from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.core.monitoring import db
from src.core.monitoring.db import SqliteMetricsDB
from src.core.monitoring.tracker import MetricsTracker, QueryMetrics


def _make_eval(
    sources_retrieved: int = 5,
    web_search_triggered: bool = False,
    latency_ms: float = 1000.0,
) -> QueryMetrics:
    return QueryMetrics(
        question="test question",
        sources_retrieved=sources_retrieved,
        web_search_triggered=web_search_triggered,
        latency_ms=latency_ms,
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
        conn.execute(
            "INSERT INTO monitoring_stats VALUES (1, 2, 1, 6, 5, 3000.0)"
        )

    stats = MetricsTracker(SqliteMetricsDB(str(db_path))).get_stats()

    assert stats == {
        "total_queries": 2,
        "web_search_rate": 0.5,
        "avg_sources_retrieved": 3.0,
        "avg_latency_ms": 1500.0,
    }


def test_sqlite_metrics_persist_recorded_totals(tmp_path):
    db_path = tmp_path / "metrics.db"
    tracker = MetricsTracker(SqliteMetricsDB(str(db_path)))
    tracker.record(_make_eval(sources_retrieved=4, web_search_triggered=True, latency_ms=500.0))

    stats = MetricsTracker(SqliteMetricsDB(str(db_path))).get_stats()

    assert stats == {
        "total_queries": 1,
        "web_search_rate": 1.0,
        "avg_sources_retrieved": 4.0,
        "avg_latency_ms": 500.0,
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


def test_avg_latency():
    tracker = MetricsTracker()
    tracker.record(_make_eval(latency_ms=1000.0))
    tracker.record(_make_eval(latency_ms=3000.0))
    stats = tracker.get_stats()
    assert stats["avg_latency_ms"] == 2000.0


def test_postgres_metrics_record_uses_atomic_increment(monkeypatch):
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    monkeypatch.setattr(db.psycopg, "connect", lambda url, autocommit: connection)

    store = db.PostgresMetricsDB("postgresql://example")
    store.record(sources_retrieved=3, web_search_triggered=True, latency_ms=250.0)

    assert cursor.execute.call_args_list[1].args[1] == (True, 3, 250.0)
    assert "total_queries = monitoring_stats.total_queries + excluded.total_queries" in (
        cursor.execute.call_args_list[1].args[0]
    )


def test_postgres_metrics_configuration_requires_database_url():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        MetricsTracker.from_settings(Settings(METRICS_BACKEND="postgres"))
