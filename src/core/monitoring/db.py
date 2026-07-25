import sqlite3
from pathlib import Path
from typing import Protocol

import psycopg

from src.core.agent.outcomes import FinalOutcome


class MetricsStore(Protocol):
    def load(self) -> dict | None: ...

    def record(
        self,
        sources_retrieved: int,
        web_search_triggered: bool,
        latency_ms: float,
        time_to_first_token_ms: float | None,
        outcome: FinalOutcome,
    ) -> None: ...

    def record_event(
        self,
        model: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        reported_cost: float | None,
        latency_ms: float,
        time_to_first_token_ms: float | None,
        outcome: FinalOutcome,
        cache_hit: bool,
    ) -> None: ...

    def load_event_stats(self) -> dict: ...

    def close(self) -> None: ...


_SQLITE_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS monitoring_stats (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    total_queries INTEGER DEFAULT 0,
    web_search_triggered INTEGER DEFAULT 0,
    total_sources_retrieved INTEGER DEFAULT 0,
    total_latency_ms REAL DEFAULT 0.0,
    total_time_to_first_token_ms REAL DEFAULT 0.0,
    time_to_first_token_samples INTEGER DEFAULT 0,
    document_answers INTEGER DEFAULT 0,
    web_answers INTEGER DEFAULT 0,
    abstentions INTEGER DEFAULT 0,
    conversational_answers INTEGER DEFAULT 0
)
"""

_POSTGRES_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS monitoring_stats (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    total_queries BIGINT DEFAULT 0,
    web_search_triggered BIGINT DEFAULT 0,
    total_sources_retrieved BIGINT DEFAULT 0,
    total_latency_ms DOUBLE PRECISION DEFAULT 0.0,
    total_time_to_first_token_ms DOUBLE PRECISION DEFAULT 0.0,
    time_to_first_token_samples BIGINT DEFAULT 0,
    document_answers BIGINT DEFAULT 0,
    web_answers BIGINT DEFAULT 0,
    abstentions BIGINT DEFAULT 0,
    conversational_answers BIGINT DEFAULT 0
)
"""

_RECORD_SQLITE = """
INSERT INTO monitoring_stats (
    id, total_queries, web_search_triggered,
    total_sources_retrieved, total_latency_ms, total_time_to_first_token_ms,
    time_to_first_token_samples,
    document_answers, web_answers, abstentions, conversational_answers
) VALUES (1, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    total_queries = monitoring_stats.total_queries + excluded.total_queries,
    web_search_triggered = monitoring_stats.web_search_triggered + excluded.web_search_triggered,
    total_sources_retrieved = monitoring_stats.total_sources_retrieved
        + excluded.total_sources_retrieved,
    total_latency_ms = monitoring_stats.total_latency_ms + excluded.total_latency_ms,
    total_time_to_first_token_ms = monitoring_stats.total_time_to_first_token_ms
        + excluded.total_time_to_first_token_ms,
    time_to_first_token_samples = monitoring_stats.time_to_first_token_samples
        + excluded.time_to_first_token_samples,
    document_answers = monitoring_stats.document_answers + excluded.document_answers,
    web_answers = monitoring_stats.web_answers + excluded.web_answers,
    abstentions = monitoring_stats.abstentions + excluded.abstentions,
    conversational_answers = monitoring_stats.conversational_answers
        + excluded.conversational_answers
"""

_RECORD_POSTGRES = _RECORD_SQLITE.replace("?", "%s")

# Per-query event rows. Deliberately no question/answer/evidence content — telemetry
# stores numbers only. reported_cost stays NULL when the provider cost is unavailable
# (LangChain drops OpenRouter's cost field on streamed calls).
_SQLITE_CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS query_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    reported_cost REAL,
    latency_ms REAL NOT NULL,
    time_to_first_token_ms REAL,
    outcome TEXT NOT NULL,
    cache_hit INTEGER NOT NULL DEFAULT 0
)
"""

_POSTGRES_CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS query_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model TEXT,
    input_tokens BIGINT,
    output_tokens BIGINT,
    reported_cost DOUBLE PRECISION,
    latency_ms DOUBLE PRECISION NOT NULL,
    time_to_first_token_ms DOUBLE PRECISION,
    outcome TEXT NOT NULL,
    cache_hit BOOLEAN NOT NULL DEFAULT FALSE
)
"""

_RECORD_EVENT_SQLITE = """
INSERT INTO query_events (
    model, input_tokens, output_tokens, reported_cost,
    latency_ms, time_to_first_token_ms, outcome, cache_hit
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_RECORD_EVENT_POSTGRES = _RECORD_EVENT_SQLITE.replace("?", "%s")

_EVENT_AVERAGES = """
SELECT AVG(input_tokens), AVG(output_tokens), AVG(reported_cost),
       AVG(CASE WHEN cache_hit THEN 1.0 ELSE 0.0 END)
FROM query_events
"""

_EVENT_MODEL_TOTALS = """
SELECT model, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(reported_cost)
FROM query_events WHERE model IS NOT NULL GROUP BY model ORDER BY model
"""

_SELECT = """
SELECT total_queries, web_search_triggered, total_sources_retrieved,
       total_latency_ms, total_time_to_first_token_ms, time_to_first_token_samples,
       document_answers, web_answers, abstentions, conversational_answers
FROM monitoring_stats WHERE id = 1
"""


class SqliteMetricsDB:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_SQLITE_CREATE_TABLE)
            conn.execute(_SQLITE_CREATE_EVENTS_TABLE)
            event_columns = {row[1] for row in conn.execute("PRAGMA table_info(query_events)")}
            if "cache_hit" not in event_columns:
                conn.execute(
                    "ALTER TABLE query_events ADD COLUMN cache_hit INTEGER NOT NULL DEFAULT 0"
                )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(monitoring_stats)")}
            if "total_sources_retrieved" not in columns:
                conn.execute(
                    "ALTER TABLE monitoring_stats "
                    "ADD COLUMN total_sources_retrieved INTEGER DEFAULT 0"
                )
                if "total_docs_retrieved" in columns:
                    conn.execute(
                        "UPDATE monitoring_stats SET total_sources_retrieved = total_docs_retrieved"
                    )
            for column, sql_type in (
                ("total_time_to_first_token_ms", "REAL"),
                ("time_to_first_token_samples", "INTEGER"),
                ("document_answers", "INTEGER"),
                ("web_answers", "INTEGER"),
                ("abstentions", "INTEGER"),
                ("conversational_answers", "INTEGER"),
            ):
                if column not in columns:
                    conn.execute(
                        f"ALTER TABLE monitoring_stats ADD COLUMN {column} {sql_type} DEFAULT 0"
                    )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def load(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT).fetchone()
        return _row_to_totals(row)

    def record(
        self,
        sources_retrieved: int,
        web_search_triggered: bool,
        latency_ms: float,
        time_to_first_token_ms: float | None,
        outcome: FinalOutcome,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                _RECORD_SQLITE,
                (
                    web_search_triggered,
                    sources_retrieved,
                    latency_ms,
                    time_to_first_token_ms or 0.0,
                    time_to_first_token_ms is not None,
                    outcome == "document_answer",
                    outcome == "web_answer",
                    outcome == "abstained",
                    outcome == "conversational",
                ),
            )

    def record_event(
        self,
        model: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        reported_cost: float | None,
        latency_ms: float,
        time_to_first_token_ms: float | None,
        outcome: FinalOutcome,
        cache_hit: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                _RECORD_EVENT_SQLITE,
                (
                    model,
                    input_tokens,
                    output_tokens,
                    reported_cost,
                    latency_ms,
                    time_to_first_token_ms,
                    outcome,
                    cache_hit,
                ),
            )

    def load_event_stats(self) -> dict:
        with self._connect() as conn:
            averages = conn.execute(_EVENT_AVERAGES).fetchone()
            model_rows = conn.execute(_EVENT_MODEL_TOTALS).fetchall()
        return _event_stats_from_rows(averages, model_rows)

    def close(self) -> None:
        pass


class PostgresMetricsDB:
    def __init__(self, database_url: str) -> None:
        self._connection = psycopg.connect(database_url, autocommit=True)
        with self._connection.cursor() as cursor:
            cursor.execute(_POSTGRES_CREATE_TABLE)
            cursor.execute(_POSTGRES_CREATE_EVENTS_TABLE)
            cursor.execute(
                "ALTER TABLE query_events "
                "ADD COLUMN IF NOT EXISTS cache_hit BOOLEAN NOT NULL DEFAULT FALSE"
            )
            for column, sql_type in (
                ("total_time_to_first_token_ms", "DOUBLE PRECISION"),
                ("time_to_first_token_samples", "BIGINT"),
                ("document_answers", "BIGINT"),
                ("web_answers", "BIGINT"),
                ("abstentions", "BIGINT"),
                ("conversational_answers", "BIGINT"),
            ):
                cursor.execute(
                    "ALTER TABLE monitoring_stats "
                    f"ADD COLUMN IF NOT EXISTS {column} {sql_type} DEFAULT 0"
                )

    def load(self) -> dict | None:
        with self._connection.cursor() as cursor:
            cursor.execute(_SELECT)
            row = cursor.fetchone()
        return _row_to_totals(row)

    def record(
        self,
        sources_retrieved: int,
        web_search_triggered: bool,
        latency_ms: float,
        time_to_first_token_ms: float | None,
        outcome: FinalOutcome,
    ) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                _RECORD_POSTGRES,
                (
                    web_search_triggered,
                    sources_retrieved,
                    latency_ms,
                    time_to_first_token_ms or 0.0,
                    time_to_first_token_ms is not None,
                    outcome == "document_answer",
                    outcome == "web_answer",
                    outcome == "abstained",
                    outcome == "conversational",
                ),
            )

    def record_event(
        self,
        model: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        reported_cost: float | None,
        latency_ms: float,
        time_to_first_token_ms: float | None,
        outcome: FinalOutcome,
        cache_hit: bool,
    ) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                _RECORD_EVENT_POSTGRES,
                (
                    model,
                    input_tokens,
                    output_tokens,
                    reported_cost,
                    latency_ms,
                    time_to_first_token_ms,
                    outcome,
                    cache_hit,
                ),
            )

    def load_event_stats(self) -> dict:
        with self._connection.cursor() as cursor:
            cursor.execute(_EVENT_AVERAGES)
            averages = cursor.fetchone()
            cursor.execute(_EVENT_MODEL_TOTALS)
            model_rows = cursor.fetchall()
        return _event_stats_from_rows(averages, model_rows)

    def close(self) -> None:
        self._connection.close()


def _event_stats_from_rows(
    averages: tuple[float | None, float | None, float | None, float | None] | None,
    model_rows: list[tuple],
) -> dict:
    avg_input, avg_output, avg_cost, cache_hit_rate = averages or (None, None, None, None)
    return {
        "avg_input_tokens": avg_input or 0.0,
        "avg_output_tokens": avg_output or 0.0,
        # None (not 0.0) when the provider never reported cost: unknown is not free.
        "avg_cost_per_query": avg_cost,
        "cache_hit_rate": cache_hit_rate or 0.0,
        "models": {
            model: {
                "queries": queries,
                "input_tokens": input_tokens or 0,
                "output_tokens": output_tokens or 0,
                "reported_cost": reported_cost,
            }
            for model, queries, input_tokens, output_tokens, reported_cost in model_rows
        },
    }


def _row_to_totals(
    row: tuple[int, int, int, float, float, int, int, int, int, int] | None,
) -> dict | None:
    if not row:
        return None

    (
        total_queries,
        web_search_triggered,
        total_sources_retrieved,
        total_latency_ms,
        total_time_to_first_token_ms,
        time_to_first_token_samples,
        document_answers,
        web_answers,
        abstentions,
        conversational_answers,
    ) = row
    return {
        "total_queries": total_queries,
        "web_search_triggered": web_search_triggered,
        "total_sources_retrieved": total_sources_retrieved,
        "total_latency_ms": total_latency_ms,
        "total_time_to_first_token_ms": total_time_to_first_token_ms,
        "time_to_first_token_samples": time_to_first_token_samples,
        "document_answers": document_answers,
        "web_answers": web_answers,
        "abstentions": abstentions,
        "conversational_answers": conversational_answers,
    }
