import sqlite3
from pathlib import Path


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS eval_stats (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    total_queries INTEGER DEFAULT 0,
    web_search_triggered INTEGER DEFAULT 0,
    total_docs_retrieved INTEGER DEFAULT 0,
    total_docs_relevant INTEGER DEFAULT 0,
    total_latency_ms REAL DEFAULT 0.0
)
"""

_UPSERT = """
INSERT INTO eval_stats (
    id, total_queries, web_search_triggered,
    total_docs_retrieved, total_docs_relevant, total_latency_ms
) VALUES (1, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    total_queries = excluded.total_queries,
    web_search_triggered = excluded.web_search_triggered,
    total_docs_retrieved = excluded.total_docs_retrieved,
    total_docs_relevant = excluded.total_docs_relevant,
    total_latency_ms = excluded.total_latency_ms
"""

_SELECT = """
SELECT total_queries, web_search_triggered, total_docs_retrieved,
       total_docs_relevant, total_latency_ms
FROM eval_stats WHERE id = 1
"""

_MIGRATE = [
    "ALTER TABLE eval_stats DROP COLUMN hallucination_passed",
    "ALTER TABLE eval_stats DROP COLUMN quality_passed",
    "ALTER TABLE eval_stats DROP COLUMN total_generation_attempts",
]


class EvalDB:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            for stmt in _MIGRATE:
                try:
                    conn.execute(stmt)
                except Exception:
                    pass  # column already gone or never existed

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def load(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT).fetchone()
        if not row:
            return None
        tq, ws, tdr, tdrel, tlat = row
        return {
            "total_queries": tq,
            "web_search_triggered": ws,
            "total_docs_retrieved": tdr,
            "total_docs_relevant": tdrel,
            "total_latency_ms": tlat,
        }

    def flush(
        self,
        total_queries: int,
        web_search_triggered: int,
        total_docs_retrieved: int,
        total_docs_relevant: int,
        total_latency_ms: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                _UPSERT,
                (
                    total_queries,
                    web_search_triggered,
                    total_docs_retrieved,
                    total_docs_relevant,
                    total_latency_ms,
                ),
            )
