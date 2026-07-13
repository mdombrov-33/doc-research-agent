import sqlite3
from pathlib import Path

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS monitoring_stats (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    total_queries INTEGER DEFAULT 0,
    web_search_triggered INTEGER DEFAULT 0,
    total_sources_retrieved INTEGER DEFAULT 0,
    total_latency_ms REAL DEFAULT 0.0
)
"""

_UPSERT = """
INSERT INTO monitoring_stats (
    id, total_queries, web_search_triggered,
    total_sources_retrieved, total_latency_ms
) VALUES (1, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    total_queries = excluded.total_queries,
    web_search_triggered = excluded.web_search_triggered,
    total_sources_retrieved = excluded.total_sources_retrieved,
    total_latency_ms = excluded.total_latency_ms
"""

_SELECT = """
SELECT total_queries, web_search_triggered, total_sources_retrieved,
       total_latency_ms
FROM monitoring_stats WHERE id = 1
"""


class MetricsDB:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(monitoring_stats)")}
            if "total_sources_retrieved" not in columns:
                conn.execute(
                    "ALTER TABLE monitoring_stats "
                    "ADD COLUMN total_sources_retrieved INTEGER DEFAULT 0"
                )
                if "total_docs_retrieved" in columns:
                    conn.execute(
                        "UPDATE monitoring_stats "
                        "SET total_sources_retrieved = total_docs_retrieved"
                    )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def load(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT).fetchone()
        if not row:
            return None
        tq, ws, tsr, tlat = row
        return {
            "total_queries": tq,
            "web_search_triggered": ws,
            "total_sources_retrieved": tsr,
            "total_latency_ms": tlat,
        }

    def flush(
        self,
        total_queries: int,
        web_search_triggered: int,
        total_sources_retrieved: int,
        total_latency_ms: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                _UPSERT,
                (
                    total_queries,
                    web_search_triggered,
                    total_sources_retrieved,
                    total_latency_ms,
                ),
            )
