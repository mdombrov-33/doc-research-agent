from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.config import Settings


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    if settings.CHECKPOINT_BACKEND == "postgres":
        if not settings.DATABASE_URL:
            raise ValueError("DATABASE_URL is required for the Postgres checkpointer")

        async with AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
        return

    db_path = settings.checkpoints_db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(db_path)
    try:
        yield AsyncSqliteSaver(connection)
    finally:
        await connection.close()
