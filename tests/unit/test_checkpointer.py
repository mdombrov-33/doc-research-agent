from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.config import Settings
from src.core.agent import checkpointer


async def test_opens_sqlite_checkpointer(tmp_path):
    settings = Settings(DATA_DIR=str(tmp_path))

    async with checkpointer.open_checkpointer(settings) as saver:
        assert isinstance(saver, AsyncSqliteSaver)

    assert (tmp_path / "checkpoints.db").exists()


async def test_opens_and_sets_up_postgres_checkpointer(monkeypatch):
    saver = AsyncMock()

    @asynccontextmanager
    async def fake_from_conn_string(url: str):
        assert url == "postgresql://example"
        yield saver

    monkeypatch.setattr(checkpointer.AsyncPostgresSaver, "from_conn_string", fake_from_conn_string)
    settings = Settings(CHECKPOINT_BACKEND="postgres", DATABASE_URL="postgresql://example")

    async with checkpointer.open_checkpointer(settings) as result:
        assert result is saver

    saver.setup.assert_awaited_once()


async def test_postgres_checkpointer_requires_database_url():
    settings = Settings(CHECKPOINT_BACKEND="postgres")

    with pytest.raises(ValueError, match="DATABASE_URL"):
        async with checkpointer.open_checkpointer(settings):
            pass
