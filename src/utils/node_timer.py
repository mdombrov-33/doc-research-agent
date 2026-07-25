import time
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from src.core.agent.state import AgentState
from src.utils.logger import logger


def timed(name: str, fn: Callable) -> Callable:
    # Two-arg signature lets LangGraph pass invocation-scoped answer-model configuration.
    def wrapper(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        start = time.monotonic()
        result = fn(state, config)
        logger.info(
            "node_complete", node=name, duration_ms=round((time.monotonic() - start) * 1000, 1)
        )
        return result

    return wrapper
