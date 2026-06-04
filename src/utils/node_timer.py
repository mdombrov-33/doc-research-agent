import time
from collections.abc import Callable
from typing import Any

from src.core.state import AgentState
from src.utils.logger import logger


def timed(name: str, fn: Callable) -> Callable:
    def wrapper(state: AgentState) -> dict[str, Any]:
        start = time.monotonic()
        result = fn(state)
        logger.info(
            "node_complete", node=name, duration_ms=round((time.monotonic() - start) * 1000, 1)
        )
        return result

    return wrapper
