from collections.abc import Callable

from src.utils.logger import logger


def with_retry[T](call: Callable[[], T], *, attempts: int = 2) -> T:
    """Retry a structured-output call on parse/validation failures.

    ChatOpenAI's max_retries covers transient API errors but not malformed model
    output; this fills that gap with a small bounded retry.
    """
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == attempts:
                raise
            logger.warning(
                "structured_output_retry", attempt=attempt, failure_type=type(exc).__name__
            )
    raise AssertionError("unreachable")
