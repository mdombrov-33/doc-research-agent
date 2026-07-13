import asyncio
from functools import lru_cache

from llm_guard import scan_prompt  # type: ignore[import-untyped]
from llm_guard.input_scanners import Toxicity  # type: ignore[import-untyped]
from opentelemetry import trace

from src.utils.logger import logger

_tracer = trace.get_tracer(__name__)

_BLOCK_INPUT = "I cannot process that request. Please ask a question about your documents."


@lru_cache(maxsize=1)
def _get_scanners() -> list:
    # Loads HuggingFace models once. warmup() primes this at startup so the first real
    # request doesn't pay the cold-start.
    scanners = [Toxicity()]
    logger.info("guardrails_initialized")
    return scanners


def warmup() -> None:
    _get_scanners()


def _is_flagged(text: str) -> bool:
    _, results_valid, _ = scan_prompt(_get_scanners(), text)
    return not all(results_valid.values())


async def check_input(question: str) -> str | None:
    with _tracer.start_as_current_span("guardrails.scan") as span:
        loop = asyncio.get_running_loop()
        try:
            flagged = await loop.run_in_executor(None, _is_flagged, question)
        except Exception as error:
            span.set_status(trace.StatusCode.ERROR, "guardrails scan failed")
            logger.error("guardrails_input_error", failure_type=type(error).__name__)
            return _BLOCK_INPUT
        span.set_attribute("guardrails.flagged", flagged)
        if flagged:
            logger.info("guardrails_input_blocked")
            return _BLOCK_INPUT
    return None
