import asyncio
from functools import lru_cache

from llm_guard import scan_prompt  # type: ignore[import-untyped]
from llm_guard.input_scanners import PromptInjection, Toxicity  # type: ignore[import-untyped]

from src.utils.logger import logger

_BLOCK_INPUT = "I cannot process that request. Please ask a question about your documents."


@lru_cache(maxsize=1)
def _get_scanners() -> list:
    # Loads HuggingFace models once. warmup() primes this at startup so the first real
    # request doesn't pay the cold-start.
    scanners = [Toxicity(), PromptInjection()]
    logger.info("guardrails_initialized")
    return scanners


def warmup() -> None:
    _get_scanners()


def _is_flagged(text: str) -> bool:
    _, results_valid, _ = scan_prompt(_get_scanners(), text)
    return not all(results_valid.values())


async def check_input(question: str) -> str | None:
    loop = asyncio.get_running_loop()
    try:
        flagged = await loop.run_in_executor(None, _is_flagged, question)
    except Exception as e:
        # Local scanners rarely throw, but if they do, fail closed (block) rather than let the
        # exception break the SSE stream — check_input runs before the handler's try/except.
        logger.error("guardrails_input_error", error=str(e))
        return _BLOCK_INPUT
    if flagged:
        logger.info("guardrails_input_blocked", preview=question[:100])
        return _BLOCK_INPUT
    return None
