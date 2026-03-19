import asyncio
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig  # type: ignore[import-untyped]

from src.utils.logger import logger


class GuardrailsWrapper:
    def __init__(self) -> None:
        config_path = Path(__file__).parent
        self.config = RailsConfig.from_path(str(config_path))
        self.rails = LLMRails(self.config)
        self._lock = asyncio.Lock()
        logger.info("NeMo Guardrails initialized")

    async def check_input(self, question: str) -> str | None:
        """Returns refusal message if NeMo blocks the input, None if safe."""
        async with self._lock:
            safe = asyncio.Event()

            async def _probe(**kwargs) -> str:
                safe.set()
                return "__SAFE__"

            self.rails.register_action(_probe, name="rag_query")

            try:
                response = await self.rails.generate_async(
                    messages=[{"role": "user", "content": question}]
                )
            except Exception as e:
                logger.error(f"NeMo input check error: {e}")
                return None  # fail open

            if safe.is_set():
                return None

            content = response.get("content", "") if isinstance(response, dict) else str(response)
            logger.info(f"NeMo blocked input: {content[:100]}")
            return content or "I cannot process that request."

    async def check_output(self, response: str) -> str | None:
        """Returns correction message if NeMo self_check_output flags the response, None if safe."""
        try:
            prompt = self.rails.runtime.llm_task_manager.render_task_prompt(
                task="self_check_output",
                context={"bot_response": response},
            )
            result = await self.rails.llm.ainvoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)
            if "yes" in content.strip().lower():
                logger.info(f"NeMo flagged output: {content[:100]}")
                return "I can only provide information related to document research and factual analysis."
            return None
        except Exception as e:
            logger.error(f"NeMo output check error: {e}")
            return None  # fail open


_instance: GuardrailsWrapper | None = None


def get_guardrails() -> GuardrailsWrapper:
    global _instance
    if _instance is None:
        _instance = GuardrailsWrapper()
    return _instance
