from collections.abc import Awaitable, Callable
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig  # type: ignore[import-untyped]

from src.utils.logger import logger


class GuardrailsWrapper:
    """Wraps the RAG agent with NeMo Guardrails for security."""

    def __init__(self) -> None:
        """Initialize guardrails with config from src/guardrails/."""
        config_path = Path(__file__).parent
        logger.info(f"Loading NeMo Guardrails config from {config_path}")

        self.config = RailsConfig.from_path(str(config_path))
        self.rails = LLMRails(self.config)
        logger.info("NeMo Guardrails initialized successfully")

    async def generate_safe(self, user_message: str) -> str:
        """
        Process user message through guardrails.

        Args:
            user_message: User's query

        Returns:
            Bot response (filtered if needed)
        """
        try:
            response = await self.rails.generate_async(
                messages=[{"role": "user", "content": user_message}]
            )

            logger.info(f"Guardrails response type: {type(response)}")
            logger.info(f"Guardrails response: {response}")

            if isinstance(response, dict):
                content = str(response.get("content", ""))
                logger.info(f"Extracted from dict: {content[:100]}")
                return content
            elif hasattr(response, "content"):
                content = str(getattr(response, "content"))
                logger.info(f"Extracted from attribute: {content[:100]}")
                return content
            else:
                content = str(response)
                logger.info(f"Converted to string: {content[:100]}")
                return content

        except Exception as e:
            logger.error(f"Guardrails error: {e}", exc_info=True)
            return "I encountered an error processing your request. Please try again."

    def register_rag_action(self, rag_function: Callable[[str], Awaitable[str]]) -> None:
        """Register the RAG agent as a custom action."""

        async def rag_query_action(question: str = "") -> str:
            logger.info(f"Calling RAG with question: '{question[:100]}'")

            try:
                answer = await rag_function(question)
                logger.info(f"RAG returned answer: {len(answer)} chars")
                return answer
            except Exception as e:
                logger.error(f"RAG error: {e}", exc_info=True)
                return "I encountered an error searching the documents."

        self.rails.register_action(rag_query_action, name="rag_query")
        logger.info("Registered RAG action")


_instance: GuardrailsWrapper | None = None


def get_guardrails() -> GuardrailsWrapper:
    """Get singleton guardrails instance."""
    global _instance
    if _instance is None:
        _instance = GuardrailsWrapper()
    return _instance
