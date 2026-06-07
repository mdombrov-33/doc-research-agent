from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import get_settings
from src.utils.logger import logger

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm(model_override: str | None = None, temperature: float = 0) -> ChatOpenAI:
    settings = get_settings()
    model = model_override or settings.LLM_MODEL
    logger.debug("llm_selected", model=model)

    return ChatOpenAI(
        api_key=SecretStr(settings.OPENROUTER_API_KEY),
        base_url=OPENROUTER_BASE_URL,
        model=model,
        temperature=temperature,
        max_retries=settings.LLM_MAX_RETRIES,
    )
