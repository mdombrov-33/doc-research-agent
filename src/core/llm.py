from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import get_settings
from src.utils.logger import logger

settings = get_settings()


def get_llm(model_override: str | None = None, temperature: float = 0):
    model = model_override or settings.get_llm_model()
    logger.info(f"Using model: {model}")

    if model_override and "/" in model_override:
        return ChatOpenAI(
            api_key=SecretStr(settings.OPENROUTER_API_KEY),
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=temperature,
        )
    elif settings.LLM_PROVIDER == "openrouter":
        return ChatOpenAI(
            api_key=SecretStr(settings.OPENROUTER_API_KEY),
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=temperature,
        )
    else:
        return ChatOpenAI(
            api_key=SecretStr(settings.OPENAI_API_KEY),
            model=model,
            temperature=temperature,
        )