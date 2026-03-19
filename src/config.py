from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_ENV: Literal["development", "production", "test"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    LLM_PROVIDER: Literal["openai", "openrouter"] = "openai"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5"

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "openai/gpt-5"

    QDRANT_MODE: Literal["local", "cloud"] = "cloud"
    QDRANT_LOCAL_URL: str = "http://localhost:6333"
    QDRANT_CLOUD_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "documents"

    @property
    def qdrant_url(self) -> str:
        return self.QDRANT_LOCAL_URL if self.QDRANT_MODE == "local" else self.QDRANT_CLOUD_URL

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    METRICS_DB_PATH: str = "./data/metrics.db"
    UPLOAD_DIR: str = "./uploads"

    API_URL: str = "http://localhost:8000"

    def get_llm_api_key(self) -> str:
        if self.LLM_PROVIDER == "openai":
            return self.OPENAI_API_KEY
        return self.OPENROUTER_API_KEY

    def get_llm_model(self) -> str:
        if self.LLM_PROVIDER == "openai":
            return self.OPENAI_MODEL
        return self.OPENROUTER_MODEL


@lru_cache
def get_settings() -> Settings:
    return Settings()
