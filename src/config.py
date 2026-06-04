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

    OPENAI_API_KEY: str = ""  # used for embeddings only
    OPENROUTER_API_KEY: str = ""
    LLM_MODEL: str = "anthropic/claude-sonnet-4.6"
    LLM_MAX_RETRIES: int = 3

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

    def get_llm_model(self) -> str:
        return self.LLM_MODEL


@lru_cache
def get_settings() -> Settings:
    return Settings()
