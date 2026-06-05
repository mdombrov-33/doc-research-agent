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

    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    LLM_MODEL: str = "anthropic/claude-sonnet-4.6"
    CLASSIFIER_MODEL: str = "openai/gpt-5.4-mini"
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

    # Cross-encoder reranking. Hybrid search pulls a wide candidate pool, then a cross-encoder
    # reorders it and we keep the user's top_k. Pool = top_k * MULTIPLIER, capped at FETCH_CAP.
    # Disable to fall back to raw hybrid ordering (fetch exactly top_k, no rerank).
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    RERANK_MULTIPLIER: int = 4
    RERANK_FETCH_CAP: int = 100

    METRICS_DB_PATH: str = "./data/metrics.db"
    UPLOAD_DIR: str = "./uploads"

    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT: str = "30/minute"
    RATE_LIMIT_STORAGE_URI: str = "memory://"

    API_URL: str = "http://localhost:8000"

    def get_llm_model(self) -> str:
        return self.LLM_MODEL


@lru_cache
def get_settings() -> Settings:
    return Settings()
