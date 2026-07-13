from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_MODELS: dict[str, str] = {
    "anthropic/claude-fable-5": "Claude Fable 5",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
    "anthropic/claude-opus-4.7": "Claude Opus 4.7",
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "openai/gpt-5.6-terra": "GPT-5.6 Terra",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "openai/gpt-5.5": "GPT 5.5",
    "openai/gpt-5.4-mini": "GPT 5.4 Mini",
    "x-ai/grok-4.5": "Grok 4.5",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
}


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
    LLM_TIMEOUT_SECONDS: float = 60
    WEB_SEARCH_TIMEOUT_SECONDS: int = 10

    QDRANT_MODE: Literal["local", "cloud"] = "cloud"
    QDRANT_LOCAL_URL: str = "http://localhost:6333"
    QDRANT_CLOUD_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "documents"
    QDRANT_QUERY_TIMEOUT_SECONDS: int = 10
    QDRANT_INGESTION_TIMEOUT_SECONDS: int = 30

    @property
    def qdrant_url(self) -> str:
        return self.QDRANT_LOCAL_URL if self.QDRANT_MODE == "local" else self.QDRANT_CLOUD_URL

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    EMBEDDING_TIMEOUT_SECONDS: float = 30
    EMBEDDING_MAX_RETRIES: int = 2

    # Cross-encoder reranking. Hybrid search pulls a wide candidate pool, then a cross-encoder
    # reorders it and we keep the user's top_k. Pool = top_k * MULTIPLIER, capped at FETCH_CAP.
    # Disable to fall back to raw hybrid ordering (fetch exactly top_k, no rerank).
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    RERANK_MULTIPLIER: int = 4
    RERANK_FETCH_CAP: int = 100

    DATA_DIR: str = "./data"
    CHECKPOINT_BACKEND: Literal["sqlite", "postgres"] = "sqlite"
    METRICS_BACKEND: Literal["sqlite", "postgres"] = "sqlite"
    DATABASE_URL: str = ""

    @property
    def metrics_db_path(self) -> str:
        return f"{self.DATA_DIR}/metrics.db"

    @property
    def checkpoints_db_path(self) -> str:
        return f"{self.DATA_DIR}/checkpoints.db"

    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024
    UPLOAD_READ_CHUNK_BYTES: int = 1024 * 1024
    MAX_PDF_PAGES: int = 200
    MAX_EXTRACTED_CHARACTERS: int = 1_000_000
    MAX_CHUNKS_PER_DOCUMENT: int = 1_000

    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT: str = "30/minute"
    RATE_LIMIT_STORAGE_URI: str = "memory://"

    API_URL: str = "http://localhost:8000"

    OTEL_ENDPOINT: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
