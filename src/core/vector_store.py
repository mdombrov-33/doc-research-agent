from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import get_settings
from src.utils.logger import logger

settings = get_settings()


@lru_cache
def get_qdrant_client() -> QdrantClient:
    if settings.QDRANT_MODE == "local":
        logger.info(f"Connecting to local Qdrant at {settings.qdrant_url}")
        return QdrantClient(url=settings.qdrant_url)

    logger.info(f"Connecting to Qdrant Cloud at {settings.qdrant_url}")
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=30,
        https=True,
        port=443,
    )


def get_embeddings() -> OpenAIEmbeddings:
    api_key = settings.get_llm_api_key()
    if not api_key:
        raise ValueError("LLM API key not configured")

    return OpenAIEmbeddings(
        api_key=SecretStr(api_key),
        model=settings.EMBEDDING_MODEL,
    )


def ensure_collection_exists() -> None:
    client = get_qdrant_client()

    if client.collection_exists(settings.QDRANT_COLLECTION_NAME):
        logger.info(f"Collection '{settings.QDRANT_COLLECTION_NAME}' already exists")
        return

    logger.info(f"Creating collection '{settings.QDRANT_COLLECTION_NAME}'")
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=settings.EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        ),
    )
    logger.info(f"Collection '{settings.QDRANT_COLLECTION_NAME}' created successfully")
