from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    SparseVectorParams,
    VectorParams,
)

from src.config import get_settings
from src.core.exceptions import EmbeddingConfigError
from src.utils.logger import logger

settings = get_settings()


@lru_cache
def get_qdrant_client() -> QdrantClient:
    if settings.QDRANT_MODE == "local":
        logger.info("qdrant_connecting", mode="local", url=settings.qdrant_url)
        return QdrantClient(url=settings.qdrant_url)

    logger.info("qdrant_connecting", mode="cloud", url=settings.qdrant_url)
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=30,
        https=True,
        port=443,
    )


def get_embeddings() -> OpenAIEmbeddings:
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise EmbeddingConfigError("LLM API key not configured")

    return OpenAIEmbeddings(
        api_key=SecretStr(api_key),
        model=settings.EMBEDDING_MODEL,
    )


# Must match the sparse vector name used by langchain_qdrant's QdrantVectorStore.
SPARSE_VECTOR_NAME = "langchain-sparse"


def ensure_collection_exists() -> None:
    client = get_qdrant_client()

    if client.collection_exists(settings.QDRANT_COLLECTION_NAME):
        logger.info("qdrant_collection_exists", collection=settings.QDRANT_COLLECTION_NAME)
        return

    logger.info("qdrant_collection_creating", collection=settings.QDRANT_COLLECTION_NAME)
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=settings.EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        ),
        sparse_vectors_config={
            # modifier=IDF lets Qdrant compute BM25 IDF from corpus statistics server-side.
            SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
        },
    )
    logger.info("qdrant_collection_created", collection=settings.QDRANT_COLLECTION_NAME)
