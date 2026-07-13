from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from pydantic import SecretStr
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

from src.config import get_settings
from src.core.exceptions import EmbeddingConfigError
from src.utils.logger import logger

SPARSE_VECTOR_NAME = "langchain-sparse"


@lru_cache
def _get_qdrant_client(timeout_seconds: int) -> QdrantClient:
    settings = get_settings()
    if settings.QDRANT_MODE == "local":
        logger.info(
            "qdrant_connecting",
            mode="local",
            url=settings.qdrant_url,
            timeout_seconds=timeout_seconds,
        )
        return QdrantClient(url=settings.qdrant_url, timeout=timeout_seconds)

    logger.info(
        "qdrant_connecting",
        mode="cloud",
        url=settings.qdrant_url,
        timeout_seconds=timeout_seconds,
    )
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=timeout_seconds,
        https=True,
        port=443,
    )


@lru_cache(maxsize=1)
def get_ingestion_qdrant_client() -> QdrantClient:
    """The Qdrant client for collection lifecycle and document indexing."""
    return _get_qdrant_client(get_settings().QDRANT_INGESTION_TIMEOUT_SECONDS)


def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise EmbeddingConfigError("LLM API key not configured")

    return OpenAIEmbeddings(
        api_key=SecretStr(api_key),
        model=settings.EMBEDDING_MODEL,
    )


def ensure_collection_exists() -> None:
    settings = get_settings()
    client = get_ingestion_qdrant_client()

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
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        field_name="metadata.entities",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        field_name="metadata.document_id",
        field_schema=PayloadSchemaType.UUID,
    )
    logger.info("qdrant_collection_created", collection=settings.QDRANT_COLLECTION_NAME)


def _get_vector_store(client: QdrantClient) -> QdrantVectorStore:
    settings = get_settings()
    logger.info(
        "vector_store_initialized",
        collection=settings.QDRANT_COLLECTION_NAME,
    )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=get_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


@lru_cache(maxsize=1)
def get_retrieval_vector_store() -> QdrantVectorStore:
    """The configured hybrid store for user queries.

    HYBRID = dense OpenAI embeddings + sparse BM25, fused server-side by Qdrant (RRF). The
    sparse vectors this relies on are declared in ensure_collection_exists above.
    """
    settings = get_settings()
    return _get_vector_store(_get_qdrant_client(settings.QDRANT_QUERY_TIMEOUT_SECONDS))


@lru_cache(maxsize=1)
def get_ingestion_vector_store() -> QdrantVectorStore:
    """The configured hybrid store for indexing uploaded documents."""
    return _get_vector_store(get_ingestion_qdrant_client())
