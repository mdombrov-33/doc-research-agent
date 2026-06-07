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
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
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
    # Index the spaCy entities so retrieval can filter on them efficiently.
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        field_name="metadata.entities",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info("qdrant_collection_created", collection=settings.QDRANT_COLLECTION_NAME)


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    """The configured hybrid store, shared by ingestion (add_documents) and retrieval (search).

    HYBRID = dense OpenAI embeddings + sparse BM25, fused server-side by Qdrant (RRF). The
    sparse vectors this relies on are declared in ensure_collection_exists above.
    """
    settings = get_settings()
    logger.info("vector_store_initialized", collection=settings.QDRANT_COLLECTION_NAME)

    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=get_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )
