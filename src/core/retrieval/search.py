from functools import lru_cache

from langchain_qdrant import QdrantVectorStore

from src.config import get_settings
from src.core.vector_store import get_embeddings, get_qdrant_client
from src.utils.logger import logger

settings = get_settings()


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    logger.info("vector_store_initialized", collection=settings.QDRANT_COLLECTION_NAME)

    vector_store = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=get_embeddings(),
    )

    return vector_store
