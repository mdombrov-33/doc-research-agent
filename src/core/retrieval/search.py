from functools import lru_cache

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from src.config import get_settings
from src.core.vector_store import SPARSE_VECTOR_NAME, get_embeddings, get_qdrant_client
from src.utils.logger import logger


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    settings = get_settings()
    logger.info("vector_store_initialized", collection=settings.QDRANT_COLLECTION_NAME)

    vector_store = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=get_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )

    return vector_store
