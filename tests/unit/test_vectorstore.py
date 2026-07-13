from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.core import vectorstore


@pytest.fixture(autouse=True)
def clear_vectorstore_caches():
    vectorstore._get_qdrant_client.cache_clear()
    vectorstore.get_ingestion_qdrant_client.cache_clear()
    vectorstore.get_retrieval_vector_store.cache_clear()
    vectorstore.get_ingestion_vector_store.cache_clear()
    yield
    vectorstore._get_qdrant_client.cache_clear()
    vectorstore.get_ingestion_qdrant_client.cache_clear()
    vectorstore.get_retrieval_vector_store.cache_clear()
    vectorstore.get_ingestion_vector_store.cache_clear()


def test_vector_stores_use_operation_specific_timeouts(monkeypatch):
    settings = Settings(
        QDRANT_MODE="local",
        QDRANT_QUERY_TIMEOUT_SECONDS=11,
        QDRANT_INGESTION_TIMEOUT_SECONDS=31,
    )
    client = MagicMock()
    vector_store = MagicMock()
    monkeypatch.setattr(vectorstore, "get_settings", lambda: settings)
    monkeypatch.setattr(vectorstore, "QdrantClient", client)
    monkeypatch.setattr(vectorstore, "QdrantVectorStore", vector_store)
    monkeypatch.setattr(vectorstore, "get_embeddings", MagicMock())

    vectorstore.get_retrieval_vector_store()
    vectorstore.get_ingestion_vector_store()

    assert client.call_args_list[0].kwargs["timeout"] == 11
    assert client.call_args_list[1].kwargs["timeout"] == 31


def test_embeddings_use_configured_timeout_and_retry_limit(monkeypatch):
    settings = Settings(
        OPENAI_API_KEY="test-key",
        EMBEDDING_TIMEOUT_SECONDS=31,
        EMBEDDING_MAX_RETRIES=1,
    )
    embeddings = MagicMock()
    monkeypatch.setattr(vectorstore, "get_settings", lambda: settings)
    monkeypatch.setattr(vectorstore, "OpenAIEmbeddings", embeddings)

    vectorstore.get_embeddings()

    assert embeddings.call_args.kwargs["timeout"] == 31
    assert embeddings.call_args.kwargs["max_retries"] == 1


def test_qdrant_readiness_checks_the_configured_collection_without_mutating(monkeypatch):
    settings = Settings(
        QDRANT_MODE="local",
        QDRANT_COLLECTION_NAME="ready-documents",
        QDRANT_QUERY_TIMEOUT_SECONDS=11,
    )
    client = MagicMock()
    client.collection_exists.return_value = True
    monkeypatch.setattr(vectorstore, "get_settings", lambda: settings)
    monkeypatch.setattr(vectorstore, "_get_qdrant_client", MagicMock(return_value=client))

    assert vectorstore.is_qdrant_ready() is True
    vectorstore._get_qdrant_client.assert_called_once_with(11)
    client.collection_exists.assert_called_once_with("ready-documents")
    client.create_collection.assert_not_called()
