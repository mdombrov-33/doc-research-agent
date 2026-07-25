from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Settings
from src.core.exceptions import DocumentLimitError, EmptyDocumentError
from src.core.ingestion import pipeline


class _FakeDoc:
    """Minimal spaCy Doc stand-in: no entities, no tokens."""

    ents: list = []

    def __iter__(self):
        return iter([])


@pytest.fixture
def vector_store():
    return MagicMock()


@pytest.fixture
def nlp():
    """spaCy stand-in: callable returning an empty Doc."""
    return MagicMock(return_value=_FakeDoc())


async def test_process_and_store_rejects_empty_document(vector_store, nlp, tmp_path, monkeypatch):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\t ", encoding="utf-8")
    monkeypatch.setattr(pipeline, "extract_from_file", AsyncMock(return_value="   \n\t "))
    with pytest.raises(EmptyDocumentError):
        await pipeline.process_and_store(
            str(path), "empty.txt", "sha", vector_store, nlp, Settings()
        )
    vector_store.add_documents.assert_not_called()


async def test_process_and_store_rejects_too_many_chunks(vector_store, nlp, tmp_path, monkeypatch):
    path = tmp_path / "doc.txt"
    path.write_text("document", encoding="utf-8")
    monkeypatch.setattr(pipeline, "extract_from_file", AsyncMock(return_value="document"))
    monkeypatch.setattr(pipeline, "chunk_text", lambda _: ["one", "two", "three"])

    with pytest.raises(DocumentLimitError):
        await pipeline.process_and_store(
            str(path),
            "doc.txt",
            "sha",
            vector_store,
            nlp,
            Settings(MAX_CHUNKS_PER_DOCUMENT=2),
        )

    vector_store.add_documents.assert_not_called()


async def test_process_and_store_returns_metadata_and_stores(
    vector_store, nlp, tmp_path, monkeypatch
):
    path = tmp_path / "doc.txt"
    body = "Lorem ipsum dolor sit amet. " * 200
    path.write_text(body, encoding="utf-8")
    logger = MagicMock()
    monkeypatch.setattr(pipeline, "extract_from_file", AsyncMock(return_value=body))
    monkeypatch.setattr(pipeline, "logger", logger)

    result = await pipeline.process_and_store(
        str(path), "doc.txt", "sha-256-digest", vector_store, nlp, Settings()
    )

    assert result["filename"] == "doc.txt"
    assert result["chunks_created"] > 0
    assert result["file_size"] == path.stat().st_size
    assert "document_id" in result
    assert logger.info.call_args.kwargs == {
        "document_id": result["document_id"],
        "file_extension": ".txt",
        "chunks": result["chunks_created"],
    }
    vector_store.add_documents.assert_called_once()
    stored_documents = vector_store.add_documents.call_args.args[0]
    assert stored_documents[0].metadata["chunk_id"] == (
        f"{result['document_id']}:{stored_documents[0].metadata['chunk_index']}"
    )
    assert stored_documents[0].metadata["file_sha256"] == "sha-256-digest"
    assert stored_documents[0].page_content.startswith("Document: doc.txt\n\n")
