from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.document_processing.document_processor import DocumentProcessor
from src.core.exceptions import EmptyDocumentError


class _FakeDoc:
    """Minimal spaCy Doc stand-in: no entities, no tokens."""

    ents: list = []

    def __iter__(self):
        return iter([])


@pytest.fixture
def processor():
    """A DocumentProcessor with the Qdrant and spaCy boundaries injected as mocks."""
    vector_store = MagicMock()
    fake_nlp = MagicMock(return_value=_FakeDoc())
    return DocumentProcessor(vector_store, fake_nlp)


def test_chunk_text_drops_short_chunks(processor):
    assert processor._chunk_text("too short") == []


def test_chunk_text_keeps_substantial_chunks(processor):
    text = "Lorem ipsum dolor sit amet. " * 200
    chunks = processor._chunk_text(text)
    assert chunks
    assert all(len(c.strip()) >= 100 for c in chunks)


async def test_process_and_store_rejects_empty_document(processor, tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\t ", encoding="utf-8")
    processor.extractor.extract_from_file = AsyncMock(return_value="   \n\t ")
    with pytest.raises(EmptyDocumentError):
        await processor.process_and_store(str(path), "empty.txt")
    processor.vector_store.add_documents.assert_not_called()


async def test_process_and_store_returns_metadata_and_stores(processor, tmp_path):
    path = tmp_path / "doc.txt"
    body = "Lorem ipsum dolor sit amet. " * 200
    path.write_text(body, encoding="utf-8")
    processor.extractor.extract_from_file = AsyncMock(return_value=body)

    result = await processor.process_and_store(str(path), "doc.txt")

    assert result["filename"] == "doc.txt"
    assert result["chunks_created"] > 0
    assert result["file_size"] == path.stat().st_size
    assert "document_id" in result
    processor.vector_store.add_documents.assert_called_once()
