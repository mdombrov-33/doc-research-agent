from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.ingestion.dedupe import find_duplicate


def _vector_store(scroll_return):
    client = MagicMock()
    client.scroll.return_value = scroll_return
    return SimpleNamespace(client=client, collection_name="documents")


def test_find_duplicate_returns_document_id_on_match():
    point = SimpleNamespace(payload={"metadata": {"document_id": "doc-42"}})
    store = _vector_store(([point], None))

    assert find_duplicate(store, "abc123") == "doc-42"

    scroll_filter = store.client.scroll.call_args.kwargs["scroll_filter"]
    condition = scroll_filter.must[0]
    assert condition.key == "metadata.file_sha256"
    assert condition.match.value == "abc123"


def test_find_duplicate_returns_none_when_no_points():
    store = _vector_store(([], None))
    assert find_duplicate(store, "abc123") is None
