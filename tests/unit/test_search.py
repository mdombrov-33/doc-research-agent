from types import SimpleNamespace
from unittest.mock import MagicMock

from qdrant_client import models

from src.core.retrieval import search


class _Doc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


# --------------------------- _entity_filter ---------------------------


def test_entity_filter_none_for_empty():
    assert search._entity_filter([]) is None


def test_entity_filter_builds_match_any():
    flt = search._entity_filter(["Acme", "Bob"])
    expected = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.entities",
                match=models.MatchAny(any=["Acme", "Bob"]),
            )
        ]
    )
    assert flt == expected


# --------------------------- _fetch_k ---------------------------


def test_fetch_k_scales_with_top_k_and_is_capped():
    s = SimpleNamespace(RERANK_ENABLED=True, RERANK_MULTIPLIER=4, RERANK_FETCH_CAP=100)
    assert search._fetch_k(5, s) == 20
    assert search._fetch_k(40, s) == 100  # 40 * 4 = 160, capped to 100


def test_fetch_k_returns_top_k_when_reranking_disabled():
    s = SimpleNamespace(RERANK_ENABLED=False, RERANK_MULTIPLIER=4, RERANK_FETCH_CAP=100)
    assert search._fetch_k(5, s) == 5


# --------------------------- hybrid_search ---------------------------


def test_hybrid_search_maps_results_and_skips_blank(monkeypatch):
    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (_Doc("real content", {"filename": "a.pdf", "chunk_index": 2, "chunk_length": 12}), 0.9),
        (_Doc("   ", {"filename": "blank.pdf"}), 0.5),  # blank → skipped
    ]
    monkeypatch.setattr(search, "get_vector_store", lambda: store)
    monkeypatch.setattr(search, "extract_entities", lambda q: [])
    monkeypatch.setattr(search, "rerank", lambda q, docs, k: docs[:k])

    out = search.hybrid_search("q", top_k=5)

    assert len(out) == 1
    assert out[0]["filename"] == "a.pdf"
    assert out[0]["source"] == "vectorstore"


def test_hybrid_search_falls_back_to_unfiltered_when_filter_empty(monkeypatch):
    store = MagicMock()
    # First (filtered) call returns nothing, second (unfiltered) returns a hit.
    store.similarity_search_with_score.side_effect = [
        [],
        [(_Doc("fallback hit", {"filename": "b.txt"}), 0.8)],
    ]
    monkeypatch.setattr(search, "get_vector_store", lambda: store)
    monkeypatch.setattr(search, "extract_entities", lambda q: ["Acme"])
    monkeypatch.setattr(search, "rerank", lambda q, docs, k: docs[:k])

    out = search.hybrid_search("q", top_k=5)

    assert store.similarity_search_with_score.call_count == 2
    assert len(out) == 1
    # The fallback re-query drops the filter.
    assert store.similarity_search_with_score.call_args.kwargs.get("filter") is None


def test_hybrid_search_overfetches_pool_and_truncates_to_top_k(monkeypatch):
    store = MagicMock()
    # Eight candidate chunks come back from hybrid search, in retrieval order.
    store.similarity_search_with_score.return_value = [
        (_Doc(f"doc {i}", {"filename": f"{i}.txt"}), 1.0 - i * 0.1) for i in range(8)
    ]
    monkeypatch.setattr(search, "get_vector_store", lambda: store)
    monkeypatch.setattr(search, "extract_entities", lambda q: [])

    captured = {}

    def fake_rerank(question, docs, top_k):
        captured["candidates"] = len(docs)
        captured["top_k"] = top_k
        return docs[:top_k]

    monkeypatch.setattr(search, "rerank", fake_rerank)

    out = search.hybrid_search("q", top_k=3)

    # The pool requested from the store scales with top_k (3 * default multiplier 4 = 12).
    assert store.similarity_search_with_score.call_args.kwargs["k"] == 12
    # The reranker saw every returned candidate and was asked for the user's top_k.
    assert captured["candidates"] == 8
    assert captured["top_k"] == 3
    # The final result is truncated to the user's top_k.
    assert len(out) == 3
