import pytest

from src.core.retrieval import rerank as reranker


class _FakeEncoder:
    """Stands in for the cross-encoder: returns preset scores in document order."""

    def __init__(self, scores):
        self._scores = scores

    def rerank(self, query, documents, **kwargs):
        return list(self._scores)


def _docs(*contents):
    return [{"content": c, "filename": f"{i}.txt"} for i, c in enumerate(contents)]


def test_rerank_reorders_by_cross_encoder_score(monkeypatch):
    docs = _docs("a", "b", "c")
    # Retrieval order is a, b, c; the cross-encoder rates c highest and a lowest.
    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda model: _FakeEncoder([0.1, 0.5, 0.9]))

    out = reranker.rerank("q", docs, top_k=3)

    assert [d["content"] for d in out] == ["c", "b", "a"]


def test_rerank_truncates_to_top_k(monkeypatch):
    docs = _docs("a", "b", "c", "d")
    monkeypatch.setattr(
        reranker, "_get_cross_encoder", lambda model: _FakeEncoder([0.4, 0.3, 0.2, 0.1])
    )

    out = reranker.rerank("q", docs, top_k=2)

    assert [d["content"] for d in out] == ["a", "b"]


def test_rerank_promotes_a_buried_hit(monkeypatch):
    # The best document sits last in retrieval order; reranking must surface it into top_k.
    docs = _docs("a", "b", "c", "d")
    monkeypatch.setattr(
        reranker, "_get_cross_encoder", lambda model: _FakeEncoder([0.1, 0.2, 0.3, 0.9])
    )

    out = reranker.rerank("q", docs, top_k=2)

    assert [d["content"] for d in out] == ["d", "c"]


def test_rerank_empty_returns_empty_without_loading_model(monkeypatch):
    monkeypatch.setattr(
        reranker, "_get_cross_encoder", lambda model: pytest.fail("must not load the encoder")
    )

    assert reranker.rerank("q", [], top_k=5) == []
