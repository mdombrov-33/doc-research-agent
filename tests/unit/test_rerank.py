from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.core.retrieval import rerank as reranker


def _with_floor(monkeypatch, floor):
    monkeypatch.setattr(reranker, "get_settings", lambda: Settings(RERANK_SCORE_FLOOR=floor))


def _with_scores(monkeypatch, scores):
    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda model: _FakeEncoder(scores))


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

    out = reranker.rerank("q", docs, limit=3)

    assert [d["content"] for d in out] == ["c", "b", "a"]


def test_rerank_truncates_to_evidence_limit(monkeypatch):
    docs = _docs("a", "b", "c", "d")
    monkeypatch.setattr(
        reranker, "_get_cross_encoder", lambda model: _FakeEncoder([0.4, 0.3, 0.2, 0.1])
    )

    out = reranker.rerank("q", docs, limit=2)

    assert [d["content"] for d in out] == ["a", "b"]


def test_rerank_promotes_a_buried_hit(monkeypatch):
    # The best document sits last in retrieval order; reranking must surface it into evidence.
    docs = _docs("a", "b", "c", "d")
    monkeypatch.setattr(
        reranker, "_get_cross_encoder", lambda model: _FakeEncoder([0.1, 0.2, 0.3, 0.9])
    )

    out = reranker.rerank("q", docs, limit=2)

    assert [d["content"] for d in out] == ["d", "c"]


def test_multipart_rerank_preserves_bounded_coverage_from_each_branch(monkeypatch):
    docs = [
        {
            "content": content,
            "filename": "book.pdf",
            "_retrieval_branch_ranks": branch_ranks,
        }
        for content, branch_ranks in (
            ("generic-1", {"1": 4, "2": 4}),
            ("generic-2", {"1": 5, "2": 5}),
            ("generic-3", {"1": 6, "2": 6}),
            ("generic-4", {"1": 7, "2": 7}),
            ("slice-mechanism", {"1": 1}),
            ("loop-mechanism", {"2": 1}),
            ("slice-fix", {"1": 2}),
            ("loop-fix-1", {"2": 2}),
            ("slice-support", {"1": 3}),
            ("loop-fix-2", {"2": 3}),
            ("noise-1", {"1": 8}),
            ("noise-2", {"2": 8}),
        )
    ]
    encoder = MagicMock()
    encoder.rerank.return_value = [
        1.2,
        1.1,
        1.0,
        0.9,
        0.8,
        0.7,
        -0.1,
        -0.2,
        -0.3,
        -0.4,
        -0.5,
        -0.6,
    ]
    monkeypatch.setattr(reranker, "_get_cross_encoder", lambda _model: encoder)
    _with_floor(monkeypatch, None)

    out = reranker.rerank("two-part question", docs, limit=8, branch_count=2)

    assert {doc["content"] for doc in out} == {
        "generic-1",
        "generic-2",
        "slice-mechanism",
        "slice-fix",
        "slice-support",
        "loop-mechanism",
        "loop-fix-1",
        "loop-fix-2",
    }
    encoder.rerank.assert_called_once()


def test_multipart_coverage_never_bypasses_rerank_floor(monkeypatch):
    docs = [
        {
            "content": "relevant",
            "filename": "book.pdf",
            "_retrieval_branch_ranks": {"1": 2, "2": 2},
        },
        {
            "content": "branch-one-below-floor",
            "filename": "book.pdf",
            "_retrieval_branch_ranks": {"1": 1},
        },
        {
            "content": "branch-two-below-floor",
            "filename": "book.pdf",
            "_retrieval_branch_ranks": {"2": 1},
        },
    ]
    _with_scores(monkeypatch, [1.0, -1.0, -2.0])
    _with_floor(monkeypatch, 0.0)

    out = reranker.rerank("two-part question", docs, limit=3, branch_count=2)

    assert [doc["content"] for doc in out] == ["relevant"]


def test_rerank_logs_every_score_and_final_selection(monkeypatch):
    docs = [
        {
            "content": content,
            "document_id": "doc",
            "chunk_id": f"doc:{index}",
            "filename": "book.pdf",
        }
        for index, content in enumerate(("a", "b", "c"))
    ]
    _with_scores(monkeypatch, [0.1, 0.9, 0.5])
    _with_floor(monkeypatch, None)
    log_info = MagicMock()
    monkeypatch.setattr(reranker.logger, "info", log_info)

    reranker.rerank("q", docs, limit=2)

    scored_logs = [
        call for call in log_info.call_args_list if call.args[0] == "rerank_candidate_scored"
    ]
    assert len(scored_logs) == 3
    assert scored_logs[0].kwargs["chunk_id"] == "doc:1"
    assert scored_logs[0].kwargs["rerank_rank"] == 1
    assert scored_logs[0].kwargs["candidate_rank"] == 2
    assert scored_logs[0].kwargs["rerank_score"] == 0.9
    assert scored_logs[0].kwargs["selected"] is True
    assert scored_logs[0].kwargs["selection_reasons"] == ["global_rerank"]
    assert scored_logs[-1].kwargs["selected"] is False


def test_rerank_empty_returns_empty_without_loading_model(monkeypatch):
    monkeypatch.setattr(
        reranker, "_get_cross_encoder", lambda model: pytest.fail("must not load the encoder")
    )

    assert reranker.rerank("q", [], limit=5) == []


def test_rerank_floor_excludes_below_threshold_docs(monkeypatch):
    docs = _docs("a", "b", "c")
    _with_scores(monkeypatch, [2.0, -1.0, 0.5])
    _with_floor(monkeypatch, 0.0)

    out = reranker.rerank("q", docs, limit=3)

    # "b" scored below the floor and is dropped; the rest survive in ranked order.
    assert [d["content"] for d in out] == ["a", "c"]


def test_rerank_floor_none_keeps_every_chunk(monkeypatch):
    docs = _docs("a", "b", "c")
    _with_scores(monkeypatch, [2.0, -1.0, 0.5])
    _with_floor(monkeypatch, None)

    out = reranker.rerank("q", docs, limit=3)

    assert [d["content"] for d in out] == ["a", "c", "b"]


def test_rerank_floor_excluding_all_returns_empty(monkeypatch):
    docs = _docs("a", "b", "c")
    _with_scores(monkeypatch, [-2.0, -1.0, -3.0])
    _with_floor(monkeypatch, 0.0)

    assert reranker.rerank("q", docs, limit=3) == []
