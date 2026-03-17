import pytest

from src.core.evaluation.metrics import EvaluationTracker, QueryEvaluation


def _make_eval(
    docs_retrieved: int = 5,
    docs_relevant: int = 4,
    web_search_triggered: bool = False,
    latency_ms: float = 1000.0,
) -> QueryEvaluation:
    return QueryEvaluation(
        question="test question",
        retrieval_precision=docs_relevant / docs_retrieved if docs_retrieved else 0.0,
        docs_retrieved=docs_retrieved,
        docs_relevant=docs_relevant,
        web_search_triggered=web_search_triggered,
        latency_ms=latency_ms,
    )


def test_empty_tracker():
    tracker = EvaluationTracker()
    stats = tracker.get_stats()
    assert stats["total_queries"] == 0
    assert stats["avg_retrieval_precision"] == 0.0
    assert stats["web_search_rate"] == 0.0


def test_single_query_stats():
    tracker = EvaluationTracker()
    tracker.record(_make_eval(docs_retrieved=5, docs_relevant=4, latency_ms=2000.0))
    stats = tracker.get_stats()
    assert stats["total_queries"] == 1
    assert stats["avg_docs_retrieved"] == 5.0
    assert stats["avg_docs_relevant"] == 4.0
    assert stats["avg_retrieval_precision"] == pytest.approx(4 / 5)
    assert stats["avg_latency_ms"] == 2000.0


def test_retrieval_precision_across_multiple_queries():
    tracker = EvaluationTracker()
    tracker.record(_make_eval(docs_retrieved=4, docs_relevant=4))  # precision 1.0
    tracker.record(_make_eval(docs_retrieved=4, docs_relevant=2))  # precision 0.5
    stats = tracker.get_stats()
    # total relevant=6, total retrieved=8 → precision = 0.75
    assert stats["avg_retrieval_precision"] == pytest.approx(6 / 8)


def test_web_search_rate():
    tracker = EvaluationTracker()
    tracker.record(_make_eval(web_search_triggered=True))
    tracker.record(_make_eval(web_search_triggered=True))
    tracker.record(_make_eval(web_search_triggered=False))
    stats = tracker.get_stats()
    assert stats["web_search_rate"] == pytest.approx(2 / 3)


def test_avg_latency():
    tracker = EvaluationTracker()
    tracker.record(_make_eval(latency_ms=1000.0))
    tracker.record(_make_eval(latency_ms=3000.0))
    stats = tracker.get_stats()
    assert stats["avg_latency_ms"] == pytest.approx(2000.0)


def test_zero_docs_retrieved_precision():
    tracker = EvaluationTracker()
    tracker.record(_make_eval(docs_retrieved=0, docs_relevant=0))
    stats = tracker.get_stats()
    assert stats["avg_retrieval_precision"] == 0.0
