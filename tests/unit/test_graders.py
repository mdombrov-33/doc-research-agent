import pytest

from src.core.grading import graders
from src.core.grading.graders import (
    GradeDocuments,
    RouteAndRewrite,
    _with_retry,
    grade_documents_batch,
    route_and_rewrite,
)
from tests.factories import make_structured_llm


def test_with_retry_recovers_after_one_failure():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("malformed output")
        return "ok"

    assert _with_retry(flaky) == "ok"
    assert calls["n"] == 2


def test_with_retry_reraises_after_exhausting_attempts():
    def always_fails():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        _with_retry(always_fails, attempts=2)


def test_route_and_rewrite_returns_structured_result(monkeypatch):
    expected = RouteAndRewrite(datasource="websearch", rewritten_query="latest python release")
    monkeypatch.setattr(graders, "get_llm", lambda *_a, **_k: make_structured_llm(expected))

    result = route_and_rewrite("what's the newest python?")

    assert result.datasource == "websearch"
    assert result.rewritten_query == "latest python release"


def test_grade_documents_batch_short_circuits_on_empty(monkeypatch):
    # No LLM call should happen for an empty document list.
    def _boom(*_a, **_k):
        raise AssertionError("get_llm must not be called for empty documents")

    monkeypatch.setattr(graders, "get_llm", _boom)
    assert grade_documents_batch("q", []) == []


def test_grade_documents_batch_maps_scores_in_order(monkeypatch):
    batch_return = [
        GradeDocuments(binary_score="yes"),
        GradeDocuments(binary_score="no"),
        GradeDocuments(binary_score="yes"),
    ]
    monkeypatch.setattr(
        graders, "get_llm", lambda *_a, **_k: make_structured_llm(batch_return=batch_return)
    )

    scores = grade_documents_batch("q", ["doc a", "doc b", "doc c"])

    assert scores == ["yes", "no", "yes"]
