import pytest

from src.core.agent import grading as graders
from src.core.agent.grading import (
    GradeDocuments,
    RouteAndRewrite,
    grade_documents_batch,
    route_and_rewrite,
)
from src.utils.retry import with_retry
from tests.factories import make_structured_llm


def test_with_retry_recovers_after_one_failure():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("malformed output")
        return "ok"

    assert with_retry(flaky) == "ok"
    assert calls["n"] == 2


def test_with_retry_reraises_after_exhausting_attempts():
    def always_fails():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        with_retry(always_fails, attempts=2)


def test_route_and_rewrite_returns_structured_result(monkeypatch):
    expected = RouteAndRewrite(datasource="websearch", rewritten_query="latest python release")
    monkeypatch.setattr(graders, "get_llm", lambda *_a, **_k: make_structured_llm(expected))

    result = route_and_rewrite("what's the newest python?")

    assert result.datasource == "websearch"
    assert result.rewritten_query == "latest python release"


def test_route_and_rewrite_injects_recent_history(monkeypatch):
    expected = RouteAndRewrite(datasource="vectorstore", rewritten_query="standalone")
    llm = make_structured_llm(expected)
    monkeypatch.setattr(graders, "get_llm", lambda *_a, **_k: llm)

    history = [
        {"role": "user", "content": "what are agentic patterns?"},
        {"role": "assistant", "content": "Prompt chaining, tool use, ..."},
    ]
    route_and_rewrite("can we expand that?", history)

    messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    system, user = messages[0]["content"], messages[1]["content"]
    assert "follow-up" in system  # standalone-rewrite instruction added
    assert "Conversation so far" in user
    assert "what are agentic patterns?" in user  # the referenced subject is available


def test_route_and_rewrite_omits_history_when_none(monkeypatch):
    expected = RouteAndRewrite(datasource="vectorstore", rewritten_query="q")
    llm = make_structured_llm(expected)
    monkeypatch.setattr(graders, "get_llm", lambda *_a, **_k: llm)

    route_and_rewrite("a standalone question")

    messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    assert "Conversation so far" not in messages[1]["content"]
    assert "follow-up" not in messages[0]["content"]


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
