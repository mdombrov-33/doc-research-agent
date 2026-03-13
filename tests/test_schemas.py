import pytest
from pydantic import ValidationError

from src.api.schemas import QueryRequest


def test_defaults():
    req = QueryRequest(question="what is RAG?")
    assert req.top_k == 5
    assert req.model is None
    assert len(req.session_id) == 36  # UUID4


def test_session_id_unique():
    a = QueryRequest(question="q")
    b = QueryRequest(question="q")
    assert a.session_id != b.session_id


def test_explicit_session_id_preserved():
    req = QueryRequest(question="q", session_id="my-session")
    assert req.session_id == "my-session"


def test_top_k_bounds():
    with pytest.raises(ValidationError):
        QueryRequest(question="q", top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(question="q", top_k=21)


def test_top_k_valid_range():
    assert QueryRequest(question="q", top_k=1).top_k == 1
    assert QueryRequest(question="q", top_k=20).top_k == 20


def test_empty_question_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_model_openrouter_format():
    req = QueryRequest(question="q", model="anthropic/claude-sonnet-4.6")
    assert req.model == "anthropic/claude-sonnet-4.6"


def test_model_openai_format():
    req = QueryRequest(question="q", model="gpt-4o")
    assert req.model == "gpt-4o"
