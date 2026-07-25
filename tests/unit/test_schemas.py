import pytest
from pydantic import ValidationError

from src.api.schemas import QueryRequest


def test_defaults():
    req = QueryRequest(question="what is RAG?")
    assert req.model is None
    assert len(req.session_id) == 36  # UUID4


def test_session_id_unique():
    a = QueryRequest(question="q")
    b = QueryRequest(question="q")
    assert a.session_id != b.session_id


def test_explicit_session_id_preserved():
    req = QueryRequest(question="q", session_id="my-session")
    assert req.session_id == "my-session"


def test_removed_retrieval_width_knob_is_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QueryRequest(question="q", top_k=5)


def test_empty_question_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_model_openrouter_format():
    req = QueryRequest(question="q", model="anthropic/claude-sonnet-4.6")
    assert req.model == "anthropic/claude-sonnet-4.6"


def test_model_openai_format():
    req = QueryRequest(question="q", model="openai/gpt-5.6-luna")
    assert req.model == "openai/gpt-5.6-luna"


def test_unsupported_model_rejected():
    with pytest.raises(ValidationError, match="Unsupported model"):
        QueryRequest(question="q", model="openai/gpt-4o")
