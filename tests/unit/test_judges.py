from evals import judges
from evals.judges import Judgment
from tests.factories import make_structured_llm


def test_normalize_maps_1_to_5_onto_0_to_1():
    assert judges.normalize(1) == 0.0
    assert judges.normalize(3) == 0.5
    assert judges.normalize(5) == 1.0


def test_judge_faithfulness_returns_judgment(monkeypatch):
    monkeypatch.setattr(
        judges, "get_llm", lambda *a, **k: make_structured_llm(Judgment(score=4, reason="ok"))
    )
    result = judges.judge_faithfulness("the context", "the answer")
    assert result.score == 4


def test_judge_answer_relevance_returns_judgment(monkeypatch):
    monkeypatch.setattr(
        judges, "get_llm", lambda *a, **k: make_structured_llm(Judgment(score=5, reason="on topic"))
    )
    result = judges.judge_answer_relevance("the question", "the answer")
    assert result.score == 5
