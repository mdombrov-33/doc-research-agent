import pytest

from src.evals.embeddings_check import check_separation, cosine


def test_cosine_identical_orthogonal_and_zero():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # zero-vector guard


class _FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[text]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


def test_check_separation_passes_when_relevant_is_closer():
    embedder = _FakeEmbedder({"q": [1.0, 0.0], "rel": [1.0, 0.0], "irr": [0.0, 1.0]})
    assert check_separation([("q", "rel", "irr")], embedder) == (1, 1)


def test_check_separation_fails_when_irrelevant_is_closer():
    embedder = _FakeEmbedder({"q": [1.0, 0.0], "rel": [0.0, 1.0], "irr": [1.0, 0.0]})
    assert check_separation([("q", "rel", "irr")], embedder) == (0, 1)
