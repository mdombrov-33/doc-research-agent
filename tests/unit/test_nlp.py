from types import SimpleNamespace

from src.core import nlp
from src.core.nlp import extract_entities


def test_extract_entities_dedupes(monkeypatch):
    # Two mentions of "Acme" must collapse to one entity.
    ents = [SimpleNamespace(text="Acme"), SimpleNamespace(text="Acme"), SimpleNamespace(text="Bob")]
    fake_nlp = lambda _text: SimpleNamespace(ents=ents)  # noqa: E731
    monkeypatch.setattr(nlp, "get_spacy_model", lambda: fake_nlp)
    result = extract_entities("Acme hired Acme employee Bob")
    assert sorted(result) == ["Acme", "Bob"]
