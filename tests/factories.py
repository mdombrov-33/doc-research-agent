from types import SimpleNamespace
from unittest.mock import MagicMock


def make_moderation(flagged: bool) -> SimpleNamespace:
    """Shape returned by AsyncOpenAI().moderations.create()."""
    return SimpleNamespace(results=[SimpleNamespace(flagged=flagged)])


def make_llm_message(content: str) -> SimpleNamespace:
    """Shape returned by a chat model's .ainvoke() — only .content is read."""
    return SimpleNamespace(content=content)


def make_structured_llm(return_value=None, batch_return=None) -> MagicMock:
    """A get_llm() stand-in whose .with_structured_output(...) yields a stub
    exposing .invoke()/.batch() with the given return values."""
    structured = MagicMock()
    structured.invoke.return_value = return_value
    structured.batch.return_value = batch_return
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm
