from unittest.mock import MagicMock


def make_structured_llm(return_value=None, batch_return=None) -> MagicMock:
    """A get_llm() stand-in whose .with_structured_output(...) yields a stub
    exposing .invoke()/.batch() with the given return values."""
    structured = MagicMock()
    structured.invoke.return_value = return_value
    structured.batch.return_value = batch_return
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm
