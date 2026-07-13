from src.config import Settings
from src.core import llm as llm_mod


def test_get_llm_sets_retry_and_timeout_policy(monkeypatch):
    monkeypatch.setattr(
        llm_mod,
        "get_settings",
        lambda: Settings(LLM_MAX_RETRIES=4, LLM_TIMEOUT_SECONDS=42),
    )
    inst = llm_mod.get_llm("a/x")
    assert inst.max_retries == 4
    assert inst.request_timeout == 42
