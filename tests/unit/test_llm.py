from src.config import Settings
from src.core import llm as llm_mod


def test_get_llm_sets_max_retries(monkeypatch):
    monkeypatch.setattr(llm_mod, "get_settings", lambda: Settings(LLM_MAX_RETRIES=4))
    inst = llm_mod.get_llm("a/x")
    assert inst.max_retries == 4
