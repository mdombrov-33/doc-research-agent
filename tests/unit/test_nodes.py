from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.agent import nodes
from src.core.agent.grading import RouteAndRewrite

# --------------------------- router_node ---------------------------


def test_router_node_routes_to_websearch_and_rewrites(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "route_and_rewrite",
        lambda q, h=None: RouteAndRewrite(datasource="websearch", rewritten_query="rewritten"),
    )
    out = nodes.router_node({"question": "raw question"})
    assert out["web_search"] is True
    assert out["question"] == "rewritten"
    # Per-query state is reset so accumulators don't leak across turns.
    assert out["web_search_done"] is False
    assert out["web_fallback_needed"] is False
    assert out["raw_documents"] is None
    assert out["docs_retrieved_total"] is None


def test_router_node_routes_to_vectorstore(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "route_and_rewrite",
        lambda q, h=None: RouteAndRewrite(datasource="vectorstore", rewritten_query="q"),
    )
    assert nodes.router_node({"question": "q"})["web_search"] is False


# --------------------------- retrieve_node ---------------------------


def test_retrieve_node_delegates_to_hybrid_search(monkeypatch):
    # The node is a thin adapter: it calls hybrid_search and maps the result into state.
    docs = [{"content": "x", "filename": "a.pdf", "source": "vectorstore"}]
    monkeypatch.setattr(nodes, "hybrid_search", lambda question, top_k: docs)

    out = nodes.retrieve_node({"question": "q", "top_k": 5})

    assert out["raw_documents"] == docs
    assert out["docs_retrieved_total"] == 1


# --------------------------- web_search_node ---------------------------


def test_web_search_node_wraps_result(monkeypatch):
    tool = MagicMock()
    tool.invoke.return_value = "web answer"
    monkeypatch.setattr(nodes, "DuckDuckGoSearchRun", lambda: tool)

    out = nodes.web_search_node({"question": "q"})

    assert out["web_search_done"] is True
    assert out["docs_retrieved_total"] == 1
    assert out["raw_documents"][0]["source"] == "web"


def test_web_search_node_swallows_errors(monkeypatch):
    tool = MagicMock()
    tool.invoke.side_effect = RuntimeError("ddg down")
    monkeypatch.setattr(nodes, "DuckDuckGoSearchRun", lambda: tool)

    out = nodes.web_search_node({"question": "q"})

    assert out["raw_documents"] == []
    assert out["docs_retrieved_total"] == 0
    assert out["web_search_done"] is True


# --------------------------- grade_documents_node ---------------------------


def _docs(n):
    return [{"content": f"doc {i}", "filename": "f", "source": "vectorstore"} for i in range(n)]


def test_grade_node_no_documents_short_circuits(monkeypatch):
    monkeypatch.setattr(nodes, "grade_documents_batch", lambda *_: pytest.fail("should not grade"))
    assert nodes.grade_documents_node({"question": "q", "raw_documents": []}) == {"documents": []}


def test_grade_node_triggers_web_fallback_when_few_relevant(monkeypatch):
    monkeypatch.setattr(nodes, "grade_documents_batch", lambda q, c: ["yes", "no", "no"])
    out = nodes.grade_documents_node(
        {"question": "q", "raw_documents": _docs(3), "web_search_done": False}
    )
    assert out["web_fallback_needed"] is True
    assert out["raw_documents"] is None
    assert len(out["documents"]) == 1


def test_grade_node_no_fallback_when_enough_relevant(monkeypatch):
    monkeypatch.setattr(nodes, "grade_documents_batch", lambda q, c: ["yes", "yes", "no"])
    out = nodes.grade_documents_node(
        {"question": "q", "raw_documents": _docs(3), "web_search_done": False}
    )
    assert out["web_fallback_needed"] is False
    assert len(out["documents"]) == 2


def test_grade_node_merges_after_web_fallback(monkeypatch):
    monkeypatch.setattr(nodes, "grade_documents_batch", lambda q, c: ["yes"])
    existing = [{"content": "vec doc", "source": "vectorstore"}]
    out = nodes.grade_documents_node(
        {
            "question": "q",
            "raw_documents": [{"content": "web doc", "source": "web"}],
            "web_search_done": True,
            "documents": existing,
        }
    )
    assert out["web_fallback_needed"] is False
    assert len(out["documents"]) == 2  # existing vectorstore + new web


# --------------------------- generate_node ---------------------------


def test_generate_node_builds_context_and_updates_history(monkeypatch):
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="the answer")
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)

    out = nodes.generate_node(
        {
            "question": "what is X?",
            "documents": [
                {"content": "from a file", "filename": "a.pdf", "source": "vectorstore"},
                {"content": "from the web", "source": "web"},
            ],
            "chat_history": [],
        }
    )

    assert out["generation"] == "the answer"
    # Context passed to the system prompt must label both sources.
    system_msg = llm.invoke.call_args.args[0][0]["content"]
    assert "[Document: a.pdf]" in system_msg
    assert "[Web Search]" in system_msg
    # History grows by the user turn + assistant turn.
    assert out["chat_history"][-2:] == [
        {"role": "user", "content": "what is X?"},
        {"role": "assistant", "content": "the answer"},
    ]


def test_generate_node_retries_once_on_empty_generation(monkeypatch):
    llm = MagicMock()
    llm.invoke.side_effect = [SimpleNamespace(content="   "), SimpleNamespace(content="real")]
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)

    out = nodes.generate_node({"question": "q", "documents": [], "chat_history": []})

    assert out["generation"] == "real"
    assert llm.invoke.call_count == 2
