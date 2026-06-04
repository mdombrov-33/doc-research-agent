from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core import nodes
from src.core.grading.graders import RouteAndRewrite


class _Doc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


# --------------------------- router_node ---------------------------


def test_router_node_routes_to_websearch_and_rewrites(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "route_and_rewrite",
        lambda q: RouteAndRewrite(datasource="websearch", rewritten_query="rewritten"),
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
        lambda q: RouteAndRewrite(datasource="vectorstore", rewritten_query="q"),
    )
    assert nodes.router_node({"question": "q"})["web_search"] is False


# --------------------------- _entity_filter ---------------------------


def test_entity_filter_none_for_empty():
    assert nodes._entity_filter([]) is None


def test_entity_filter_builds_match_any():
    flt = nodes._entity_filter(["Acme", "Bob"])
    assert flt is not None
    assert flt.must[0].match.any == ["Acme", "Bob"]


# --------------------------- retrieve_node ---------------------------


def test_retrieve_node_maps_results_and_skips_blank(monkeypatch):
    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (_Doc("real content", {"filename": "a.pdf", "chunk_index": 2, "chunk_length": 12}), 0.9),
        (_Doc("   ", {"filename": "blank.pdf"}), 0.5),  # blank → skipped
    ]
    monkeypatch.setattr(nodes, "get_vector_store_tool", lambda: store)
    monkeypatch.setattr(nodes, "extract_entities", lambda q: [])

    out = nodes.retrieve_node({"question": "q", "top_k": 5})

    assert out["docs_retrieved_total"] == 1
    assert out["raw_documents"][0]["filename"] == "a.pdf"
    assert out["raw_documents"][0]["source"] == "vectorstore"


def test_retrieve_node_falls_back_to_unfiltered_when_filter_empty(monkeypatch):
    store = MagicMock()
    # First (filtered) call returns nothing, second (unfiltered) returns a hit.
    store.similarity_search_with_score.side_effect = [
        [],
        [(_Doc("fallback hit", {"filename": "b.txt"}), 0.8)],
    ]
    monkeypatch.setattr(nodes, "get_vector_store_tool", lambda: store)
    monkeypatch.setattr(nodes, "extract_entities", lambda q: ["Acme"])

    out = nodes.retrieve_node({"question": "q", "top_k": 5})

    assert store.similarity_search_with_score.call_count == 2
    assert out["docs_retrieved_total"] == 1
    # The fallback re-query drops the filter.
    assert store.similarity_search_with_score.call_args.kwargs.get("filter") is None


# --------------------------- web_search_node ---------------------------


def test_web_search_node_wraps_result(monkeypatch):
    tool = MagicMock()
    tool.invoke.return_value = "web answer"
    monkeypatch.setattr(nodes, "get_web_search_tool", lambda: tool)

    out = nodes.web_search_node({"question": "q"})

    assert out["web_search_done"] is True
    assert out["docs_retrieved_total"] == 1
    assert out["raw_documents"][0]["source"] == "web"


def test_web_search_node_swallows_errors(monkeypatch):
    tool = MagicMock()
    tool.invoke.side_effect = RuntimeError("ddg down")
    monkeypatch.setattr(nodes, "get_web_search_tool", lambda: tool)

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
