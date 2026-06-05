from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core import nodes
from src.core.agent import build_graph
from src.core.grading.graders import RouteAndRewrite


class _Doc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


@pytest.fixture
def graph():
    return build_graph()


def _run(graph, inputs):
    config = {"configurable": {"thread_id": "test-thread"}}
    return graph.invoke(inputs, config=config)


def test_web_fallback_loop_merges_vectorstore_and_web(graph, monkeypatch):
    # Router picks the vector store; retrieval yields a single relevant doc.
    monkeypatch.setattr(
        nodes,
        "route_and_rewrite",
        lambda q, h=None: RouteAndRewrite(datasource="vectorstore", rewritten_query="q"),
    )
    monkeypatch.setattr(nodes, "extract_entities", lambda q: [])

    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (_Doc("vector doc", {"filename": "a.pdf"}), 0.9)
    ]
    monkeypatch.setattr(nodes, "get_vector_store_tool", lambda: store)

    web_tool = MagicMock()
    web_tool.invoke.return_value = "web result"
    monkeypatch.setattr(nodes, "get_web_search_tool", lambda: web_tool)

    # Everything graded relevant; the single vector doc (<2) forces a web fallback.
    monkeypatch.setattr(nodes, "grade_documents_batch", lambda q, c: ["yes"] * len(c))

    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="final answer")
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)

    state = _run(graph, {"question": "raw question", "top_k": 5})

    # Fallback path was taken: web search ran and its doc was merged in.
    web_tool.invoke.assert_called_once()
    sources = {d["source"] for d in state["documents"]}
    assert sources == {"vectorstore", "web"}
    assert state["generation"] == "final answer"


def test_no_fallback_when_enough_relevant_docs(graph, monkeypatch):
    monkeypatch.setattr(
        nodes,
        "route_and_rewrite",
        lambda q, h=None: RouteAndRewrite(datasource="vectorstore", rewritten_query="q"),
    )
    monkeypatch.setattr(nodes, "extract_entities", lambda q: [])

    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (_Doc("doc one", {"filename": "a.pdf"}), 0.9),
        (_Doc("doc two", {"filename": "b.pdf"}), 0.8),
    ]
    monkeypatch.setattr(nodes, "get_vector_store_tool", lambda: store)

    web_tool = MagicMock()
    monkeypatch.setattr(nodes, "get_web_search_tool", lambda: web_tool)
    monkeypatch.setattr(nodes, "grade_documents_batch", lambda q, c: ["yes"] * len(c))

    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="answer")
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)

    state = _run(graph, {"question": "q", "top_k": 5})

    # Two relevant docs is enough — no web search.
    web_tool.invoke.assert_not_called()
    assert {d["source"] for d in state["documents"]} == {"vectorstore"}
