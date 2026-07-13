import uuid
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.core.agent import nodes, tools
from src.core.agent.graph import build_graph


@pytest.fixture
def graph():
    return build_graph(checkpointer=MemorySaver())


def _patch_agent_llm(monkeypatch, script):
    """Script the agent's tool-calling LLM: each invoke returns the next message."""
    fake = MagicMock()
    fake.invoke.side_effect = script
    binder = MagicMock()
    binder.bind_tools.return_value = fake
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: binder)


def _run(graph, question, model=None, top_k=None):
    configurable = {"thread_id": str(uuid.uuid4())}
    if model is not None:
        configurable["model"] = model
    if top_k is not None:
        configurable["top_k"] = top_k
    config = {"configurable": configurable}
    inputs = {"messages": [HumanMessage(content=question)]}
    return graph.invoke(inputs, config=config)


def _retrieve_call(cid="c1"):
    return AIMessage(
        content="", tool_calls=[{"name": "retrieve_documents", "args": {"query": "q"}, "id": cid}]
    )


def _web_call(cid="c2"):
    return AIMessage(
        content="", tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": cid}]
    )


def test_graph_retrieves_then_answers(graph, monkeypatch):
    _patch_agent_llm(monkeypatch, [_retrieve_call(), AIMessage(content="final answer")])
    seen = {}
    monkeypatch.setattr(
        tools,
        "hybrid_search",
        lambda q, k: seen.update(top_k=k)
        or [
            {"content": "one", "filename": "a.pdf", "source": "document"},
            {"content": "two", "filename": "b.pdf", "source": "document"},
        ],
    )

    state = _run(graph, "q", top_k=11)

    assert seen["top_k"] == 11
    assert state["messages"][-1].content == "final answer"


def test_graph_falls_back_to_web_search(graph, monkeypatch):
    _patch_agent_llm(
        monkeypatch, [_retrieve_call(), _web_call(), AIMessage(content="final answer")]
    )
    monkeypatch.setattr(
        tools,
        "hybrid_search",
        lambda q, k: [{"content": "vec", "filename": "a.pdf", "source": "document"}],
    )
    web_tool = MagicMock()
    web_tool.invoke.return_value = [
        {"title": "Web result", "link": "https://example.com", "snippet": "web result"}
    ]
    monkeypatch.setattr(tools, "DuckDuckGoSearchResults", lambda **_: web_tool)

    state = _run(graph, "q")

    web_tool.invoke.assert_called_once()
    assert state["messages"][-1].content == "final answer"
