from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.agent import nodes

# --------------------------- agent_node ---------------------------


def test_agent_node_binds_tools_and_prepends_system_prompt(monkeypatch):
    bound = MagicMock()
    bound.invoke.return_value = AIMessage(content="hi")
    llm = MagicMock()
    llm.bind_tools.return_value = bound
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)

    out = nodes.agent_node({"messages": [HumanMessage(content="q")]})

    llm.bind_tools.assert_called_once()
    sent = bound.invoke.call_args.args[0]
    assert isinstance(sent[0], SystemMessage)  # system prompt leads
    assert isinstance(sent[1], HumanMessage)
    assert out["messages"][0].content == "hi"


# --------------------------- grade_documents_node ---------------------------


def _docs(*sources):
    return [
        {"content": f"doc {i}", "filename": f"f{i}", "source": s} for i, s in enumerate(sources)
    ]


def _tool_msg(docs, name="retrieve_documents", tcid="1", mid="tm1"):
    return ToolMessage(content="raw", tool_call_id=tcid, name=name, id=mid, artifact=docs)


def test_grade_node_returns_empty_without_tool_messages():
    msgs = [HumanMessage(content="q"), AIMessage(content="answer")]
    assert nodes.grade_documents_node({"messages": msgs}) == {}


def test_grade_node_filters_to_relevant_and_rewrites_message(monkeypatch):
    monkeypatch.setattr(nodes, "grade_documents_batch", lambda q, c: ["yes", "no"])
    docs = _docs("vectorstore", "vectorstore")
    msgs = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "retrieve_documents", "args": {}, "id": "1"}]),
        _tool_msg(docs),
    ]

    out = nodes.grade_documents_node({"messages": msgs})

    assert len(out["documents"]) == 1
    assert out["documents"][0]["filename"] == "f0"
    assert out["docs_retrieved_total"] == 2
    # The tool message is replaced in place (same id) with only the surviving doc.
    assert out["messages"][0].id == "tm1"
    assert out["messages"][0].artifact == out["documents"]


def test_grade_node_signals_no_relevant_results(monkeypatch):
    monkeypatch.setattr(nodes, "grade_documents_batch", lambda q, c: ["no"])
    msgs = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "retrieve_documents", "args": {}, "id": "1"}]),
        _tool_msg(_docs("vectorstore")),
    ]

    out = nodes.grade_documents_node({"messages": msgs})

    assert out["documents"] == []
    assert "No relevant results" in out["messages"][0].content
    assert "web_search_used" not in out  # retrieval-only cycle


def test_grade_node_marks_web_search_used(monkeypatch):
    monkeypatch.setattr(nodes, "grade_documents_batch", lambda q, c: ["yes"])
    msgs = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "1"}]),
        _tool_msg(_docs("web"), name="web_search"),
    ]

    out = nodes.grade_documents_node({"messages": msgs})

    assert out["web_search_used"] is True
    assert out["documents"][0]["source"] == "web"
