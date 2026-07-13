from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.agent import nodes
from src.core.agent.tools import retrieve_documents


def test_agent_node_binds_tools_and_prepends_system_prompt(monkeypatch):
    bound = MagicMock()
    bound.invoke.return_value = AIMessage(content="hi")
    llm = MagicMock()
    llm.bind_tools.return_value = bound
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)

    out = nodes.agent_node({"messages": [HumanMessage(content="q")]})

    llm.bind_tools.assert_called_once_with([retrieve_documents])
    sent = bound.invoke.call_args.args[0]
    assert isinstance(sent[0], SystemMessage)
    assert isinstance(sent[1], HumanMessage)
    assert out["messages"][0].content == "hi"


def test_agent_node_bounds_history_and_excludes_tool_payloads(monkeypatch):
    bound = MagicMock()
    bound.invoke.return_value = AIMessage(content="hi")
    llm = MagicMock()
    llm.bind_tools.return_value = bound
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(
        nodes,
        "get_settings",
        lambda: MagicMock(CONVERSATION_HISTORY_TURNS=2),
    )

    nodes.agent_node(
        {
            "messages": [
                HumanMessage(content="first question"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "retrieve_documents",
                            "args": {"query": "first"},
                            "id": "first-retrieval",
                        }
                    ],
                ),
                ToolMessage(
                    content="OLD_TOOL_PAYLOAD",
                    tool_call_id="first-retrieval",
                    name="retrieve_documents",
                ),
                AIMessage(content="first answer"),
                HumanMessage(content="second question"),
                AIMessage(content="stale query-model reply"),
                AIMessage(content="second answer"),
                HumanMessage(content="current question"),
            ]
        }
    )

    sent = bound.invoke.call_args.args[0]
    assert isinstance(sent[0], SystemMessage)
    assert [message.content for message in sent[1:]] == [
        "second question",
        "second answer",
        "current question",
    ]
    assert not any(isinstance(message, ToolMessage) for message in sent)
