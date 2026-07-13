from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
