from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from src.core.agent.prompts import AGENT_SYSTEM_PROMPT
from src.core.agent.state import AgentState
from src.core.agent.tools import TOOLS
from src.core.llm import get_llm


def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable") or {}


def agent_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """The ReAct brain: decides which tool to call, or writes the final answer.

    bind_tools lets the model emit tool calls; when it stops calling tools, its content is
    the answer and tools_condition routes the graph to END. The model is per-request
    (from config), defaulting to LLM_MODEL when unset."""
    llm = get_llm(_configurable(config).get("model")).bind_tools(TOOLS)
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}
