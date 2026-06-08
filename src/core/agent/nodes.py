from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from opentelemetry import trace

from src.core.agent.prompts import AGENT_SYSTEM_PROMPT
from src.core.agent.state import AgentState
from src.core.agent.tools import TOOLS
from src.core.llm import get_llm

_tracer = trace.get_tracer(__name__)

MAX_TOOL_CALLS = 3


def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable") or {}


def agent_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """The ReAct brain: decides which tool to call, or writes the final answer."""
    model = _configurable(config).get("model")
    llm = get_llm(model).bind_tools(TOOLS)
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *state["messages"]]
    with _tracer.start_as_current_span("agent.llm_call") as span:
        span.set_attribute("agent.model", model or "default")
        response = llm.invoke(messages)
        span.set_attribute("agent.has_tool_calls", bool(getattr(response, "tool_calls", None)))
    return {"messages": [response]}


def post_tools_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Increments the tool call counter and injects a stop message when the limit is reached."""
    count = (state.get("tool_call_count") or 0) + 1
    result: dict[str, Any] = {"tool_call_count": 1}
    if count >= MAX_TOOL_CALLS:
        result["messages"] = [
            HumanMessage(
                content="You have reached the maximum number of tool calls. Write your final answer now using the context you have gathered."  # noqa: E501
            )
        ]
    return result
