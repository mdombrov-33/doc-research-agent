from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.core.agent.grading import grade_documents_batch
from src.core.agent.prompts import AGENT_SYSTEM_PROMPT
from src.core.agent.state import AgentState
from src.core.agent.tools import TOOLS, format_docs
from src.core.llm import get_llm
from src.utils.logger import logger


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


def _last_question(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _recent_tool_messages(messages: list[AnyMessage]) -> list[ToolMessage]:
    """Tool messages produced by the most recent ToolNode run (since the last AI turn)."""
    collected: list[ToolMessage] = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            collected.append(m)
        elif isinstance(m, AIMessage):
            break
    collected.reverse()
    return collected


def grade_documents_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Score the freshly retrieved docs, drop irrelevant ones, and rewrite the tool message
    so the agent reasons only over what survived (an empty result nudges it to try another
    tool — this is the corrective step in CRAG)."""
    messages = state["messages"]
    tool_messages = _recent_tool_messages(messages)
    if not tool_messages:
        return {}

    question = _last_question(messages)
    updated: list[ToolMessage] = []
    relevant_all: list[dict] = []
    retrieved_total = 0
    web_used = False

    for tm in tool_messages:
        docs = tm.artifact or []
        retrieved_total += len(docs)
        if tm.name == "web_search":
            web_used = True
        if not docs:
            continue
        scores = grade_documents_batch(question, [d["content"] for d in docs])
        relevant = [d for d, score in zip(docs, scores) if score == "yes"]
        relevant_all.extend(relevant)
        content = format_docs(relevant) if relevant else "No relevant results found."
        # Same id => add_messages replaces the original ToolMessage in place.
        updated.append(
            ToolMessage(
                content=content,
                tool_call_id=tm.tool_call_id,
                name=tm.name,
                id=tm.id,
                artifact=relevant,
            )
        )

    logger.info("docs_graded", total=retrieved_total, relevant=len(relevant_all))

    result: dict[str, Any] = {
        "messages": updated,
        "documents": relevant_all,
        "docs_retrieved_total": retrieved_total,
    }
    # Only ever set True so an earlier retrieve cycle's False can't clobber it within a query.
    if web_used:
        result["web_search_used"] = True
    return result
