from typing import Annotated, Required, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def _add_or_reset_list(left: list, right: list | None) -> list:
    if right is None:
        return []
    return left + right


def _add_or_reset_int(left: int, right: int | None) -> int:
    if right is None:
        return 0
    return left + right


class AgentState(TypedDict, total=False):
    # The ReAct loop: agent decides a tool, ToolNode runs it, grade filters the result, repeat.
    # Persisted per thread_id by the SqliteSaver checkpointer, so this list is the
    # conversation history across turns — no separate chat_history field needed.
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    # Graded-relevant docs for the current query. The reducers accumulate across tool calls
    # within a query and reset (on a None update from the request) at the start of the next one.
    documents: Annotated[list[dict], _add_or_reset_list]
    docs_retrieved_total: Annotated[int, _add_or_reset_int]  # pre-grade count, for metrics
    web_search_used: bool  # whether the web_search tool ran this query, for metrics
