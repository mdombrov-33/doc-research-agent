from typing import Annotated, Required, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def _add_or_reset_int(left: int, right: int | None) -> int:
    if right is None:
        return 0
    return left + right


class AgentState(TypedDict, total=False):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    tool_call_count: Annotated[int, _add_or_reset_int]
