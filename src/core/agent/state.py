from typing import Annotated, Required, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.core.agent.outcomes import FinalOutcome


class AgentState(TypedDict, total=False):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    evidence_sufficient: bool
    supporting_source_ids: list[str]
    outcome: FinalOutcome
