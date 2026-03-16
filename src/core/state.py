import operator
from typing import Annotated, Required, TypedDict


class AgentState(TypedDict, total=False):
    question: Required[str]
    generation: str
    web_search: bool
    raw_documents: Annotated[list[str], operator.add]
    documents: list[str]
    generation_attempts: int
    docs_retrieved_total: Annotated[int, operator.add]
    chat_history: list[dict[str, str]]
    model: str | None
    top_k: int
