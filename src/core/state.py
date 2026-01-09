from operator import add
from typing import Annotated, Required, TypedDict


class AgentState(TypedDict, total=False):
    question: Required[str]
    generation: str
    web_search: bool
    documents: Annotated[list[str], add]
    retrieval_attempts: int
    generation_attempts: int
