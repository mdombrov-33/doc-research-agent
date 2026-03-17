from typing import Annotated, Required, TypedDict


def _add_or_reset_list(left: list, right: list | None) -> list:
    if right is None:
        return []
    return left + right


def _add_or_reset_int(left: int, right: int | None) -> int:
    if right is None:
        return 0
    return left + right


class AgentState(TypedDict, total=False):
    question: Required[str]
    generation: str
    web_search: bool
    raw_documents: Annotated[list[dict], _add_or_reset_list]
    documents: list[dict]
    generation_attempts: int
    docs_retrieved_total: Annotated[int, _add_or_reset_int]
    chat_history: list[dict[str, str]]
    model: str | None
    top_k: int
