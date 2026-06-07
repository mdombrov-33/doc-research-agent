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
    web_search: bool  # router decision: run web search in parallel  # noqa: E501
    raw_documents: Annotated[
        list[dict], _add_or_reset_list
    ]  # parallel fan-in merge, reset each query  # noqa: E501
    documents: list[dict]  # graded/filtered docs passed to generator  # noqa: E501
    docs_retrieved_total: Annotated[int, _add_or_reset_int]  # accumulated retrieval count for eval
    chat_history: list[dict[str, str]]
    web_search_done: bool  # guard: prevents fallback loop
    web_fallback_needed: bool  # grader signal: not enough docs, trigger fallback  # noqa: E501
    model: str | None
    top_k: int
