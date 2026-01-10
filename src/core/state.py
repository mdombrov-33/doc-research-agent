from typing import Required, TypedDict


class AgentState(TypedDict, total=False):
    question: Required[str]
    generation: str
    web_search: bool
    explicit_web_search: bool
    documents: list[str]
    retrieval_attempts: int
    generation_attempts: int
    hallucination_grounded: str
    answer_quality: str
    docs_retrieved_total: int
