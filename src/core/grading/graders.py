from typing import Literal

from pydantic import BaseModel, Field

from src.config import get_settings
from src.core import prompts
from src.core.llm import get_llm
from src.utils.logger import logger

settings = get_settings()


class RouteAndRewrite(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(
        description="Route to 'vectorstore' or 'websearch' based on the question"
    )
    rewritten_query: str = Field(
        description="Optimized version of the question for semantic search. Preserve all parts of multi-part questions."
    )


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Relevance score 'yes' or 'no'")


def route_and_rewrite(question: str, model: str | None = None) -> RouteAndRewrite:
    llm = get_llm(model)
    structured_llm = llm.with_structured_output(RouteAndRewrite)  # type: ignore[misc]

    messages = [
        {
            "role": "system",
            "content": (
                f"{prompts.ROUTER_SYSTEM_PROMPT}\n\n"
                "Also rewrite the question into an optimized search query "
                "(remove filler words, expand abbreviations, use precise terminology)."
            ),
        },
        {"role": "user", "content": prompts.ROUTER_USER_PROMPT.format(question=question)},
    ]

    result: RouteAndRewrite = structured_llm.invoke(messages)  # type: ignore[assignment]
    logger.debug("route_and_rewrite", datasource=result.datasource, query=result.rewritten_query)
    return result


def grade_documents_batch(question: str, documents: list[str]) -> list[str]:
    if not documents:
        return []

    llm = get_llm("openai/gpt-5.4-mini")
    structured_llm = llm.with_structured_output(GradeDocuments)  # type: ignore[misc]

    batch_messages = []
    for document in documents:
        messages = [
            {"role": "system", "content": prompts.DOCUMENT_GRADER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.DOCUMENT_GRADER_USER_PROMPT.format(
                    question=question, document=document
                ),
            },
        ]
        batch_messages.append(messages)

    results = structured_llm.batch(batch_messages)
    scores = [result.binary_score for result in results]  # type: ignore[attr-defined]
    logger.debug("grading_batch", total=len(documents), relevant=scores.count("yes"))
    return scores
