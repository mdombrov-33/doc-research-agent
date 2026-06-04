from typing import Any, Literal, cast

from langchain_core.language_models import LanguageModelInput
from pydantic import BaseModel, Field

from src.config import get_settings
from src.core import prompts
from src.core.constants import CLASSIFIER_MODEL
from src.core.llm import get_llm
from src.utils.logger import logger

settings = get_settings()


class RouteAndRewrite(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(
        description="Route to 'vectorstore' or 'websearch' based on the question"
    )
    rewritten_query: str = Field(
        description="Optimized version of the question for semantic search. Preserve all parts of multi-part questions."  # noqa
    )


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Relevance score 'yes' or 'no'")


def route_and_rewrite(question: str) -> RouteAndRewrite:
    llm = get_llm(CLASSIFIER_MODEL)
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


def grade_documents_batch(question: str, documents: list[str]) -> list[Literal["yes", "no"]]:
    if not documents:
        return []

    llm = get_llm(CLASSIFIER_MODEL)
    structured_llm = llm.with_structured_output(GradeDocuments)  # type: ignore[misc]

    batch_messages: list[LanguageModelInput] = []
    for document in documents:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.DOCUMENT_GRADER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.DOCUMENT_GRADER_USER_PROMPT.format(
                    question=question, document=document
                ),
            },
        ]
        batch_messages.append(messages)

    results = cast(list[GradeDocuments], structured_llm.batch(batch_messages))
    scores = cast(list[Literal["yes", "no"]], [result.binary_score for result in results])
    logger.debug("grading_batch", total=len(documents), relevant=scores.count("yes"))
    return scores
