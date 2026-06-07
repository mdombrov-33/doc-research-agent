from typing import Any, Literal, cast

from langchain_core.language_models import LanguageModelInput
from pydantic import BaseModel, Field

from src.config import get_settings
from src.core.agent.prompts import (
    DOCUMENT_GRADER_SYSTEM_PROMPT,
    DOCUMENT_GRADER_USER_PROMPT,
)
from src.core.llm import get_llm
from src.utils.logger import logger
from src.utils.retry import with_retry


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Relevance score 'yes' or 'no'")


def grade_documents_batch(question: str, documents: list[str]) -> list[Literal["yes", "no"]]:
    if not documents:
        return []

    llm = get_llm(get_settings().CLASSIFIER_MODEL)
    structured_llm = llm.with_structured_output(GradeDocuments)

    batch_messages: list[LanguageModelInput] = []
    for document in documents:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": DOCUMENT_GRADER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": DOCUMENT_GRADER_USER_PROMPT.format(question=question, document=document),
            },
        ]
        batch_messages.append(messages)

    results = cast(list[GradeDocuments], with_retry(lambda: structured_llm.batch(batch_messages)))
    scores = cast(list[Literal["yes", "no"]], [result.binary_score for result in results])
    logger.debug("grading_batch", total=len(documents), relevant=scores.count("yes"))
    return scores
