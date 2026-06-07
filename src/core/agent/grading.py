from typing import Any, Literal, cast

from langchain_core.language_models import LanguageModelInput
from pydantic import BaseModel, Field

from src.config import get_settings
from src.core.agent.prompts import (
    DOCUMENT_GRADER_SYSTEM_PROMPT,
    DOCUMENT_GRADER_USER_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_PROMPT,
)
from src.core.llm import get_llm
from src.utils.logger import logger
from src.utils.retry import with_retry


class RouteAndRewrite(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(
        description="Route to 'vectorstore' or 'websearch' based on the question"
    )
    rewritten_query: str = Field(
        description="Optimized version of the question for semantic search. Preserve all parts of multi-part questions."  # noqa
    )


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Relevance score 'yes' or 'no'")


def _format_history(chat_history: list[dict[str, str]], max_turns: int = 2) -> str:
    """Render the last `max_turns` exchanges as a compact transcript for query rewriting.
    Bounded on purpose: older turns add tokens and pull the rewrite off-topic."""
    recent = chat_history[-max_turns * 2 :]
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent)


def route_and_rewrite(
    question: str, chat_history: list[dict[str, str]] | None = None
) -> RouteAndRewrite:
    llm = get_llm(get_settings().CLASSIFIER_MODEL)
    structured_llm = llm.with_structured_output(RouteAndRewrite)  # type: ignore[misc]

    system_content = (
        f"{ROUTER_SYSTEM_PROMPT}\n\n"
        "Also rewrite the question into an optimized search query "
        "(remove filler words, expand abbreviations, use precise terminology)."
    )
    user_content = ROUTER_USER_PROMPT.format(question=question)

    # History-aware rewrite: a follow-up like "expand on that" carries no search terms
    # on its own. Give the router the recent transcript so it can resolve the reference
    # into a standalone query before anything retrieves. Standalone questions pass through.
    if chat_history:
        system_content += (
            "\n\nThe question may be a follow-up that refers to earlier conversation "
            "(e.g. 'expand on that', 'the third one'). If so, rewrite it into a standalone "
            "query that names the referenced subject explicitly. If the question already "
            "stands on its own, leave its meaning unchanged."
        )
        user_content = f"Conversation so far:\n{_format_history(chat_history)}\n\n{user_content}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    result: RouteAndRewrite = with_retry(lambda: structured_llm.invoke(messages))  # type: ignore[assignment]
    logger.debug("route_and_rewrite", datasource=result.datasource, query=result.rewritten_query)
    return result


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
