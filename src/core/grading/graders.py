from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from src.config import get_settings
from src.core import prompts
from src.utils.logger import logger

settings = get_settings()


class RouteAndRewrite(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(
        description="Route to 'vectorstore' or 'websearch' based on the question"
    )
    rewritten_query: str = Field(
        description="Optimized version of the question for semantic search"
    )


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Relevance score 'yes' or 'no'")



def get_llm(model_override: str | None = None):
    model = model_override or settings.get_llm_model()
    logger.info(f"Using model: {model}")

    # Auto-detect provider: OpenRouter models contain '/', OpenAI models don't
    if model_override and "/" in model_override:
        llm = ChatOpenAI(
            api_key=SecretStr(settings.OPENROUTER_API_KEY),
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=0,
        )
    elif settings.LLM_PROVIDER == "openrouter":
        llm = ChatOpenAI(
            api_key=SecretStr(settings.OPENROUTER_API_KEY),
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=0,
        )
    else:
        llm = ChatOpenAI(
            api_key=SecretStr(settings.OPENAI_API_KEY),
            model=model,
            temperature=0,
        )

    return llm


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
    logger.info(f"Routed to: {result.datasource}, rewritten: '{result.rewritten_query}'")
    return result


def grade_documents_batch(question: str, documents: list[str], model: str | None = None) -> list[str]:
    if not documents:
        return []

    llm = get_llm(model)
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
    logger.info(f"Batch graded {len(documents)} documents: {scores.count('yes')} relevant")

    return scores
