from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from src.config import get_settings
from src.core import prompts
from src.utils.logger import logger

settings = get_settings()


class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(
        description="Route to 'vectorstore' or 'websearch' based on the question"
    )


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Relevance score 'yes' or 'no'")


class GradeHallucinations(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Answer resolves question, 'yes' or 'no'"
    )


def get_llm():
    api_key = settings.get_llm_api_key()
    model = settings.get_llm_model()

    if settings.LLM_PROVIDER == "openrouter":
        llm = ChatOpenAI(
            api_key=SecretStr(api_key),
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=0,
        )
    else:
        llm = ChatOpenAI(
            api_key=SecretStr(api_key),
            model=model,
            temperature=0,
        )

    return llm


def route_question(question: str) -> str:
    llm = get_llm()
    structured_llm = llm.with_structured_output(RouteQuery)  # type: ignore[misc]

    messages = [
        {"role": "system", "content": prompts.ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.ROUTER_USER_PROMPT.format(question=question)},
    ]

    result: RouteQuery = structured_llm.invoke(messages)  # type: ignore[assignment]

    logger.info(f"Routed question to: {result.datasource}")
    return result.datasource


def grade_document_relevance(question: str, document: str) -> str:
    llm = get_llm()
    structured_llm = llm.with_structured_output(GradeDocuments)  # type: ignore[misc]

    messages = [
        {"role": "system", "content": prompts.DOCUMENT_GRADER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": prompts.DOCUMENT_GRADER_USER_PROMPT.format(
                question=question, document=document
            ),
        },
    ]

    result: GradeDocuments = structured_llm.invoke(messages)  # type: ignore[assignment]

    return result.binary_score


def check_hallucination(documents: list[str], generation: str) -> str:
    llm = get_llm()
    structured_llm = llm.with_structured_output(GradeHallucinations)  # type: ignore[misc]

    docs_text = "\n\n".join(documents)

    messages = [
        {"role": "system", "content": prompts.HALLUCINATION_GRADER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": prompts.HALLUCINATION_GRADER_USER_PROMPT.format(
                documents=docs_text, generation=generation
            ),
        },
    ]

    result: GradeHallucinations = structured_llm.invoke(messages)  # type: ignore[assignment]

    logger.info(f"Hallucination check: {result.binary_score}")
    return result.binary_score


def grade_answer_quality(question: str, generation: str) -> str:
    llm = get_llm()
    structured_llm = llm.with_structured_output(GradeAnswer)  # type: ignore[misc]

    messages = [
        {"role": "system", "content": prompts.ANSWER_GRADER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": prompts.ANSWER_GRADER_USER_PROMPT.format(
                question=question, generation=generation
            ),
        },
    ]

    result: GradeAnswer = structured_llm.invoke(messages)  # type: ignore[assignment]

    logger.info(f"Answer quality: {result.binary_score}")
    return result.binary_score


def rewrite_query(question: str) -> str:
    llm = get_llm()

    messages = [
        {"role": "system", "content": prompts.QUERY_REWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.QUERY_REWRITER_USER_PROMPT.format(question=question)},
    ]

    result = llm.invoke(messages)

    rewritten = result.content if isinstance(result.content, str) else str(result.content)

    logger.info(f"Rewritten query: {question} -> {rewritten}")
    return rewritten
