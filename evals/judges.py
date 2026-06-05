"""
Level 2 - Generation

Was the answer grounded in context and on-topic?

Two independent questions about a generated answer:
  - faithfulness:     are the answer's claims supported by the retrieved context
                      (i.e. no hallucination)?
  - answer_relevance: does the answer actually address the question?

Each judge returns an integer 1-5 score; ``normalize`` maps it to 0-1 so thresholds
read naturally. The judge runs at temperature 0 with structured output for stability.
"""

from typing import Literal, cast

from pydantic import BaseModel, Field

from src.core.llm import get_llm

JUDGE_MODEL = "openai/gpt-5.4-mini"

FAITHFULNESS_PROMPT = """
You are grading whether an ANSWER is faithful to the provided CONTEXT.
A faithful answer asserts only what the context supports. Penalize any claim that is not
stated in, or cannot be directly inferred from, the context (a hallucination).

Score 1-5:
  5 = every claim is fully supported by the context
  3 = mostly supported, with minor unsupported details
  1 = largely unsupported or contradicts the context

CONTEXT:
{context}

ANSWER:
{answer}"""

ANSWER_RELEVANCE_PROMPT = """
You are grading whether an ANSWER addresses the QUESTION.
Judge only relevance, not factual accuracy: does it stay on topic and respond to what was
actually asked?

Score 1-5:
  5 = directly and completely answers the question
  3 = partially answers, or is padded with irrelevant content
  1 = off-topic or does not answer

QUESTION:
{question}

ANSWER:
{answer}"""


class Judgment(BaseModel):
    # Literal (an enum in the JSON schema) rather than a constrained int: some providers
    # reject minimum/maximum on integer types in structured-output schemas.
    score: Literal[1, 2, 3, 4, 5] = Field(description="1 = worst, 5 = best")
    reason: str = Field(description="one short sentence")


def normalize(score: int) -> float:
    """Map a 1-5 score onto 0-1."""
    return (score - 1) / 4


def _judge(prompt: str, model: str) -> Judgment:
    structured = get_llm(model, temperature=0).with_structured_output(Judgment)
    return cast(Judgment, structured.invoke([{"role": "user", "content": prompt}]))


def judge_faithfulness(context: str, answer: str, model: str = JUDGE_MODEL) -> Judgment:
    return _judge(FAITHFULNESS_PROMPT.format(context=context, answer=answer), model)


def judge_answer_relevance(question: str, answer: str, model: str = JUDGE_MODEL) -> Judgment:
    return _judge(ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer), model)
