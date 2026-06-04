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

from src.constants import JUDGE_MODEL
from src.core.llm import get_llm
from src.prompts import ANSWER_RELEVANCE_PROMPT, FAITHFULNESS_PROMPT


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
