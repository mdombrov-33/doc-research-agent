from typing import Literal, TypeAlias

FinalOutcome: TypeAlias = Literal["document_answer", "web_answer", "abstained"]


def normalize_outcome(value: object) -> FinalOutcome:
    """Return a safe final outcome from a graph-state value."""
    if value == "document_answer" or value == "web_answer" or value == "abstained":
        return value
    return "abstained"
