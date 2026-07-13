from typing import Literal, TypeAlias

FinalOutcome: TypeAlias = Literal["document_answer", "web_answer", "abstained"]
FinalStopReason: TypeAlias = Literal[
    "document_evidence_sufficient",
    "web_evidence_sufficient",
    "insufficient_evidence_after_web",
    "retrieval_not_requested",
    "unknown",
]


def normalize_outcome(value: object) -> FinalOutcome:
    """Return a safe final outcome from a graph-state value."""
    if value == "document_answer":
        return "document_answer"
    if value == "web_answer":
        return "web_answer"
    if value == "abstained":
        return "abstained"
    return "abstained"


def normalize_stop_reason(value: object) -> FinalStopReason:
    """Return a safe terminal reason from a graph-state value."""
    if value == "document_evidence_sufficient":
        return "document_evidence_sufficient"
    if value == "web_evidence_sufficient":
        return "web_evidence_sufficient"
    if value == "insufficient_evidence_after_web":
        return "insufficient_evidence_after_web"
    if value == "retrieval_not_requested":
        return "retrieval_not_requested"
    return "unknown"
