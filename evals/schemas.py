from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

Identifier = Annotated[
    str,
    Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9_-]*$"),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpectedOutcome(StrEnum):
    DOCUMENT_GROUNDED = "document_grounded"
    WEB_GROUNDED = "web_grounded"
    ABSTENTION = "abstention"


class Pack(StrictModel):
    id: Identifier
    title: str = Field(min_length=1)
    description: str | None = None


class Fact(StrictModel):
    id: Identifier
    text: str = Field(min_length=1)


class LedgerPassage(StrictModel):
    id: Identifier
    text: str = Field(min_length=1)
    fact_ids: list[Identifier] = Field(min_length=1)


class DocumentSpec(StrictModel):
    id: Identifier
    pack_id: Identifier
    filename: str = Field(min_length=1)
    title: str = Field(min_length=1)
    passages: list[LedgerPassage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_filename_and_passages(self) -> "DocumentSpec":
        path = PurePosixPath(self.filename)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise ValueError("document filename must be a safe basename")
        if path.suffix.lower() not in {".pdf", ".docx", ".txt"}:
            raise ValueError("document filename must use PDF, DOCX, or TXT")
        _require_unique([passage.id for passage in self.passages], "passage IDs")
        return self


class FactLedger(StrictModel):
    version: int = Field(default=1, ge=1)
    packs: list[Pack] = Field(min_length=1)
    facts: list[Fact] = Field(min_length=1)
    documents: list[DocumentSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> "FactLedger":
        pack_ids = _require_unique([pack.id for pack in self.packs], "pack IDs")
        fact_ids = _require_unique([fact.id for fact in self.facts], "fact IDs")
        _require_unique([document.id for document in self.documents], "document IDs")
        _require_unique([document.filename for document in self.documents], "document filenames")

        for document in self.documents:
            if document.pack_id not in pack_ids:
                raise ValueError(
                    f"document {document.id!r} references unknown pack {document.pack_id!r}"
                )
            for passage in document.passages:
                unknown_facts = set(passage.fact_ids) - fact_ids
                if unknown_facts:
                    raise ValueError(
                        f"passage {document.id}/{passage.id} references unknown facts "
                        f"{sorted(unknown_facts)}"
                    )
        return self


class ReferencePassage(StrictModel):
    document_id: Identifier
    passage_id: Identifier


class DocumentJudgment(StrictModel):
    document_id: Identifier
    relevance: Literal[1, 2]


class ExpectedToolCall(StrictModel):
    name: Identifier
    arguments: dict[str, Any] = Field(default_factory=dict)


class EvaluationTurn(StrictModel):
    question: str = Field(min_length=1)
    expected_outcome: ExpectedOutcome
    reference_answer: str | None = None
    reference_passages: list[ReferencePassage] = Field(default_factory=list)
    document_relevance: list[DocumentJudgment] = Field(default_factory=list)
    expected_tools: list[ExpectedToolCall] = Field(default_factory=list)
    web_fixture_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_expected_result(self) -> "EvaluationTurn":
        if self.expected_outcome != ExpectedOutcome.ABSTENTION and not self.reference_answer:
            raise ValueError("answered outcomes require a reference answer")
        if (
            self.expected_outcome == ExpectedOutcome.ABSTENTION
            and self.reference_answer is not None
        ):
            raise ValueError("abstention cannot define a reference answer")
        if self.expected_outcome == ExpectedOutcome.DOCUMENT_GROUNDED:
            if self.web_fixture_id is not None:
                raise ValueError("document-grounded outcomes cannot use a web fixture")
            if not any(item.relevance == 2 for item in self.document_relevance):
                raise ValueError("document-grounded outcomes require relevance-2 evidence")
            if not self.reference_passages:
                raise ValueError("document-grounded outcomes require reference passages")
        elif self.web_fixture_id is None:
            raise ValueError("web-grounded and abstention outcomes require a web fixture")

        _require_unique(
            [judgment.document_id for judgment in self.document_relevance],
            "document relevance judgments",
        )
        return self


class EvaluationCase(StrictModel):
    id: Identifier
    tags: list[Identifier] = Field(min_length=1)
    turns: list[EvaluationTurn] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_tags(self) -> "EvaluationCase":
        _require_unique(self.tags, "case tags")
        return self


class WebResult(StrictModel):
    title: str = Field(min_length=1)
    url: HttpUrl
    content: str = Field(min_length=1)


class WebFixture(StrictModel):
    id: Identifier
    query: str = Field(min_length=1)
    results: list[WebResult]


class Benchmark(StrictModel):
    ledger: FactLedger
    cases: list[EvaluationCase] = Field(min_length=1)
    web_fixtures: list[WebFixture] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "Benchmark":
        _require_unique([case.id for case in self.cases], "case IDs")
        fixture_ids = _require_unique(
            [fixture.id for fixture in self.web_fixtures], "web fixture IDs"
        )
        documents = {document.id: document for document in self.ledger.documents}

        for case in self.cases:
            for turn in case.turns:
                if turn.web_fixture_id is not None and turn.web_fixture_id not in fixture_ids:
                    raise ValueError(
                        f"case {case.id!r} references unknown web fixture {turn.web_fixture_id!r}"
                    )
                judged_documents = {judgment.document_id for judgment in turn.document_relevance}
                unknown_documents = judged_documents - documents.keys()
                if unknown_documents:
                    raise ValueError(
                        f"case {case.id!r} references unknown documents {sorted(unknown_documents)}"
                    )
                for reference in turn.reference_passages:
                    document = documents.get(reference.document_id)
                    if document is None:
                        raise ValueError(
                            f"case {case.id!r} references unknown document "
                            f"{reference.document_id!r}"
                        )
                    passage_ids = {passage.id for passage in document.passages}
                    if reference.passage_id not in passage_ids:
                        raise ValueError(
                            f"case {case.id!r} references unknown passage "
                            f"{reference.document_id}/{reference.passage_id}"
                        )
                    if reference.document_id not in judged_documents:
                        raise ValueError(
                            f"case {case.id!r} has a reference passage without a document "
                            f"relevance judgment for {reference.document_id!r}"
                        )
        return self


class RetrievedContext(StrictModel):
    document_id: Identifier
    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    rank: int = Field(ge=1)
    score: float | None = None


class ToolCallRecord(StrictModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class CitationRecord(StrictModel):
    source_id: str = Field(min_length=1)
    source_type: Literal["document", "web"]
    excerpt: str = Field(min_length=1)
    document_id: str | None = None
    chunk_id: str | None = None
    url: HttpUrl | None = None


class TokenUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class RunTurn(StrictModel):
    question: str = Field(min_length=1)
    retrieved_contexts: list[RetrievedContext] = Field(default_factory=list)
    response: str | None
    actual_outcome: ExpectedOutcome | None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    citations: list[CitationRecord] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0)
    token_usage: TokenUsage | None = None
    application_error: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "RunTurn":
        if self.application_error is None and (
            self.actual_outcome is None or self.response is None
        ):
            raise ValueError("successful run turns require actual_outcome and response")
        return self


class RunConfiguration(StrictModel):
    generator: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    reranker: str | None = None
    chunking: dict[str, Any]
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    framework_versions: dict[str, str] = Field(default_factory=dict)


class NormalizedRunRecord(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    case_id: Identifier
    configuration: RunConfiguration
    turns: list[RunTurn] = Field(min_length=1, max_length=2)


def _require_unique(values: list[str], label: str) -> set[str]:
    unique = set(values)
    if len(unique) != len(values):
        raise ValueError(f"{label} must be unique")
    return unique
