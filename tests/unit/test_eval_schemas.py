import json

import pytest
from pydantic import ValidationError

from evals.schemas import (
    Benchmark,
    DocumentJudgment,
    DocumentSpec,
    EvaluationCase,
    EvaluationTurn,
    ExpectedOutcome,
    Fact,
    FactLedger,
    LedgerPassage,
    NormalizedRunRecord,
    Pack,
    RunConfiguration,
    RunTurn,
    WebFixture,
    WebResult,
)
from evals.validate import load_benchmark


def _ledger() -> FactLedger:
    return FactLedger(
        packs=[Pack(id="policy", title="Policy pack")],
        facts=[Fact(id="leave-days", text="Employees receive 24 leave days.")],
        documents=[
            DocumentSpec(
                id="handbook",
                pack_id="policy",
                filename="handbook.txt",
                title="Employee handbook",
                passages=[
                    LedgerPassage(
                        id="annual-leave",
                        text="Full-time employees receive 24 leave days each year.",
                        fact_ids=["leave-days"],
                    )
                ],
            )
        ],
    )


def _document_case() -> EvaluationCase:
    return EvaluationCase(
        id="leave-policy",
        tags=["single-document"],
        turns=[
            EvaluationTurn(
                question="How many leave days do employees receive?",
                expected_outcome=ExpectedOutcome.DOCUMENT_GROUNDED,
                reference_answer="Full-time employees receive 24 leave days each year.",
                reference_passages=[{"document_id": "handbook", "passage_id": "annual-leave"}],
                document_relevance=[DocumentJudgment(document_id="handbook", relevance=2)],
                expected_tools=[
                    {
                        "name": "retrieve_documents",
                        "arguments": {"query": "employee leave days"},
                    }
                ],
            )
        ],
    )


def test_benchmark_accepts_cross_referenced_data():
    benchmark = Benchmark(ledger=_ledger(), cases=[_document_case()])

    assert benchmark.cases[0].turns[0].document_relevance[0].relevance == 2


def test_benchmark_rejects_unknown_reference_passage():
    case = _document_case()
    case.turns[0].reference_passages[0].passage_id = "missing"

    with pytest.raises(ValidationError, match="unknown passage"):
        Benchmark(ledger=_ledger(), cases=[case])


def test_web_outcome_requires_a_known_fixture():
    case = EvaluationCase(
        id="current-weather",
        tags=["web"],
        turns=[
            EvaluationTurn(
                question="What is the current weather?",
                expected_outcome=ExpectedOutcome.WEB_GROUNDED,
                reference_answer="The fixed fixture says it is sunny.",
                web_fixture_id="weather",
            )
        ],
    )

    with pytest.raises(ValidationError, match="unknown web fixture"):
        Benchmark(ledger=_ledger(), cases=[case])


def test_load_benchmark_validates_files_and_artifacts(tmp_path):
    root = tmp_path / "benchmark"
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "handbook.txt").write_text("Employees receive 24 leave days.")
    (root / "fact_ledger.json").write_text(_ledger().model_dump_json(indent=2))
    (root / "cases.jsonl").write_text(_document_case().model_dump_json() + "\n")
    (root / "web_fixtures.json").write_text("[]\n")

    benchmark = load_benchmark(root)

    assert benchmark.ledger.documents[0].id == "handbook"


def test_load_benchmark_reports_invalid_jsonl_line(tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    (root / "fact_ledger.json").write_text(_ledger().model_dump_json())
    (root / "cases.jsonl").write_text(json.dumps({"id": "incomplete"}) + "\n")
    (root / "web_fixtures.json").write_text("[]\n")

    with pytest.raises(ValueError, match=r"cases\.jsonl:1"):
        load_benchmark(root)


def test_normalized_run_requires_output_when_application_succeeds():
    with pytest.raises(ValidationError, match="actual_outcome and response"):
        NormalizedRunRecord(
            case_id="leave-policy",
            configuration=RunConfiguration(
                generator="provider/model",
                embedding_model="provider/embedding",
                chunking={"size": 800, "overlap": 100},
            ),
            turns=[
                RunTurn(
                    question="How many leave days do employees receive?",
                    actual_outcome=None,
                    response=None,
                )
            ],
        )


def test_abstention_fixture_can_be_validated():
    case = EvaluationCase(
        id="unknown-policy",
        tags=["abstention"],
        turns=[
            EvaluationTurn(
                question="What is the lunar office policy?",
                expected_outcome=ExpectedOutcome.ABSTENTION,
                web_fixture_id="no-results",
            )
        ],
    )
    fixture = WebFixture(
        id="no-results",
        query="lunar office policy",
        results=[
            WebResult(
                title="Unrelated result",
                url="https://example.com/unrelated",
                content="This page does not answer the question.",
            )
        ],
    )

    benchmark = Benchmark(ledger=_ledger(), cases=[case], web_fixtures=[fixture])

    assert benchmark.cases[0].turns[0].expected_outcome == ExpectedOutcome.ABSTENTION
