import argparse
import json
from pathlib import Path
from typing import Any, cast

import pymupdf
from docx import Document as DocxDocument
from pydantic import ValidationError

from evals.schemas import Benchmark, EvaluationCase, FactLedger, WebFixture

DEFAULT_BENCHMARK_ROOT = Path(__file__).parent / "benchmark"


def load_benchmark(root: Path) -> Benchmark:
    ledger_path = root / "fact_ledger.json"
    cases_path = root / "cases.jsonl"
    fixtures_path = root / "web_fixtures.json"

    ledger = _validate_json_file(ledger_path, FactLedger)
    cases = _validate_json_lines(cases_path, EvaluationCase)
    fixtures = _validate_json_list(fixtures_path, WebFixture)
    benchmark = Benchmark(ledger=ledger, cases=cases, web_fixtures=fixtures)

    missing_artifacts = [
        document.filename
        for document in ledger.documents
        if not (root / "corpus" / document.filename).is_file()
    ]
    if missing_artifacts:
        raise ValueError(f"missing corpus artifacts: {sorted(missing_artifacts)}")
    _validate_artifact_contents(benchmark, root / "corpus")
    return benchmark


def _validate_json_file(path: Path, model: type[FactLedger]) -> FactLedger:
    try:
        return model.model_validate_json(path.read_text())
    except (OSError, ValidationError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error


def _validate_json_lines(path: Path, model: type[EvaluationCase]) -> list[EvaluationCase]:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"{path}: {error}") from error

    values: list[EvaluationCase] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            values.append(model.model_validate_json(line))
        except (ValidationError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    if not values:
        raise ValueError(f"{path}: expected at least one evaluation case")
    return values


def _validate_json_list(path: Path, model: type[WebFixture]) -> list[WebFixture]:
    try:
        raw: Any = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array")
    try:
        return [model.model_validate(value) for value in raw]
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error


def _validate_artifact_contents(benchmark: Benchmark, corpus_dir: Path) -> None:
    for document in benchmark.ledger.documents:
        artifact_text = _normalize_whitespace(_read_artifact(corpus_dir / document.filename))
        for passage in document.passages:
            expected = _normalize_whitespace(passage.text)
            if expected not in artifact_text:
                raise ValueError(f"{document.filename}: missing ledger passage {passage.id!r}")


def _read_artifact(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".docx":
        docx_document = DocxDocument(str(path))
        return "\n".join(paragraph.text for paragraph in docx_document.paragraphs)
    with pymupdf.open(path) as opened_pdf:
        pdf_document: Any = opened_pdf
        return "\n".join(
            cast(str, pdf_document.load_page(index).get_text("text"))
            for index in range(pdf_document.page_count)
        )


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the fixed RAG evaluation benchmark")
    parser.add_argument("--root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    args = parser.parse_args()

    try:
        benchmark = load_benchmark(args.root)
    except ValueError as error:
        parser.error(str(error))

    print(
        f"Validated {len(benchmark.ledger.documents)} documents, "
        f"{len(benchmark.cases)} cases, and "
        f"{len(benchmark.web_fixtures)} web fixtures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
