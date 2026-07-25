from datetime import date

import pytest

from evals.author import render_corpus
from evals.schemas import DocumentSpec, Fact, FactLedger, LedgerPassage, Pack
from src.core.ingestion.extract import extract_from_file


@pytest.mark.parametrize("extension", [".txt", ".docx", ".pdf"])
async def test_rendered_corpus_works_with_production_extraction(tmp_path, extension):
    sentence = "Full-time employees receive twenty-four annual leave days each year."
    ledger = FactLedger(
        packs=[Pack(id="people", title="People policies")],
        facts=[Fact(id="annual-leave", text=sentence)],
        documents=[
            DocumentSpec(
                id=f"handbook-{extension[1:]}",
                pack_id="people",
                filename=f"handbook{extension}",
                title="Employee handbook",
                published_on=date(2026, 1, 1),
                passages=[
                    LedgerPassage(
                        id="leave",
                        heading="Annual leave",
                        text=sentence,
                        fact_ids=["annual-leave"],
                    )
                ],
            )
        ],
    )

    render_corpus(ledger, tmp_path)
    rendered = tmp_path / f"handbook{extension}"
    first_render = rendered.read_bytes()
    render_corpus(ledger, tmp_path)
    extracted = await extract_from_file(
        str(rendered),
        rendered.name,
        max_pdf_pages=10,
        max_extracted_characters=10_000,
    )

    assert rendered.read_bytes() == first_render
    assert sentence in " ".join(extracted.split())
