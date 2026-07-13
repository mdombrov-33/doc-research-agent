import pytest
from pydantic import ValidationError

from src.core.citations import (
    SourceCitation,
    citation_from_artifact,
    citations_from_artifacts,
    citations_referenced_by_answer,
)


def test_document_artifact_becomes_citation_with_short_evidence_excerpt():
    citation = citation_from_artifact(
        {
            "source": "document",
            "document_id": "doc-123",
            "chunk_id": "doc-123:4",
            "filename": "report.pdf",
            "page": 2,
            "content": "Useful evidence\nfrom the report.",
        }
    )

    assert citation is not None
    assert citation.model_dump(exclude_none=True) == {
        "source_id": "document:doc-123:4",
        "source_type": "document",
        "title": "report.pdf",
        "document_id": "doc-123",
        "chunk_id": "doc-123:4",
        "page": 2,
        "excerpt": "Useful evidence from the report.",
    }


def test_web_artifact_becomes_citation_only_with_real_http_url():
    citation = citation_from_artifact(
        {
            "source": "web",
            "title": "Official release notes",
            "url": "https://example.com/releases",
            "content": "Version 2.0 was released today.",
        }
    )

    assert citation is not None
    assert citation.source_id == "web:https://example.com/releases"
    assert citation.source_type == "web"
    assert citation.url == "https://example.com/releases"


def test_invalid_evidence_is_not_presented_as_a_citation():
    assert citation_from_artifact({"source": "web", "content": "No URL"}) is None
    assert citation_from_artifact({"source": "document", "content": "No document id"}) is None


def test_citations_deduplicate_by_source_id_and_preserve_evidence_order():
    artifacts = [
        {
            "source": "web",
            "title": "First",
            "url": "https://example.com/first",
            "content": "First result",
        },
        {
            "source": "web",
            "title": "First duplicate title",
            "url": "https://example.com/first",
            "content": "Repeated result",
        },
        {
            "source": "document",
            "document_id": "doc-1",
            "chunk_index": 0,
            "filename": "notes.txt",
            "content": "Document result",
        },
    ]

    citations = citations_from_artifacts(artifacts)

    assert [citation.source_id for citation in citations] == [
        "web:https://example.com/first",
        "document:doc-1:0",
    ]


def test_citations_referenced_by_answer_ignores_unreferenced_and_unknown_evidence():
    artifacts = [
        {
            "source": "document",
            "document_id": "doc-1",
            "chunk_index": 0,
            "filename": "notes.txt",
            "content": "Document result",
        },
        {
            "source": "web",
            "title": "Official source",
            "url": "https://example.com/release",
            "content": "Web result",
        },
    ]
    answer = (
        "The release is available now [web:https://example.com/release]. "
        "Unknown evidence [document:not-real:9]. "
        "The release remains available [web:https://example.com/release]."
    )

    citations = citations_referenced_by_answer(artifacts, answer)

    assert [citation.source_id for citation in citations] == ["web:https://example.com/release"]


def test_source_citation_rejects_source_type_specific_fields():
    with pytest.raises(ValidationError, match="Web citations require an HTTP"):
        SourceCitation(
            source_id="web:invalid",
            source_type="web",
            title="Invalid",
            excerpt="Evidence",
        )
