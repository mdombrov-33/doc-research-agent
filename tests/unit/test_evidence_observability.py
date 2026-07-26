from types import SimpleNamespace

from src.core import evidence_observability


def _document() -> dict:
    return {
        "content": "  A retained slice\nkeeps the backing array alive.  ",
        "document_id": "doc-1",
        "chunk_id": "doc-1:26",
        "chunk_index": 26,
        "filename": "book.pdf",
        "page": 101,
        "source": "document",
    }


def test_evidence_log_fields_include_bounded_content_preview_in_development(monkeypatch):
    monkeypatch.setattr(
        evidence_observability,
        "get_settings",
        lambda: SimpleNamespace(APP_ENV="development"),
    )

    fields = evidence_observability.evidence_log_fields(_document())

    assert fields == {
        "source_id": "document:doc-1:26",
        "source_type": "document",
        "document_id": "doc-1",
        "chunk_id": "doc-1:26",
        "chunk_index": 26,
        "filename": "book.pdf",
        "page": 101,
        "content_chars": 51,
        "content_sha256": fields["content_sha256"],
        "content_preview": "A retained slice keeps the backing array alive.",
    }
    assert len(fields["content_sha256"]) == 16


def test_evidence_log_fields_exclude_content_preview_outside_development(monkeypatch):
    monkeypatch.setattr(
        evidence_observability,
        "get_settings",
        lambda: SimpleNamespace(APP_ENV="production"),
    )

    fields = evidence_observability.evidence_log_fields(_document())

    assert "content_preview" not in fields
    assert "filename" not in fields
    assert fields["source_id"] == "document:doc-1:26"
    assert fields["content_chars"] == 51
    assert len(fields["content_sha256"]) == 16


def test_text_log_fields_only_expose_text_in_development(monkeypatch):
    monkeypatch.setattr(
        evidence_observability,
        "get_settings",
        lambda: SimpleNamespace(APP_ENV="development"),
    )
    development = evidence_observability.text_log_fields("first\n  focused query", field="query")

    monkeypatch.setattr(
        evidence_observability,
        "get_settings",
        lambda: SimpleNamespace(APP_ENV="production"),
    )
    production = evidence_observability.text_log_fields("first\n  focused query", field="query")

    assert development["query_preview"] == "first focused query"
    assert production == {
        "query_chars": 21,
        "query_sha256": development["query_sha256"],
    }
