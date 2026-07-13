from collections.abc import Iterable, Mapping
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

SourceType = Literal["document", "web"]

MAX_CITATION_EXCERPT_CHARS = 280


class SourceCitation(BaseModel):
    """A client-facing reference to one piece of evidence used for a response."""

    source_id: str = Field(min_length=1)
    source_type: SourceType
    title: str = Field(min_length=1)
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = Field(default=None, ge=1)
    url: str | None = None
    excerpt: str = Field(min_length=1, max_length=MAX_CITATION_EXCERPT_CHARS)

    @model_validator(mode="after")
    def validate_source_fields(self) -> "SourceCitation":
        if self.source_type == "document":
            if not self.document_id or not self.chunk_id:
                raise ValueError("Document citations require document_id and chunk_id")
            if self.url is not None:
                raise ValueError("Document citations cannot include a URL")
        else:
            if self.document_id is not None or self.chunk_id is not None:
                raise ValueError("Web citations cannot include document identifiers")
            if not self.url or not _is_http_url(self.url):
                raise ValueError("Web citations require an HTTP(S) URL")
        return self


def citations_from_artifacts(artifacts: Iterable[Mapping[str, Any]]) -> list[SourceCitation]:
    """Build ordered, deduplicated client citations from tool evidence artifacts.

    Invalid artifacts are intentionally omitted: evidence that cannot identify its source must
    not be presented to users as a citation.
    """
    citations: list[SourceCitation] = []
    seen_source_ids: set[str] = set()
    for artifact in artifacts:
        citation = citation_from_artifact(artifact)
        if citation is not None and citation.source_id not in seen_source_ids:
            seen_source_ids.add(citation.source_id)
            citations.append(citation)
    return citations


def citation_from_artifact(artifact: Mapping[str, Any]) -> SourceCitation | None:
    """Translate the internal document or web evidence shape into the public contract."""
    excerpt = _excerpt(artifact.get("content"))
    if not excerpt:
        return None

    if artifact.get("source") == "web":
        title = _text(artifact.get("title"))
        url = _text(artifact.get("url"))
        if not title or not url or not _is_http_url(url):
            return None
        return SourceCitation(
            source_id=f"web:{url}",
            source_type="web",
            title=title,
            url=url,
            excerpt=excerpt,
        )

    document_id = _text(artifact.get("document_id"))
    if not document_id:
        return None
    chunk_index = artifact.get("chunk_index", 0)
    chunk_id = _text(artifact.get("chunk_id")) or f"{document_id}:{chunk_index}"
    title = _text(artifact.get("filename")) or "Unknown document"
    page = artifact.get("page")
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        page = None
    return SourceCitation(
        source_id=f"document:{chunk_id}",
        source_type="document",
        title=title,
        document_id=document_id,
        chunk_id=chunk_id,
        page=page,
        excerpt=excerpt,
    )


def _excerpt(value: Any) -> str | None:
    content = _text(value)
    if not content:
        return None
    compact = " ".join(content.split())
    if len(compact) <= MAX_CITATION_EXCERPT_CHARS:
        return compact
    return f"{compact[: MAX_CITATION_EXCERPT_CHARS - 1].rstrip()}…"


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
