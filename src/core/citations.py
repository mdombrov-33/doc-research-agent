import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

SourceType = Literal["document", "web"]

MAX_CITATION_EXCERPT_CHARS = 280
_SOURCE_ID_MARKER = re.compile(r"\[(?P<source_id>(?:document|web):[^\]\s]+)\]")
_SOURCE_ID_WITH_LEADING_SPACE_MARKER = re.compile(
    r"[ \t]*\[(?:document|web):[^\]\s]+\]"
)


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


def citations_referenced_by_answer(
    artifacts: Iterable[Mapping[str, Any]], answer: str
) -> list[SourceCitation]:
    """Return only real evidence explicitly referenced by the final answer, in answer order."""
    citations_by_id = {
        citation.source_id: citation for citation in citations_from_artifacts(artifacts)
    }
    selected: list[SourceCitation] = []
    seen_source_ids: set[str] = set()
    for source_id in source_ids_from_answer(answer):
        citation = citations_by_id.get(source_id)
        if citation is not None and source_id not in seen_source_ids:
            seen_source_ids.add(source_id)
            selected.append(citation)
    return selected


def source_ids_from_answer(answer: str) -> list[str]:
    """Extract square-bracket source IDs in their first-mentioned order."""
    return [match.group("source_id") for match in _SOURCE_ID_MARKER.finditer(answer)]


def strip_source_ids(answer: str) -> str:
    """Remove internal source-ID markers before an answer is displayed to a user."""
    return _SOURCE_ID_WITH_LEADING_SPACE_MARKER.sub("", answer)


class CitationMarkerRedactor:
    """Hide source-ID markers while preserving ordinary streaming text.

    Model tokens may split a marker across arbitrary chunks. Horizontal whitespace before a
    possible marker remains buffered so a hidden marker does not leave a space before
    punctuation in the displayed answer.
    """

    def __init__(self) -> None:
        self._pending = ""

    def push(self, chunk: str) -> str:
        """Return the visible portion of a streamed model chunk."""
        self._pending += chunk
        visible: list[str] = []

        while self._pending:
            marker_start = self._pending.find("[")
            if marker_start == -1:
                visible_text = self._pending.rstrip(" \t")
                visible.append(visible_text)
                self._pending = self._pending[len(visible_text) :]
                break

            marker_end = self._pending.find("]", marker_start + 1)
            prefix = self._pending[:marker_start]
            if marker_end == -1:
                visible_prefix = prefix.rstrip(" \t")
                visible.append(visible_prefix)
                self._pending = self._pending[len(visible_prefix) :]
                break

            marker = self._pending[marker_start : marker_end + 1]
            if _SOURCE_ID_MARKER.fullmatch(marker):
                visible.append(prefix.rstrip(" \t"))
            else:
                visible.append(prefix + marker)
            self._pending = self._pending[marker_end + 1 :]

        return "".join(visible)

    def flush(self) -> str:
        """Return any buffered ordinary text once the model has finished streaming."""
        visible = strip_source_ids(self._pending)
        self._pending = ""
        return visible


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
