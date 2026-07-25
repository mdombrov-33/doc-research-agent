from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from src.config import get_settings
from src.core.citations import citation_from_artifact

_CONTENT_PREVIEW_CHARS = 1_500
_DIGEST_CHARS = 16


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


def _preview(value: str) -> str:
    compact = " ".join(value.split())
    if len(compact) <= _CONTENT_PREVIEW_CHARS:
        return compact
    return f"{compact[: _CONTENT_PREVIEW_CHARS - 1].rstrip()}…"


def text_log_fields(value: str, *, field: str) -> dict[str, Any]:
    """Return correlatable text metadata plus a bounded local-development preview."""
    fields: dict[str, Any] = {
        f"{field}_chars": len(value),
        f"{field}_sha256": _digest(value),
    }
    if get_settings().APP_ENV == "development":
        fields[f"{field}_preview"] = _preview(value)
    return fields


def evidence_log_fields(
    artifact: Mapping[str, Any], *, include_preview: bool = True
) -> dict[str, Any]:
    """Describe one evidence item without exposing its full content."""
    content = str(artifact.get("content", ""))
    citation = citation_from_artifact(artifact)
    source_type = "web" if artifact.get("source") == "web" else "document"
    fields: dict[str, Any] = {
        "source_type": source_type,
        "content_chars": len(content),
        "content_sha256": _digest(content),
    }
    if citation is not None:
        fields["source_id"] = citation.source_id

    metadata_keys = (
        ("document_id", "document_id"),
        ("chunk_id", "chunk_id"),
        ("chunk_index", "chunk_index"),
        ("page", "page"),
    )
    for output_key, artifact_key in metadata_keys:
        value = artifact.get(artifact_key)
        if value is not None:
            fields[output_key] = value

    if get_settings().APP_ENV == "development":
        for key in ("filename", "title", "url"):
            value = artifact.get(key)
            if value is not None:
                fields[key] = value
        if include_preview:
            fields["content_preview"] = _preview(content)
    return fields
