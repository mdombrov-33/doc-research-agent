# 03 — Ingestion dedupe (file-level)

## Goal

Re-uploading the same file re-extracts, re-chunks, and re-embeds everything — wasted OpenAI
spend and duplicate chunks polluting retrieval.

## Decided design

- **File-level only.** Chunk-level embedding caching was rejected: it requires abandoning
  `add_documents` for manual embedding management, and only pays off for a document-update
  workflow that doesn't exist (append-only corpus).
- SHA-256 the uploaded bytes; store the hash as `file_sha256` in every chunk payload at
  indexing time.
- On upload, a filtered Qdrant lookup on `file_sha256`; on match, return the existing
  `document_id` plus a `duplicate: true` flag in the response, skipping
  extract/chunk/enrich/embed entirely.
- **A deduped upload must not bump `corpus_version`** — the corpus didn't change, so the
  answer cache (item 02) must not be flushed.

## Files

- `src/api/handlers/upload.py` — hash while streaming to disk, duplicate lookup, response
  flag, conditional version bump.
- `src/core/ingestion/index.py` — `file_sha256` in chunk metadata.
- Upload response schema.

## Verification

- Unit/integration tests: same bytes twice → second response has `duplicate: true`, same
  `document_id`, no new chunks, no version bump; different file → normal path.
- `make test`, `make lint`.

## Definition of done

Code + tests green; `docs/architecture.md` ingestion section and `README.md` (upload API)
updated; `progress.md` marked.
