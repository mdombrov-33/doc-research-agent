# 04 — Contextual chunk headers

## Goal

The embedded text of each chunk is the bare chunk; a chunk like "It costs $200/month" is
unfindable once separated from its document (Anthropic's context-dilution findings — the
strongest retrieval-quality lever in the audit).

## Decided design

- **Prepend `Document: {filename}` into each chunk's `page_content`** before indexing. The
  header flows into the dense embedding, BM25 sparse indexing, and the evidence the answer
  model sees — the last part is intentional (helps the model attribute evidence).
- The extractor produces flat text with no section headings, so the header is filename-only.
- Prepend at the enrich/index stage, after chunking and the ≥100-char chunk filter, so the
  header doesn't inflate chunk-length decisions.
- `add_documents` stays; no manual embedding management.
- **LLM contextualization (situating strings) is deferred** until the free header's eval
  delta plateaus.
- Existing indexed documents only benefit on re-upload — accepted at dev stage; no migration.

## Files

- `src/core/ingestion/enrich.py` or `index.py` — header prepend.
- `evals/` — no code change, but the gate below.

## Verification

- **Gate: `make eval-retrieval` before and after** — recall@k / nDCG must not regress; the
  change only ships if the numbers hold or improve (golden-set corpus re-indexed with
  headers for the "after" run).
- Unit test: indexed `page_content` carries the header; chunk filter unaffected.
- `make test`, `make lint`.

## Definition of done

Code + tests green; eval delta recorded in this file; `docs/architecture.md` ingestion
section updated; `progress.md` marked.
