# 05 — Conversational no-retrieval path

## Goal

A greeting or meta-question ("what can you do?") flows through the full graph and ends as a
hard abstention — technically correct, bad UX, and it pollutes `abstention_rate`.

## Decided design

- **Conservative rules only.** Normalized exact/near-exact match against a fixed
  greeting/thanks set plus help patterns ("what can you do", "help", …). Anything with real
  content words falls through to the graph. False positives are ~impossible by construction;
  the ambiguous tail ("can you help me with something?") takes the normal graph path — same
  outcome as today, no worse.
- A classifier tail (`CLASSIFIER_MODEL`) was rejected: it adds an LLM call and latency to the
  hot path of every short query to fix a paper cut.
- Returns a **fixed capabilities-style response** (what the app does, how to use it) without
  touching retrieval.
- **New outcome value `conversational`**: added to `FinalOutcome` / `normalize_outcome`
  (`src/core/agent/outcomes.py`), counted via a new aggregate column using the existing
  column-migration pattern in `db.py` so it isn't miscounted as an abstention; excluded from
  `abstention_rate`; gets its own rate in stats. Recorded in `query_events` too.
- **Runs pre-graph in `stream.py`**: guardrail → conversational check → cache check (item
  02) → graph.
- No checkpoint write for these turns — losing "hi" from history is harmless.

## Files

- `src/api/handlers/stream.py` — the check + fixed-response short-circuit.
- New small module or function for the rule set (e.g. `src/core/conversational.py`).
- `src/core/agent/outcomes.py`, `src/core/monitoring/db.py`/`tracker.py`, stats schema.

## Verification

- Unit tests: each rule-set entry short-circuits; content-bearing queries (incl. tricky ones
  like "hi, what does the contract say about X?") fall through; outcome recorded as
  `conversational` and excluded from abstention_rate.
- Manual SSE smoke: "hi" answers instantly with the fixed response.
- `make test`, `make lint`.

## Definition of done

Code + tests green; `docs/architecture.md` flow/outcomes updated; `README.md` if it lists
outcomes; `progress.md` marked.
