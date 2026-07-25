# 02 — Answer cache (exact + semantic)

## Goal

A repeated question currently pays the full graph (~5–15 s, full token spend). A cache turns
a repeat into milliseconds and zero tokens. Depends on item 01 so savings are measurable.

## Decided design

- **One Qdrant collection `answer_cache`, both layers in one entry.** Point = question
  embedding (`text-embedding-3-small`) + payload: `question_hash` (SHA-256 of the normalized
  question), `question`, `answer`, `sources` (JSON), `model`, `top_k`, `corpus_version`,
  `namespace: "default"`, `created_at`. No Redis/GPTCache — we already run Qdrant and the
  embedder.
- **L1 exact**: filtered lookup on `question_hash + model + top_k + corpus_version +
  namespace` — no embedding call. **L2 semantic** on exact miss: embed once, vector search
  with the same payload filter, accept at cosine ≥ 0.95. The spaCy entity-match guard is
  deferred until real collisions are observed.
- **Cache key includes `model` and `top_k`**: per-request knobs change the answer and the
  sources list; serving one model's answer as another's misrepresents provenance.
- **Invalidation: invalidate-all-on-upload.** A monotonically increasing `corpus_version`
  lives as a reserved point (nil-UUID id, zero vector) in `answer_cache`; every successful
  non-duplicate upload bumps it. Only current-version entries are served; on bump, delete
  older-version points. Rationale: the corpus is append-only and a new upload can make any
  cached answer stale — one integer comparison makes staleness impossible by construction.
  The selective supporting-doc design was rejected as a correctness bug.
- **TTL 24 h** as growth/staleness backstop, enforced by a `created_at` range filter at read
  (Qdrant has no native TTL).
- **Only `document_answer` outcomes are cached.** Web answers (live-search-dependent) and
  abstentions (fixable by tomorrow's web) always re-run the graph.
- **First-turn-only rule**: consult/populate the cache only when the session's checkpoint has
  no prior messages (`agent.aget_state(config)`). A follow-up like "expand on that" must
  never hit the cache.
- **Serving a hit**: one SSE `token` event with the full answer, then the normal `done` event
  with the cached sources. Record `cache_hit=true` in `query_events` with cost 0.
- **Checkpoint continuity on a hit**: after serving from cache, write the Human + AI messages
  into the checkpoint (`aupdate_state`) so later follow-ups still see the turn in history.
- **Placement** in `stream.py`: guardrail → conversational check (item 05) → cache check →
  graph.

## Files

- `src/core/answer_cache.py` (new) — collection bootstrap, version point, lookup/store/flush.
- `src/api/handlers/stream.py` — cache consult/serve/populate around the graph run.
- `src/api/handlers/upload.py` — version bump on successful upload.
- `src/core/monitoring/` — `cache_hit` column in `query_events`, `cache_hit_rate` in stats.

## Verification

- Unit tests: key composition, version filtering, TTL filter, outcome gating,
  first-turn gating, hit replay shape.
- Manual SSE smoke: repeat question → instant answer + identical sources; upload → same
  question re-runs graph; follow-up turn bypasses cache; checkpoint history intact after hit.
- `make test`, `make lint`.

## Definition of done

Code + tests green; smoke checks pass; `docs/architecture.md` (flow + new collection) and
`README.md` updated; `progress.md` marked.
