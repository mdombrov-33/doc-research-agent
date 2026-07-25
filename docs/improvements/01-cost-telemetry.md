# 01 — Token/cost telemetry per query

## Goal

`QueryMetrics` records latency and counts but not tokens or cost. Every later cost decision
(cache ROI, routing, model choice) is guesswork without this, so it lands before the cache.

## Decided design

- **New per-query rows table `query_events`** (sqlite and postgres backends): timestamp,
  model, input_tokens, output_tokens, reported_cost, latency_ms, time_to_first_token_ms,
  outcome. Item 02 later adds a `cache_hit` column. No question/answer/evidence content —
  preserves the telemetry privacy stance (architecture §14).
- The existing aggregate `monitoring_stats` row stays untouched; new stats are computed by
  SQL over `query_events`.
- **Cost source: OpenRouter-reported actual cost**, requested via usage accounting
  (`extra_body={"usage": {"include": true}}` on the ChatOpenAI calls) and read from response
  metadata. A static price table was explicitly rejected (goes stale, ignores prompt-cache
  discounts). If the cost field turns out not to survive LangChain streaming, record tokens
  and leave `reported_cost` NULL — do not build the price table.
- **Token capture**: `on_chat_model_end` events in the `astream_events` loop in
  `src/api/handlers/stream.py` carry `usage_metadata`; sum across the turn's LLM calls
  (agent, classifier, answer). The streamed answer call needs `stream_usage=True` on the
  ChatOpenAI client (`src/core/llm.py`).
- **Embeddings excluded** from per-query cost: the query-time embedding call is negligible
  next to LLM spend; ingestion embedding cost is per-upload, not per-query.
- New fields in `/api/monitoring/stats`: `avg_input_tokens`, `avg_output_tokens`,
  `avg_cost_per_query`, plus a per-model breakdown.

## Files

- `src/core/monitoring/db.py` — `query_events` table (both backends), `record_event` on the
  `MetricsStore` protocol, stats queries.
- `src/core/monitoring/tracker.py` — extend `QueryMetrics` with token/cost fields; record
  both aggregate row and event row.
- `src/core/llm.py` — usage accounting `extra_body`, `stream_usage=True`.
- `src/api/handlers/stream.py` — collect usage/cost from `on_chat_model_end` events.
- Stats route/schema for the new fields.

## Empirical finding (2026-07-25)

Probed OpenRouter through this exact stack. `usage_metadata` (token counts) survives both
`ainvoke` and streaming. The provider **cost** field lands in `response_metadata["token_usage"]["cost"]`
on a plain `ainvoke`, but LangChain drops it whenever the call is streamed — and the stream
handler runs the graph through `astream_events`, which streams *every* model call (even nodes
written as `.invoke`). So `reported_cost` is captured-if-present but is currently always NULL.
Per the decision above, we record tokens and leave cost NULL; no price table. The column and the
`extra_body` usage-accounting request stay in place so cost populates automatically if a
non-streamed path (e.g. the item 02 cache) or a LangChain fix ever provides it.

## Verification

- Unit tests: db `record_event`/stats queries (both backends' SQL), tracker field flow,
  stream handler usage collection (fake events).
- Empirical check against OpenRouter: done — see finding above (cost does not survive streaming).
- `make test`, `make lint`.

## Definition of done

Code + tests green; `/api/monitoring/stats` shows the new fields; `docs/architecture.md`
telemetry section and `README.md` (if it documents stats) updated; `progress.md` marked.
