# 07 — Uncached agentic RAG latency

## Goal

Make a new, uncached, document-grounded query feel interactive without converting the app into
naive one-shot RAG or removing its production safeguards.

The acceptance target is measured from API receipt to the first **visible answer token**:

- p50 ≤ 8 seconds;
- p95 ≤ 12 seconds.

Also measure full-answer completion. With concise-by-default answers, the initial target is:

- p50 ≤ 20 seconds;
- p95 ≤ 30 seconds.

These targets apply to the uncached document-answer path. Cache hits, conversational replies,
web fallback, and abstentions are reported separately. The selected UI answer model is recorded
with every result so unlike models are not silently mixed.

## Baseline and diagnosis

The motivating trace used a 17-page PDF (25 chunks) and a multipart security question:

| Stage | Observed wall time |
| --- | ---: |
| input guardrail | 2.24 s |
| agent/planner | 3.35 s |
| two retrievals plus two reranks | about 7.22 s |
| evidence assessment | 8.15 s |
| answer node, full generation | 30.20 s |
| reported first answer token | 21.74 s |
| API request, full completion | 52.23 s |

The repeated query completed in 1.94 seconds because the answer cache hit; 1.49 seconds of that
was still the input guardrail. The cache is working, but it does not solve the uncached path.

The current first-token measurement is optimistic: `stream.py` starts its graph clock only after
the input guardrail, conversational check, session-state probe, and answer-cache miss. The
21.74-second value was therefore roughly 24–25 seconds from the user's request. The improvement
must first make this metric honest.

The dominant structural waste is not Qdrant. The trace shows two agent-generated retrieval tool
calls. Each independently fetched about 42 candidates and ran the same CPU cross-encoder, so the
request paid for two contending reranks and then sent 20 chunks through both LLM stages. The
assessor and answer model consequently received much more evidence than they needed.

## Architectural constraints

These are red lines, not optional optimizations:

- **Keep LangGraph and the agentic retrieval planner.** The planner remains a mandatory graph
  node for substantive questions and remains responsible for history-aware search formulation.
- **Keep hybrid dense + sparse Qdrant retrieval, entity supplementation, the calibrated rerank
  floor, and cross-encoder reranking.**
- **Reranking is mandatory on every document-serving path.** There is no latency fast path that
  returns raw hybrid ordering.
- **Keep the model evidence-sufficiency gate before answer generation.** It fails closed and
  continues to validate supporting source IDs.
- **Keep input guardrails synchronous and fail closed before sending the question to an external
  model.**
- **Keep assessor-driven web fallback and safe abstention.**
- Do not add an automatic second document-retrieval attempt in this slice.
- OpenRouter remains the model transport. Replacing it with LiteLLM or conducting a broad
  provider migration is out of scope.

## Decided design

### 1. Measure user-visible latency

Capture the request clock at the stream handler boundary, before guardrails and every other
pre-graph step. Carry that same clock through all short-circuit and graph paths.

Record two distinct timings:

- **TTFT**: request receipt → first non-empty answer text emitted to the client;
- **completion**: request receipt → terminal SSE event.

Progress/status events, if added later, do not count as an answer token. Refusals, fixed
conversational replies, cached answers, document answers, web answers, abstentions, errors, and
timeouts retain separate path/outcome labels.

Keep the existing node timers and tracing, but make the hot path diagnosable as these stages:

1. input guardrail;
2. session/cache preflight;
3. planner;
4. candidate retrieval;
5. reranking;
6. document evidence assessment;
7. answer-provider TTFT;
8. answer completion.

Per-query telemetry records the selected answer model, planner model, assessor model, query
shape (single or multipart), candidate count, evidence count, and cache/path outcome. This is
stage attribution, not a new observability platform.

### 2. Preserve the agent, separate its model role

Today `agent_node` uses the user-selected answer model. That makes a small, structured planning
step pay answer-model latency.

Split the roles:

- **planner model** — recent-history interpretation, standalone query formulation, and the
  retrieval tool call;
- **assessor model** — structured evidence-sufficiency verdict;
- **answer model** — the model selected in the UI, used only for final synthesis.

Planner and assessor are separately configurable even if both initially point to the existing
Luna model. This keeps their prompts, token limits, and future model choices independent without
changing the UI answer-model contract. Haiku or Flash can later be selected through
configuration; choosing a new default model is not part of this improvement.

The existing conservative pre-graph conversational matcher remains. Every other substantive
question enters the LangGraph planner. The planner does not answer the user.

### 3. Replace repeated retrieval executions with one batched agent tool

Change the document tool contract from “one query produces one final reranked evidence set” to
“one agent plan produces one final reranked evidence set.”

The planner makes exactly one document-tool call containing:

- one search query for a simple question; or
- at most two focused search queries for a genuinely multipart question.

For a simple first-turn question, the expected query is the user's wording without gratuitous
rewriting. For a context-dependent follow-up, the planner still resolves the query using the
recent conversation history, as it does today. The two-query maximum is enforced by the tool
schema/runtime contract, not only by prompt wording.

This remains agentic RAG: the model interprets the question and chooses the search formulation.
The optimization is that the retrieval subsystem executes the resulting plan as one bounded
unit rather than treating each tool call as an independent end-to-end RAG pipeline.

### 4. Retrieve concurrently, merge once, rerank once

Separate candidate collection from final evidence selection. The current `hybrid_search()`
combines both, which forces every query branch to rerank independently.

The batched retrieval operation:

1. runs the one or two Qdrant hybrid searches concurrently;
2. keeps entity supplementation within each search branch;
3. merges results without favoring whichever branch happens to finish first;
4. deduplicates chunks by stable chunk identity;
5. applies one global candidate budget;
6. reranks the merged pool exactly once against the current user question;
7. applies the existing calibrated rerank-score floor;
8. returns one bounded evidence set.

The internal budgets are:

- **candidate budget: 40 unique chunks** before cross-encoder reranking;
- **evidence budget: at most 8 chunks** after reranking and the calibrated floor.

For two queries, 40 is the global merged maximum, not 40 per branch. The merge must be
rank-aware and deterministic rather than concatenating one branch ahead of the other. Empty
post-floor evidence keeps today's safe behavior: the assessor sees no support and the graph
falls back to web or abstains.

The reranker is warmed at startup as today. Its serving failure must be visible and must not
silently degrade to raw vector ordering. CPU/thread configuration and batch behavior may be
tuned, but replacing the cross-encoder with a naive similarity cutoff is outside the design.

### 5. Remove public `top_k`

`top_k` currently mixes three different concerns: Qdrant fetch width, reranker output width,
and the amount of evidence shown to LLMs. It also lets the UI accidentally expand the most
expensive parts of the request.

Remove it from:

- `QueryRequest` and the `/api/stream` contract;
- the Streamlit slider and request body;
- `RunnableConfig`;
- the retrieval tool interface;
- answer-cache lookup/store identity;
- public docs and tests.

Replace it with the two internal concepts above: candidate budget and evidence budget. They are
application policy, not user knobs.

The personal-project cache is disposable: clear existing answer-cache entries when this change
ships. Do not build cache migration or pipeline-version machinery for this slice.

### 6. Bound auxiliary LLM work

The planner and assessor prompts remain narrow and use structured outputs/tool arguments. They
should not produce prose, chain-of-thought, or answer-sized completions.

The assessor remains universal for document evidence in the first implementation. Do not add a
score-only sufficiency bypass. The present golden set contains positive questions but not enough
out-of-corpus and partially answerable cases to justify removing the model gate.

Assessment receives only the final eight-chunk evidence set, not every per-query candidate.
Malformed output, invalid supporting IDs, timeouts, and model failures continue to fail closed.

If document evidence is insufficient, retain the current single web-fallback path and reassess
the combined evidence. Do not add a document re-plan/retrieve loop in Improvement 07; that would
create another unbounded latency branch and is a separate quality feature.

### 7. Keep answers concise by default

The motivating answer generated 1,625 output tokens and held the request open long after the
first token. Preserve streaming and citation requirements, but make ordinary answers
proportional to the question and target roughly 800–1,000 output tokens at most. A user can ask
for expansion in a follow-up.

This is not permission to truncate mid-citation or omit required parts of a multipart answer.
Prompt guidance and the model's output limit must leave room for a complete, cited response.

### 8. Optimize safeguards in place

The input toxicity scanner already loads at startup and runs off the event loop. Keep both
patterns. Its observed 1.5–2.2-second warm latency is now part of the real TTFT budget, so inspect
the actual inference work for avoidable per-request setup, CPU-thread contention, or duplicate
scanning.

No external planner, embedding, retrieval, or cache-semantic-search call may begin before the
guardrail passes merely to make the stopwatch look better.

## Graph after the change

The graph remains recognizably the current agentic graph:

```text
agent/planner
      │ one batched retrieve_documents call
      ▼
batched retrieval
  ├─ query A: hybrid + entity candidates ─┐
  └─ query B: hybrid + entity candidates ─┤ optional
                                          ▼
                               merge + deduplicate
                                          │ ≤ 40
                                          ▼
                                one cross-encoder rerank
                                          │ ≤ 8
                                          ▼
                                  assess_evidence
                                    │          │
                              sufficient   insufficient
                                    │          ▼
                                    │      web_fallback
                                    │          │
                                    │     assess_evidence
                                    ▼          ▼
                                  answer    answer/abstain
```

This is not a deterministic replacement for the agent. It is a deeper retrieval tool beneath
the agent: one bounded plan in, one quality-controlled evidence set out.

## Expected latency effect

The design attacks each measured source of pre-answer delay:

| Current cost | Structural change |
| --- | --- |
| answer model used for 3.35 s planner step | dedicated small planner role |
| two independent 42-candidate retrieval pipelines | one batched retrieval plan |
| two contending cross-encoder passes | one merged 40-candidate rerank |
| 20 chunks sent into assessment and generation | at most 8 final evidence chunks |
| 8.15 s open-ended assessment | bounded structured assessor role and smaller context |
| 1,625-token ordinary answer | concise, bounded answer policy |
| TTFT excludes guard/cache preflight | request-boundary clock |

This spec does not promise that OpenRouter or a particular selectable answer model will always
meet the SLO. It makes the application's own work bounded and exposes the remaining provider
time honestly.

## Files and module boundaries

- `src/config.py`
  - separate planner and assessor model settings;
  - internal candidate/evidence budgets;
  - remove the `top_k`-derived rerank sizing policy.
- `src/api/schemas.py`, `ui.py`
  - remove public `top_k`.
- `src/api/handlers/stream.py`
  - start request-wide timing at the handler boundary;
  - remove `top_k` propagation and cache identity;
  - record path/stage dimensions without changing SSE answer semantics.
- `src/core/agent/prompts.py`, `nodes.py`
  - keep the mandatory agent planner;
  - bind the planner model to the batched tool contract;
  - keep assessment fail-closed and bound its input/output;
  - make answer verbosity proportional and bounded.
- `src/core/agent/tools.py`, `graph.py`
  - expose one batched retrieval tool call while preserving the existing LangGraph
    agent → tool → assessment topology and web loop.
- `src/core/retrieval/search.py`
  - split per-query candidate collection from merged evidence selection;
  - concurrent branches, stable deduplication, global candidate budget.
- `src/core/retrieval/rerank.py`
  - one rerank against the current user question;
  - evidence budget replaces `top_k`;
  - retain calibrated floor and startup warmup.
- `src/core/answer_cache.py`
  - remove `top_k` payload/filter/index usage; existing cache data may be discarded.
- `src/core/monitoring/`
  - make request-wide TTFT/completion and the relevant path/model dimensions queryable.
- Tests, `README.md`, and `docs/architecture.md`
  - replace the old one-query/one-rerank and public-`top_k` contracts.

## Delivery slices

Implement in reviewable slices so measurement proves each structural change:

1. **Honest clock and stage attribution** — correct TTFT/completion boundaries before changing
   behavior.
2. **Role separation** — planner no longer inherits the selected answer model; assessor gets its
   own setting.
3. **Batched retrieval boundary** — one/two planned queries, concurrent candidate searches,
   merge/deduplicate, one rerank, eight evidence chunks.
4. **Remove `top_k`** — API, UI, config propagation, cache, docs, and tests.
5. **Bound LLM payloads** — focused auxiliary completions and concise answer policy.
6. **Regression and latency proof** — run the same uncached queries repeatedly and record the
   before/after stage breakdown.

Slices may be combined when a temporary interface would be misleading, but the final behavior
must not ship with two retrieval contracts or a hidden compatibility path.

## Verification

### Functional

- A simple document question produces one planner tool call containing one query.
- A context-dependent follow-up uses recent conversation context and produces one standalone
  query.
- A multipart question can produce two queries but never more.
- Two query branches execute concurrently, merge deterministically, and deduplicate repeated
  chunks.
- No more than 40 unique candidates enter one reranker call.
- No more than eight post-floor document chunks reach assessment and answer.
- Exactly one document rerank occurs per substantive turn.
- The calibrated score floor still removes below-threshold chunks.
- Empty post-floor evidence still reaches web fallback and then abstains if necessary.
- Invalid assessor source IDs and assessor failures still fail closed.
- The selected UI model is used only for answer synthesis, not planning.
- Cache hits still preserve checkpoint history for later follow-ups.
- `top_k` is absent from API schema, UI, config, cache, docs, and serving tests.

### Quality

Keep the existing positive golden-set retrieval checks and add a small set of:

- out-of-corpus questions;
- partially answerable questions;
- multipart questions whose evidence is split across chunks;
- follow-ups that require history-aware query formulation.

The refactor must preserve document-level recall and citation validity. The new negative cases
must continue to fall back or abstain rather than passing the assessor because the context is
smaller.

### Performance

Use a clean answer cache and an already-started application so ingestion and model download are
not mixed into query latency. Run a representative set containing simple, follow-up, and
multipart document questions. Report p50/p95 for:

- real user-visible TTFT;
- full completion;
- every hot-path stage listed above;
- the number of candidate branches, reranker calls, candidates, and final evidence chunks.

The headline pass condition is the 8-second p50 / 12-second p95 uncached document-answer TTFT
target without a retrieval or citation-quality regression. Web-escalated requests are reported
but evaluated separately.

Run the normal test, lint, type, retrieval-eval, and full graph-eval targets used by the repo.

## Rejected approaches

- Removing the LangGraph agent and replacing it with a fixed route.
- Skipping reranking for “easy” queries.
- Passing raw hybrid results when the reranker fails.
- Removing the model evidence gate based only on retrieval scores.
- Adding a second document retrieval/reformulation loop in the latency slice.
- Increasing Qdrant parallelism while retaining independent per-query rerank pipelines.
- Treating cache-hit latency as proof that uncached latency is fixed.
- Starting TTFT after guardrails or counting progress events as answer tokens.
- Migrating from OpenRouter to LiteLLM as a presumed latency fix.
- Adding cache migration/version infrastructure for this personal project.

## Definition of done

The graph is still agentic; each turn performs at most one batched document-tool call and one
mandatory merged rerank; public `top_k` is gone; assessment receives at most eight document
chunks and remains fail-closed; request-boundary TTFT and completion are recorded honestly;
quality checks remain green; the before/after stage and p50/p95 results are recorded in this
file; `docs/architecture.md`, `README.md`, environment documentation, and `progress.md` match
the shipped behavior.

## Implementation evidence — 2026-07-25

Implemented:

- independent planner, assessor, and selectable answer-model roles with bounded completions;
- one runtime-bounded planner tool call containing one or two queries;
- concurrent hybrid/entity candidate branches, deterministic RRF merge, stable deduplication,
  global 40-candidate cap, and one mandatory cross-encoder pass to at most eight chunks;
- removal of the public retrieval-width knob from request/UI/config/cache/evals/docs, with a
  one-time reset of legacy cache entries;
- API-handler-boundary TTFT/completion clocks plus safe role/path/query-shape/candidate/evidence/
  reranker telemetry;
- concise synthesis cap and live graph cases for out-of-corpus, partial, multipart, and
  history-aware follow-up behavior.

The input guardrail path was audited during implementation. It already constructs and warms one
cached `Toxicity` scanner, performs one scan per request, and runs inference off the event loop;
there is no duplicate scan or per-request model setup to remove. Its optional ONNX path is not
available in the current dependency set, so no unmeasured runtime/dependency change was made to
this fail-closed gate.

Local deterministic verification passes:

- `uv run pytest -q` — 183 tests;
- `uv run ruff check .`;
- `uv run mypy src`.

The paid live model calls needed for uncached p50/p95 and the full retrieval/graph eval were not
run during implementation. Do not mark this item fully done until a warm app with an empty answer
cache has produced the before/after stage table and confirmed or rejected the headline SLO.
