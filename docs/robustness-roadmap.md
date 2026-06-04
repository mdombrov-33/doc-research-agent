# Robustness Roadmap

> **Purpose of this file.** A self-contained, resumable plan to harden the
> doc-research-agent. It is written so that a fresh session (after `/clear`) can
> read *only this file* plus the referenced source and continue the work. Each
> phase is independently executable and has explicit verification criteria.
> Update the checkboxes and the "Status log" at the bottom as work progresses.

Origin: review of the system against *AI Engineering* (Chip Huyen), ch. 10.
The book's architecture progression is: enhance context → guardrails → router/gateway
→ caching → agent patterns → observability. We already have context (hybrid RAG +
entity filter), a router node, and OpenRouter as a de-facto gateway. This roadmap
fills the remaining gaps: **retries/fallbacks, guardrail correctness, real evaluation,
and small cleanups.**

---

## Working principles (from CLAUDE.md — obey these)

- **Simplicity first.** Minimum code that solves the problem. No speculative config.
- **Surgical changes.** Every changed line traces to a roadmap item. Don't refactor
  adjacent code. Match existing style.
- **Goal-driven.** Each phase below states success criteria. Loop until verified
  (lint + mypy + pytest green, plus the phase's own checks).
- Run quality gates with: `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest`

---

## Current architecture (as of this writing)

Agent graph (`src/core/agent.py`):
```
router → retrieve → [websearch?] → grade_documents → [web-fallback loop] → generate → END
```
- **LLM access** — `src/core/llm.py::get_llm()` returns a `langchain_openai.ChatOpenAI`
  pointed at OpenRouter (`https://openrouter.ai/api/v1`). Called via
  `.invoke / .ainvoke / .batch / .with_structured_output`. **No retry, no fallback today.**
- **Router / grading** — `src/core/grading/graders.py` uses `with_structured_output`
  (Pydantic `RouteAndRewrite`, `GradeDocuments`). Classifier model in
  `src/core/constants.py::CLASSIFIER_MODEL`.
- **Guardrails** — `src/guardrails/guardrails_wrapper.py`: OpenAI moderation API
  (raw `AsyncOpenAI`, separate `OPENAI_API_KEY`) + a homegrown LLM injection
  classifier on input; moderation on output.
- **Streaming + eval recording** — `src/api/handlers/stream.py`. Output guardrail
  runs *after* the full response is already streamed to the user.
- **Eval** — `src/core/evaluation/metrics.py` (`EvaluationTracker`) + `db.py`
  (single-row SQLite aggregate). `retrieval_precision` = grader-kept / retrieved.
- **Middleware** — `RequestLoggingMiddleware` is inlined in `src/main.py`.
- **CI** — `.github/workflows/*.yml`: ruff check + format + mypy + pytest + docker build.

Existing docs: `docs/architecture.md`, `docs/guardrails.md`, `docs/streaming.md`,
`docs/memory.md`.

---

## Key decisions (made — don't relitigate unless explicitly revisited)

1. **Retries use LangChain's built-in mechanisms, NOT tenacity.** `ChatOpenAI`
   already ships `max_retries` + backoff. A hand-rolled tenacity wrapper (as used in
   another project that calls `litellm` directly) would reinvent this. The only raw call
   is the moderation `AsyncOpenAI` call, which already fails open — leave it unless flaky.
   **No fallback model** — the app lets users pick the model in the UI, so silently
   answering with a *different* model would contradict that choice. Only same-model
   retry (`max_retries`) + structured-output retry + empty-generation retry are used.
2. **Streaming output-guardrail policy = LOG-ONLY (default).** Tokens reach the user
   before moderation can finish, so output moderation becomes a **monitoring signal,
   not a blocker**. This is the honest tradeoff Chip names (output guardrails don't work
   in stream-completion mode). *If* stronger safety is later required, switch to
   "buffer-then-flush" or "incremental batch moderation" — but that is out of scope here.
3. **`retrieval_precision` as computed today is circular** (relevant == what the
   in-pipeline grader kept). It stays as a telemetry signal but is NOT a quality metric.
   Real retrieval quality comes from the offline golden-set eval (Phase 3).
4. **No semantic cache, no separate model gateway.** Chip flags semantic cache as
   dubious; OpenRouter already is the gateway; the router node already exists.

---

## Phase 1 — Retries & fallbacks (smallest, highest robustness-per-line)

**Goal:** transient LLM/provider errors, malformed structured output, and empty
generations no longer fail a request when a retry would fix them.

**Changes**
- `src/core/llm.py::get_llm()` — add `max_retries` (e.g. 3) to `ChatOpenAI`.
  (Fallback model deliberately NOT added — see Decision #1; conflicts with UI model choice.)
- **Structured-output retry** — `graders.py`: `with_structured_output` can return/raise
  on malformed output. Add a small bounded retry (1 extra attempt) — either via
  `include_raw=True` + manual revalidation, or a thin helper. Apply to both
  `route_and_rewrite` and `grade_documents_batch`. Keep it tiny; no new dependency.
- **Empty generation retry** — `nodes.py::generate_node`: if `generation` is empty/
  whitespace, retry once before returning. Log `generation_empty_retry`.
- Add config knobs in `src/config.py` (retry count, fallback model) with sane defaults;
  don't over-parameterize.

**Verify**
- [ ] Unit test: a mocked LLM that raises a transient error once then succeeds → call succeeds.
- [ ] Unit test: structured output invalid once then valid → returns parsed model.
- [ ] Unit test: empty generation then non-empty → returns non-empty.
- [ ] `uv run mypy src` and full pytest green.

---

## Phase 2 — Guardrails refactor (correctness, not more features)

**Goal:** lower input latency, make the output policy explicit, reduce false-refusal risk.

**Changes** (`src/guardrails/guardrails_wrapper.py` + `src/api/handlers/stream.py`)
- **Parallelize input checks** — moderation and injection checks are independent;
  run them with `asyncio.gather`. Preserve fail-open behavior on exceptions.
- **Injection classifier** — DECISION: kept. Parallelizing it with moderation removes the
  latency objection, and it's a real security control for a user-facing doc agent. Revisit
  (drop or soften) only if production data shows a high false-refusal rate.
- **Output guardrail = log-only** (Decision #2). In `stream.py`, keep calling
  `check_output` but treat a flag as a logged monitoring event; do **not** rely on it to
  block (the content already streamed). Make the code/comment say this explicitly so the
  decorative-blocker confusion is gone. (If product later wants blocking, that's a
  separate buffered-streaming change.)
- Document the `OPENAI_API_KEY` (moderation) vs `OPENROUTER_API_KEY` (everything else)
  split in `docs/guardrails.md`.

**Verify**
- [ ] Existing `tests/unit/test_guardrails.py` still green; add a test that input checks
      run concurrently (or at least that both are invoked and gather is used).
- [ ] Output-flag path emits a log event and does NOT change streamed content semantics.
- [ ] `docs/guardrails.md` updated.

---

## Phase 3 — Evaluation pipeline (biggest lift; split into 3a/3b/3c)

**Goal:** real, ground-truth-based evaluation across Chip's three RAG levels, gated in CI.
Separate **offline evaluation** (golden set, quality metrics) from **online telemetry**
(latency, web-search rate). The current `EvaluationTracker` is online telemetry — keep it.

### 3a — Ranking metrics + golden dataset (do first)
- New `evals/` directory:
  - `evals/golden.jsonl` (or `.yaml`) — records: `question`, `relevant_doc_ids`
    (or filename + chunk substring to match), optional `reference_answer`,
    optional `expected_route` (`vectorstore` | `websearch`).
  - A fixed, version-controlled corpus to ingest for repeatable retrieval (small).
- Pure metric functions (trivially unit-testable) — Recall@k, Precision@k, MRR, MAP,
  NDCG@k. Put them in `src/core/evaluation/` (e.g. `ranking.py`), keep them dependency-free.
- An offline runner (`evals/run_eval.py` or `python -m ...`) that: ingests the corpus,
  runs retrieval for each golden question, computes the ranking metrics vs labels,
  prints a report, and **exits non-zero if below thresholds** (thresholds in config).

**Verify (3a)**
- [ ] Unit tests for each metric against hand-computed expected values.
- [ ] Runner produces a report on the golden set and respects thresholds (exit code).

### 3b — Generation quality (LLM-as-judge)
- Faithfulness/groundedness (is the answer supported by retrieved context?) and
  answer-relevance, computed via an LLM judge (reuse `get_llm`, a cheap model).
- Add to the runner's report; optionally gate (faithfulness threshold) once stable.

**Verify (3b)**
- [ ] Judge functions return structured scores; unit test with mocked judge LLM.
- [ ] Runner includes generation scores in the report.

### 3c — Embedding regression guard + CI gate + per-query telemetry
- **Embedding guard** — a regression test: for each golden question, the labeled
  relevant chunk must out-rank a random/irrelevant chunk by embedding similarity;
  fail if the margin collapses (catches an embedding-model swap regression).
- **CI gate** — extend `.github/workflows` with an eval job running a small,
  deterministic subset on PRs (cost/latency aware); full set nightly (schedule).
- **Per-query telemetry** — evolve `EvaluationTracker`/`db.py` from a single aggregate
  row to per-query rows (request_id, route, k, latency, web_search) so metrics can be
  broken down by axis (Chip: log everything, break down by user/release/version/time).
  Keep `get_stats()` working (aggregate over rows). This is a schema change to
  `src/core/evaluation/db.py` — migrate carefully (there's already a `_MIGRATE` pattern).

**Verify (3c)**
- [ ] Embedding guard test passes on current model, fails if a relevant/irrelevant pair
      is inverted (simulate in test).
- [ ] CI job runs the small eval subset and can fail the build below threshold.
- [ ] `/api/evaluation/stats` still returns sensible aggregates after schema change.

---

## Phase 4 — Move logging middleware out of main.py (quick cleanup, do anytime)

**Goal:** `main.py` is pure app wiring.

**Changes**
- Move `RequestLoggingMiddleware` from `src/main.py` to `src/api/middleware.py`.
- Import and `app.add_middleware(...)` in `main.py`. Nothing else changes.

**Verify**
- [ ] App boots; `tests/integration/test_api.py` green; request_id header still set.

---

## Suggested order

1. **Phase 4** (5-min cleanup, unblocks nothing but reduces noise) — optional first.
2. **Phase 1** (retries) — biggest robustness for least code.
3. **Phase 2** (guardrails) — correctness + latency.
4. **Phase 3a → 3b → 3c** (eval) — largest; land incrementally, each slice shippable.

Explicitly **out of scope** (do not build now): semantic/exact caching, a separate
model gateway, LangSmith/OTel tracing. Revisit only if a measured need appears.

---

## Status log

> Append one line per session: date — what landed — what's next.

- 2026-06-04 — Roadmap created. Nothing implemented yet. Next: Phase 1 (or 4).
- 2026-06-04 — Middleware extracted (`src/api/middleware.py`). Retries landed: ChatOpenAI
  `max_retries` (fallback model dropped, see Decision #1), structured-output retry, empty-gen
  retry. Guardrails: input checks parallelized, injection classifier kept, output check made
  explicitly best-effort. Next: Phase 3a (golden set + ranking metrics).
