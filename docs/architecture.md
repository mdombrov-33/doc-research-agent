# Architecture

This is the canonical reference for how the Document Research Agent works. Open it to
re-understand any part of the system: ingestion, retrieval, the LangGraph agent and its
reducers, grading, fallbacks, streaming, guardrails, memory, evaluation, and deployment.

File paths are given throughout so you can jump from a concept to the code.

---

## 1. What it is

An **adaptive agentic RAG** service. You upload documents; you ask questions. For each
question the agent:

1. screens the input for safety,
2. decides whether the answer needs a live web search **in addition to** your documents,
3. retrieves candidate chunks with **hybrid search**, then re-orders them with a
   **cross-encoder reranker**,
4. **grades** each candidate for relevance, and if too little is relevant, **falls back**
   to web search,
5. **generates** an answer grounded in what survived grading, streaming it token by token,
6. screens the output, and records telemetry.

It is "adaptive" because the path through the graph changes per query: web search can run
*proactively* (in parallel with retrieval) or *reactively* (as a fallback when retrieval
comes up short). Conversations are multi-turn — history is kept per session.

**Two LLM providers, by role:**
- **OpenRouter** — every chat/LLM call (routing, grading, generation, injection check, eval
  judges). Lets the UI swap models freely. See `src/core/llm.py`.
- **OpenAI (direct)** — embeddings (`text-embedding-3-small`) and the Moderation API. See
  `src/core/vector_store.py:38` and `src/guardrails/guardrails_wrapper.py:24`.

---

## 2. Request lifecycle (the big picture)

```
POST /api/stream  (QueryRequest: question, session_id, model?, top_k)
      │
      ▼
Guardrails — INPUT check        moderation API ‖ injection classifier  (parallel)
      │  (refuse here → never enters the graph)
      ▼
┌──────────────────────────── LangGraph agent ────────────────────────────┐
│  router → [retrieve (+ websearch?)] → grade_documents ⇄ websearch → generate │
└──────────────────────────────────────────────────────────────────────────┘
      │  generate's tokens stream out as they are produced (SSE)
      ▼
Guardrails — OUTPUT check       moderation API (best-effort, post-stream)
      │
      ▼
MetricsTracker.record(...)      latency, docs retrieved/relevant, web-search rate
      │
      ▼
SSE stream closes  ({"done": true, "sources": [...]})
```

Input guardrails are a **hard gate** (they run before the graph). Output guardrails are a
**soft signal** — by the time they run, tokens have already streamed to the client (see §10).

---

## 3. Process startup & dependency injection

Expensive, process-wide resources are built **once** at startup and stashed on
`app.state`; request handlers read them back through tiny provider functions. This keeps
per-request work cheap and makes everything trivially mockable in tests.

- `src/main.py` — the `lifespan` context manager runs on boot:
  - `ensure_collection_exists()` — create the Qdrant collection + indexes if missing.
  - `app.state.vector_store` — the hybrid `QdrantVectorStore`.
  - `app.state.nlp` — the loaded spaCy model (`en_core_web_sm`).
  - `app.state.agent` — the compiled LangGraph app.
  - `app.state.guardrails` — the `GuardrailsWrapper`.
  - `app.state.metrics_tracker` — the online telemetry tracker (loads prior totals from SQLite).
- `src/api/dependencies.py` — `get_vector_store`, `get_nlp`, `get_agent`, `get_guardrails`,
  `get_metrics_tracker` just return the corresponding `app.state` object via FastAPI `Depends`.
- `src/api/routes.py` — three routes wire those dependencies into the handlers.

Settings and singletons are cached: `get_settings()` is `@lru_cache`'d (`src/config.py`), as
are `get_qdrant_client`, `get_vector_store`, the spaCy model, and the cross-encoder loader.

---

## 4. Ingestion pipeline (`POST /api/upload`)

`src/api/handlers/upload.py` → `src/core/document_processing/`.

```
upload file (pdf / docx / txt)
   │  written to a temp path under UPLOAD_DIR, deleted in a finally-block
   ▼
extract text            TextExtractor  (text_processor.py)
   │   .pdf → PyMuPDF per page   .docx → python-docx paragraphs   .txt → aiofiles
   ▼
chunk                   RecursiveCharacterTextSplitter  (document_processor.py:_chunk_text)
   │   chunk_size=1200 chars, chunk_overlap=240, split on ¶ → line → sentence → … 
   │   chunks shorter than 100 chars are dropped
   ▼
enrich each chunk       spaCy NER + keywords  (document_processor.py:_enrich_chunks)
   │   entities  = ent.text         (first 10)
   │   entity_types = ent.label_    (first 10)
   │   keywords  = NOUN/PROPN, non-stopword, len>2, deduped (first 15)
   ▼
store in Qdrant         vector_store.add_documents(...)
       computes BOTH vectors per chunk:
         • dense  = OpenAI text-embedding-3-small (1536-d)
         • sparse = BM25 (FastEmbed "Qdrant/bm25")
       payload = {document_id, filename, chunk_index, chunk_length,
                  entities, entity_types, keywords, file_extension}
```

Why enrich with entities? So retrieval can **filter** on them (see §7, Retrieve). The same
spaCy NER runs at ingestion and at query time, so a query's entities match the stored ones.

---

## 5. The vector store & hybrid search

`src/core/vector_store.py` (collection setup) and `src/core/retrieval/search.py` (the store).

The Qdrant collection holds **two vectors per chunk**:

| Vector | What | Config |
|---|---|---|
| Dense | semantic meaning | `text-embedding-3-small`, 1536-d, **cosine** distance |
| Sparse | exact terms (BM25) | named `langchain-sparse`, IDF computed **server-side** (`Modifier.IDF`) |

Plus a **payload index** on `metadata.entities` (KEYWORD) so entity filtering is fast.

**Why hybrid?** Dense (bi-encoder) search matches meaning even when words differ, but is weak
on exact tokens — names, codes, rare jargon. BM25 is the opposite. Running both and merging
recovers a class of misses neither catches alone.

**How they merge — Reciprocal Rank Fusion (RRF).** Qdrant fuses the two ranked lists by rank
position, not raw score: a chunk scores `Σ 1/(k + rank_i)` across the lists it appears in.
RRF needs no score normalization or weight tuning, which is what makes mixing cosine and BM25
scales robust. `RetrievalMode.HYBRID` in `search.py` turns this on; it's a single Qdrant query.

> Bi-encoder vs cross-encoder: hybrid search is still **bi-encoder** — query and chunk are
> embedded separately and compared. That is fast but approximate. The reranker (§9) is a
> **cross-encoder** that reads query and chunk *together* for a sharper relevance judgment.
> We use the fast one to shortlist and the sharp one to re-order the shortlist.

---

## 6. The LangGraph agent — state, reducers, topology

`src/core/agent.py` builds the graph; `src/core/state.py` defines the shared state;
`src/core/nodes.py` holds the node functions.

### 6.1 The graph

```
                         START
                           │
                           ▼
                        router            (classify + rewrite query)
                  web_search? ┌─────────────┐
            ┌─────────────────┤             │
            ▼ (always)        ▼ (if needed) │
        retrieve           websearch        │   ← proactive parallel branch
            │                  │             │
            └────────┬─────────┘             │
                     ▼   (fan-in: raw_documents merged by reducer)
              grade_documents ───────────────┘
                     │   <2 relevant AND web not yet tried?
                     │        │ yes → loop back to websearch  (reactive fallback)
                     │ no
                     ▼
                  generate          (stream answer; update chat_history)
                     │
                     ▼
                    END
```

Edges (`agent.py`):
- entry: `router`
- `router` → **conditional**: `["retrieve", "websearch"]` if the router set `web_search`,
  else `["retrieve"]`. Returning a *list* fans out to parallel branches.
- `retrieve` → `grade_documents`; `websearch` → `grade_documents`
- `grade_documents` → **conditional**: `"websearch"` if `web_fallback_needed and not
  web_search_done`, else `"generate"`
- `generate` → `END`
- Compiled with a `MemorySaver` checkpointer (multi-turn memory, §11).

Each node is wrapped by `timed(...)` (`src/utils/node_timer.py`) which logs a `node_complete`
event with `duration_ms`.

### 6.2 State and the custom reducers (the tricky part)

`AgentState` is a `TypedDict`. Most fields are plain (last write wins), but two carry
**reducers** — functions that decide how a node's output is merged into the channel:

```python
raw_documents:        Annotated[list[dict], _add_or_reset_list]
docs_retrieved_total: Annotated[int,        _add_or_reset_int]
```

```python
def _add_or_reset_list(left, right):      # right is the node's returned value
    if right is None: return []           # explicit reset
    return left + right                   # otherwise append/merge

def _add_or_reset_int(left, right):
    if right is None: return 0
    return left + right
```

Why these exist — two jobs in one channel:

1. **Parallel fan-in merge.** When `retrieve` and `websearch` run in the same superstep, both
   write `raw_documents`. Without a reducer LangGraph would error on the concurrent write;
   `_add_or_reset_list` concatenates them instead. Same for the counts.
2. **Per-query / per-pass reset.** The checkpointer persists state across turns *and* across
   the fallback loop. A node returning `None` for the channel resets it to empty. The router
   resets at the start of every query (so last turn's docs don't bleed in); `grade_documents`
   resets `raw_documents` before the fallback loop so the second grading pass sees only the
   fresh web docs.

The plain (no-reducer) channel `documents` is the **durable** graded result; `raw_documents`
is **scratch** that gets reset between passes. That split is what makes the fallback loop
safe — see the trace in §8.

### 6.3 AgentState fields

| Field | Reducer | Meaning |
|---|---|---|
| `question` | — | the query; **rewritten** by the router |
| `web_search` | — | router decision: also run web search in parallel |
| `raw_documents` | merge/reset | candidate chunks, fan-in target, reset each query/pass |
| `documents` | — | graded, relevant docs passed to `generate` |
| `docs_retrieved_total` | sum/reset | how many candidates were retrieved (for metrics) |
| `web_search_done` | — | guard: web search has already run (prevents infinite loop) |
| `web_fallback_needed` | — | grader signal: too few relevant docs, trigger fallback |
| `generation` | — | the answer text |
| `chat_history` | — | full turn history; injected into the generate prompt and the router's history-aware rewrite |
| `model` | — | per-query LLM override from the UI |
| `top_k` | — | per-query retrieval depth from the UI |

---

## 7. Node by node

### Router — `router_node` / `route_and_rewrite` (`graders.py`)
One **structured-output** LLM call (model = `CLASSIFIER_MODEL`, temp 0) does two things at once:
- **classify**: `vectorstore` or `websearch` (does this need live/web info?), and
- **rewrite**: turn the question into a search-optimized query (drop filler, expand
  abbreviations, keep all parts of multi-part questions).

The document store **always runs** — web search is *additive*, never a replacement. The call
is wrapped in `with_retry` (§17) to survive a malformed structured response. The node also
resets `raw_documents`/`docs_retrieved_total` so the new query starts clean.

**History-aware rewrite.** Memory (§12) and *retrieval* are two different problems. The
checkpointer gives the *generator* the conversation, but a follow-up like *"expand on that"*
or *"the third one"* carries **no search terms** on its own — so retrieval, which only sees
the query string, would fail. The fix lives entirely in this one step: `router_node` reads
`chat_history` (already in state) and passes the **last two turns** to `route_and_rewrite`,
which adds them to the prompt and instructs the model to resolve the reference into a
**standalone** query *before anything retrieves*. Key design choices:

- **Contextualize once, at the rewrite boundary.** Retrieval stays a pure function of a query
  string; history never leaks into retrieve/grade. One place to reason about, easy to test.
- **Bounded** to the last two turns (`_format_history`) — older turns add tokens and pull the
  rewrite off-topic.
- **Conditional**: a question that already stands alone passes through unchanged; the
  history block and the rewrite instruction are only added when `chat_history` is non-empty.
- **Free**: this enriches the router call we already make every turn — no extra round-trip.

The before/after query is visible in the `route_decision` log (`query=`). The current eval
can't catch regressions here (golden questions are all standalone) — validating it properly
would mean adding **multi-turn golden cases**.

### Retrieve — `retrieve_node` (`nodes.py`)
1. **Extract query entities** with spaCy and build an **entity filter** (`_entity_filter`):
   restrict to chunks whose stored `entities` overlap the query's. This sharpens precision
   when a query names something specific.
2. **Compute `fetch_k`** (`_fetch_k`): the candidate pool to pull *before* reranking. With
   reranking on, `fetch_k = min(top_k × RERANK_MULTIPLIER, RERANK_FETCH_CAP)` (default
   `top_k×4`, capped 100); with it off, `fetch_k = top_k`.
3. **Hybrid search** for `fetch_k` candidates. If the entity filter matched nothing,
   **fall back** to an unfiltered hybrid search (`entity_fallback`).
4. Map results to dicts (`content`, `filename`, `chunk_index`, `chunk_length`,
   `source="vectorstore"`), skipping blanks. Log `docs_retrieved` with score stats.
5. **Rerank** the pool with the cross-encoder and keep `top_k` (§9). Reranking off → just
   trim to `top_k`.

### Web Search — `web_search_node` (`nodes.py`)
Calls DuckDuckGo (`get_web_search_tool`) with the rewritten query; wraps the result as a
single `source="web"` doc. **Fails soft**: on any error it logs and returns an empty list, so
a search outage never breaks the request. Sets `web_search_done=True`.

### Grade Documents — `grade_documents_node` (`nodes.py`)
- No candidates → return `documents: []` (short-circuit).
- Otherwise **batch** structured grading (`grade_documents_batch`): one `yes/no` relevance
  verdict per doc, all in one batched call (`structured_llm.batch`, `with_retry`).
- Three outcomes:
  1. **Fallback**: `< 2` relevant **and** web search not yet done → return the few relevant
     docs into `documents`, **reset `raw_documents`**, set `web_fallback_needed=True`. The
     conditional edge then routes to `websearch`.
  2. **Post-fallback merge**: if `web_search_done` is already true → **merge** the newly
     graded web docs with the previously graded `documents` and clear the fallback flag.
  3. **Normal**: enough relevant docs → pass them straight to `generate`.

### Generate — `generate_node` (`nodes.py`)
- Builds context by labeling each doc: `[Document: <filename>]` or `[Web Search]`, joined.
- Calls the LLM (model = per-query `model` override or default `LLM_MODEL`, **temp 0.7**),
  with the system prompt (context), `chat_history`, then the user question.
- **Empty-generation retry**: if the model returns whitespace, it retries once.
- Appends the user+assistant turn to `chat_history` and returns it (persisted for next turn).
- Its tokens are what stream to the client (§10).

---

## 8. Walkthroughs: the two web-search paths

**Proactive (parallel).** Router decides the query needs current info →
`web_search=True` → fan-out to `retrieve` **and** `websearch` in one superstep. Both write
`raw_documents`; the reducer merges them. `grade_documents` grades the combined set once and
goes to `generate`.

**Reactive (fallback loop).** Router routes to documents only.
1. `router` resets `raw_documents` → `[]`.
2. `retrieve` → `raw_documents = [doc-chunks]`.
3. `grade_documents`: fewer than 2 pass → saves them in `documents`, **resets**
   `raw_documents` → `[]`, sets `web_fallback_needed=True`.
4. Conditional edge → `websearch` → `raw_documents = [web]`, `web_search_done=True`.
5. Back to `grade_documents`: now grades only the **web** docs (raw was reset), then
   **merges** them with the earlier `documents`. `web_fallback_needed=False`.
6. Edge → `generate`.

The reset-vs-durable split (`raw_documents` scratch, `documents` durable) is exactly why the
second grading pass doesn't re-grade the vector docs and nothing is double-counted.

---

## 9. Reranking (cross-encoder)

`src/core/retrieval/reranker.py`. Retrieval is tuned for recall (cast a wide net); the
reranker is tuned for precision (sharpen the order).

- **Model**: `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed's `TextCrossEncoder` (ONNX, CPU,
  no torch). Loaded once and cached (`@lru_cache`).
- **Flow**: `retrieve` over-fetches `fetch_k` candidates → `rerank(query, docs, top_k)` scores
  each `(query, chunk)` pair *together* → sort by score → keep `top_k`.
- **`top_k` is the UI knob** (how many docs come back); **`fetch_k` is internal** (the pool to
  choose from, a multiple of `top_k`). The user always gets exactly `top_k`, just better-chosen.
- **Logging**: `documents_reranked` reports `candidates`, `returned`, and **`promoted`** — how
  many returned docs ranked *below* `top_k` in the raw hybrid order, i.e. hits reranking
  rescued. `promoted > 0` proves it changed the outcome.
- **Toggle**: `RERANK_ENABLED=false` reverts to raw hybrid ordering with no model load.
- **Deployment**: the model is **baked into the Docker image** at build time
  (`FASTEMBED_CACHE_PATH=/app/.model_cache`) so the first request doesn't pay a ~9s download.
  See §19.

---

## 10. Streaming (SSE)

`src/api/handlers/stream.py`. `POST /api/stream` returns `text/event-stream`.

The handler drives the graph with `agent.astream_events(..., version="v2")` and filters two
event kinds:
- `on_chat_model_stream` **where `metadata.langgraph_node == "generate"`** → forwards the
  token. This filter is important: the router and grader also call the LLM and emit token
  events, but only the *generate* node's tokens should reach the user.
- `on_chain_end` with `name == "LangGraph"` → the final graph state, used to extract
  `sources`/`sources_count`, whether web search fired, and `docs_retrieved_total`.

Event shapes the client sees:
```
data: {"token": "partial text"}                                  # during generation
data: {"done": true, "sources_count": N, "sources": [...], "session_id": "..."}  # success
data: {"token": "...", "done": true, "correction": true}         # output flagged (see §11)
data: {"error": "...", "done": true}                             # on error
```
Headers disable proxy buffering (`X-Accel-Buffering: no`, `Cache-Control: no-cache`). After
the stream the handler runs the output guardrail and records metrics.

---

## 11. Guardrails

`src/guardrails/guardrails_wrapper.py`.

- **Input (hard gate, before the graph).** Runs two checks concurrently with
  `asyncio.gather`: OpenAI **Moderation API** (harmful content) and an **injection classifier**
  (`CLASSIFIER_MODEL` prompted to spot jailbreak/probing). If either fires, the request is
  refused immediately and the graph never runs.
- **Output (soft signal, after the stream).** Moderation only. Because tokens have **already
  streamed** to the client, this cannot hard-block; it emits a `correction` event and serves
  as a monitoring signal — not a guarantee.
- **Fail-open.** Both checks swallow their own errors and return "not flagged", so a guardrails
  outage degrades safety but never takes the agent down.

---

## 12. Memory & multi-turn

The graph is compiled with a `MemorySaver` checkpointer (`agent.py`). State is keyed by
`thread_id`, which the stream handler sets to the request's `session_id`
(`config={"configurable": {"thread_id": session_id}}`). Across turns with the same
`session_id`, `chat_history` accumulates and is injected into the generate prompt. The router
resets the per-query scratch channels each turn so only `chat_history` carries over.

That history feeds **two** consumers, and it's worth keeping them distinct:
- **Conversational memory** — the *generator* sees the full `chat_history`, so it can answer
  *"expand on that"* coherently.
- **Conversational retrieval** — the *router* sees the last two turns and rewrites a
  context-dependent follow-up into a standalone search query, so retrieval finds the right
  documents even when the question names nothing on its own (the history-aware rewrite, §7).

> `MemorySaver` is **in-process**: history lives in memory and is lost on restart, and isn't
> shared across instances. Fine for a single-instance app; swap for a persistent checkpointer
> if you need durability or horizontal scale.

---

## 13. Models, providers & configuration

Five model roles:

| Role | Setting / value | Provider | Temp | Where |
|---|---|---|---|---|
| Generation | `LLM_MODEL` (UI-overridable) | OpenRouter | 0.7 | `generate_node` |
| Classifier | `CLASSIFIER_MODEL` | OpenRouter | 0 | router, grader, injection check |
| Embeddings | `EMBEDDING_MODEL` (`text-embedding-3-small`) | OpenAI | — | ingestion + retrieval |
| Reranker | `RERANK_MODEL` (MiniLM cross-encoder) | local ONNX | — | `retrieve_node` |
| Judge | `JUDGE_MODEL` (eval only) | OpenRouter | 0 | `evals/judges.py` |

**Config vs constants** (the project's rule):
- `src/config.py` (`Settings`, pydantic-settings) — anything the running app reads that can
  vary by environment. **Every field is overridable by an env var of the same name**; `.env`
  only needs the per-environment values (secrets, deployment URLs, toggles). Unset fields use
  their code default.
- `src/constants.py` — genuine code-fixed invariants only (currently just
  `SPARSE_VECTOR_NAME`, which must match what `langchain-qdrant` expects). Single-use constants
  are scoped to their file (e.g. the OpenRouter base URL in `llm.py`).
- Eval-only values (thresholds, `K`, `JUDGE_MODEL`, judge prompts) live in `evals/`, never in
  the app package.

Key settings: `RERANK_*` (§9), `QDRANT_MODE` (`local`|`cloud`, picks the URL/credentials),
`LLM_MAX_RETRIES` (§17), `METRICS_DB_PATH` (§14). See `.env.example` for the full annotated list.

---

## 14. Monitoring — online telemetry

`src/core/monitoring/tracker.py` + `db.py`. This is **live production telemetry**, distinct
from offline evaluation (§15).

After each query the stream handler calls `MetricsTracker.record(QueryMetrics(...))`. The
tracker keeps running totals (thread-safe via a `Lock`) and **persists them to SQLite**
(`METRICS_DB_PATH`), reloading on startup so stats survive restarts. `GET /api/monitoring/stats`
returns the aggregates, which the Streamlit UI shows:

| Stat | Meaning |
|---|---|
| `total_queries` | queries served |
| `web_search_rate` | fraction of queries that triggered web search |
| `avg_docs_retrieved` | mean candidates retrieved per query |
| `avg_docs_relevant` | mean docs passing the grader |
| `avg_retrieval_precision` | relevant ÷ retrieved (corpus-wide) |
| `avg_latency_ms` | mean end-to-end latency |

---

## 15. Evaluation — offline quality gate

`evals/` (repo root, **not** part of the shipped app). Full details in `evals/README.md`;
the summary:

It runs the **real pipeline** (`retrieve_node`, `generate_node`) against a fixed labelled
**golden set** (`evals/golden.jsonl`) over a fixed **corpus** (`evals/corpus/`, ingested into
an isolated `documents_eval` Qdrant collection). It scores three independent levels:

| Level | Question | Metrics | File |
|---|---|---|---|
| 1. Retrieval | did we fetch the right docs? | recall@k, precision@k, MRR, MAP, nDCG | `ranking.py` |
| 2. Generation | grounded + on-topic answer? | faithfulness, answer-relevance (LLM judge) | `judges.py` |
| 3. Embeddings | does the embedder still separate? | cosine separation guard | `embeddings_check.py` |

Each aggregate is compared to a threshold; below it, the run exits non-zero. **Two tiers:**
- **Default** (`make eval-retrieval`, run in CI on push to main): retrieval + embeddings only.
  Deterministic, cheap (no generation, no judges) — needs only `OPENAI_API_KEY` for embeddings.
- **`--full`** (`make eval`, local): adds generation + LLM judges. Expensive and noisy, so it
  never gates CI.

Because the metrics are scored at the **document (filename) level** on a small, distinct
corpus, the gate today catches *catastrophic* retrieval regressions, not subtle drift. It is a
regression tripwire — the value is comparing today's numbers to tomorrow's after a change.

This is the offline counterpart to §14: monitoring measures live traffic; evals measure
quality against known-correct answers.

---

## 16. API reference

| Endpoint | Purpose | Request | Response |
|---|---|---|---|
| `POST /api/stream` | RAG query, streamed | `{question, session_id?, model?, top_k?}` (`top_k` 1–20, default 5) | `text/event-stream` (§10) |
| `POST /api/upload` | ingest a document | multipart file (`.pdf`/`.docx`/`.txt`) | `{document_id, filename, chunks_created, file_size}` |
| `GET /api/monitoring/stats` | live telemetry | — | aggregates (§14) |
| `GET /health` | liveness | — | `{status, environment, llm_model}` |

Schemas: `src/api/schemas.py`. Every request gets an `x-request-id` (header + bound into
every log line) via `RequestLoggingMiddleware` (`src/api/middleware.py`).

---

## 17. Error handling & resilience

| Failure | Handling | Where |
|---|---|---|
| Transient LLM API errors | `ChatOpenAI(max_retries=LLM_MAX_RETRIES)` | `llm.py` |
| Malformed structured output | `with_retry` (bounded retry on parse/validation) | `utils/retry.py`, used in `graders.py` |
| Empty generation | one retry | `generate_node` |
| Web search outage | fail soft → empty docs, request continues | `web_search_node` |
| Guardrails outage | fail open → treat as not flagged | `guardrails_wrapper.py` |
| Entity filter over-narrows | fall back to unfiltered hybrid search | `retrieve_node` |
| Empty / unsupported upload | typed exceptions → 400/500, temp file always cleaned up | `handlers/upload.py` |

---

## 18. Project layout

```
src/
  api/            delivery layer: routes, handlers (stream/upload), middleware, DI, schemas
  core/
    agent.py      builds + compiles the LangGraph
    state.py      AgentState + the custom reducers
    nodes.py      the five node functions
    grading/      router + document grader (structured-output LLM calls)
    retrieval/    hybrid search store + cross-encoder reranker
    document_processing/  text extraction, chunking, spaCy enrichment
    monitoring/   online telemetry (tracker + SQLite)
    llm.py, tools.py, exceptions.py
  guardrails/     input/output safety wrapper
  utils/          logger, node timer, retry
  config.py       Settings (env-overridable)   constants.py  (true constants)   prompts.py
  main.py         FastAPI app + lifespan wiring
evals/            offline evaluation (golden set, corpus, metrics) — NOT shipped
tests/            unit + integration
ui.py             Streamlit frontend
```

Principle: the **app package stays free of eval-only concerns**; `evals/` and `tests/` are dev
tooling (excluded from the Docker image via `.dockerignore`).

---

## 19. Deployment notes

- **Docker**: multi-stage build (`Dockerfile`). The runtime stage bakes the reranker model
  into the image (`RUN ... _get_cross_encoder(...)` into `FASTEMBED_CACHE_PATH=/app/.model_cache`).
- **Why not `/tmp`**: fastembed's default cache is `/tmp/fastembed_cache`, but Cloud Run mounts
  a fresh in-memory tmpfs over `/tmp` per instance — a model baked there would vanish at
  runtime. The persistent `/app/.model_cache` path is the fix; baking it means cold starts load
  from disk (~0.7s) instead of downloading (~9s).
- **Target**: GCP Cloud Run (Terraform in `terraform/gcp`, 1 vCPU / 2Gi). spaCy + onnxruntime
  + the reranker fit in 2Gi; watch for OOM if you scale the corpus or models.
- **CI**: `.github/workflows/ci.yml` (ruff, format, mypy, pytest, docker build) and
  `eval.yml` (retrieval eval gate on push to main; needs `OPENAI_API_KEY`).
- **Config**: see `.env.example`. Secrets (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, Qdrant
  cloud creds) and deployment values go in the environment; everything else has a default.
```
make up            # boot Qdrant + API via docker compose
make dev           # run the API locally (uvicorn --reload)
make ui            # run the Streamlit UI
make test          # pytest
make eval          # full offline eval (local)
make eval-retrieval # retrieval+embeddings eval (what CI runs)
```
