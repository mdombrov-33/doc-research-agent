# Architecture

This is the canonical reference for how the Document Research Agent works. Open it to
re-understand any part of the system: ingestion, retrieval, the LangGraph agent and its
tools, grading, the web-search fallback, streaming, guardrails, memory, evaluation, and
deployment.

File paths are given throughout so you can jump from a concept to the code.

---

## 1. What it is

An **agentic RAG** service. You upload documents; you ask questions. For each question the
agent:

1. screens the input for safety,
2. **decides for itself** which tool to use — search your documents, or search the live web,
3. retrieves candidate chunks with **hybrid search**, then re-orders them with a
   **cross-encoder reranker**,
4. **grades** each retrieved chunk for relevance and drops the rest; an empty result nudges
   the agent to try a different tool,
5. loops — reason, act, observe — until it has enough context, then **writes the answer**,
   streaming it token by token,
6. records telemetry.

It is "agentic" because the path through the graph is **not fixed**: a tool-calling LLM picks
its own trajectory. A question answerable from your files retrieves once and answers. A
question the files don't cover retrieves, sees nothing relevant, and **falls back to web
search** on its own. This is the **ReAct** loop (reason → act → observe) with a **corrective
grading** step bolted on (the CRAG idea) — see §6. Conversations are multi-turn; history is
persisted per session (§12).

**Two LLM providers, by role:**
- **OpenRouter** — every chat/LLM call (the agent's reasoning + answer, document grading, and
  the eval judges). Lets the UI swap models freely. See `src/core/llm.py`.
- **OpenAI (direct)** — embeddings only (`text-embedding-3-small`), for ingestion and
  retrieval. See `src/core/vectorstore.py`.

Safety screening uses **neither** provider: it runs **local** HuggingFace models via
`llm-guard` (§11).

---

## 2. Request lifecycle (the big picture)

```
POST /api/stream  (QueryRequest: question, session_id?, model?, top_k?)
      │
      ▼
Guardrails — INPUT check        llm-guard scanners (Toxicity + PromptInjection), local
      │  (flagged → refusal, the graph never runs)
      ▼
┌──────────────────── LangGraph agent (ReAct + corrective grading) ────────────────────┐
│                                                                                        │
│        agent ──tools_condition──► tools ──► grade ──┐                                  │
│          ▲                                          │                                  │
│          └──────────────────────────────────────────┘   loop until no tool calls      │
│          │                                                                             │
│          └─tools_condition─► END  (the agent wrote an answer instead of calling a tool)│
└────────────────────────────────────────────────────────────────────────────────────┘
      │  the agent's final-answer tokens stream out as they are produced (SSE)
      ▼
MetricsTracker.record(...)      latency, docs retrieved/relevant, web-search rate
      │
      ▼
SSE stream closes  ({"done": true, "sources": [...]})
```

Input guardrails are a **hard gate** (they run before the graph). There is **no output
guardrail** — once the agent starts answering, tokens stream straight to the client (§11).

---

## 3. Process startup & dependency injection

Expensive, process-wide resources are built **once** at startup and stashed on
`app.state`; request handlers read them back through tiny provider functions. This keeps
per-request work cheap and makes everything trivially mockable in tests.

- `src/main.py` — the `lifespan` context manager runs on boot:
  - `ensure_collection_exists()` — create the Qdrant collection + indexes if missing.
  - `app.state.vector_store` — the hybrid `QdrantVectorStore`.
  - `app.state.nlp` — the loaded spaCy model (`en_core_web_sm`).
  - `app.state.agent` — the compiled LangGraph app (built **here**, inside the running event
    loop, because the async SQLite checkpointer grabs the loop at construction — §12).
  - `guardrails.warmup()` — primes the llm-guard scanners so the first request doesn't pay
    their cold start (§11).
  - `rerank.warmup()` — primes the cross-encoder for the same reason (§9).
  - `app.state.metrics_tracker` — the online telemetry tracker (loads prior totals from SQLite).
- `src/api/dependencies.py` — `get_vector_store`, `get_nlp`, `get_agent`,
  `get_metrics_tracker` just return the corresponding `app.state` object via FastAPI `Depends`.
- `src/api/routes.py` — the routes wire those dependencies into the handlers.

Settings and singletons are cached: `get_settings()` is `@lru_cache`'d (`src/config.py`), as
are `get_qdrant_client`, `get_vector_store`, the spaCy model, the cross-encoder loader, and
the llm-guard scanners.

---

## 4. Ingestion pipeline (`POST /api/upload`)

`src/api/handlers/upload.py` → `src/core/ingestion/` (`pipeline.py` orchestrates the steps).

```
upload file (pdf / docx / txt)
   │  written to a temp path under UPLOAD_DIR, deleted in a finally-block
   ▼
extract text            extract.py  (extract_from_file)
   │   .pdf → PyMuPDF per page   .docx → python-docx paragraphs   .txt → aiofiles
   ▼
chunk                   chunk.py  (chunk_text → RecursiveCharacterTextSplitter)
   │   chunk_size=1200 chars, chunk_overlap=240, split on ¶ → line → sentence → … 
   │   chunks shorter than 100 chars are dropped
   ▼
enrich each chunk       enrich.py  (enrich_chunks → spaCy NER + keywords)
   │   entities     = ent.text     (first 10)
   │   entity_types = ent.label_   (first 10)
   │   keywords     = NOUN/PROPN, non-stopword, len>2, deduped (first 15)
   ▼
store in Qdrant         index.py  (index_chunks → vector_store.add_documents)
       computes BOTH vectors per chunk:
         • dense  = OpenAI text-embedding-3-small (1536-d)
         • sparse = BM25 (FastEmbed "Qdrant/bm25")
       payload = {document_id, filename, chunk_index, chunk_length,
                  entities, entity_types, keywords, file_extension}
```

Why enrich with entities? So retrieval can **filter** on them (see §5). The same spaCy NER
runs at ingestion and at query time, so a query's entities match the stored ones.

---

## 5. The vector store & hybrid search

`src/core/vectorstore.py` (collection setup + the store) and `src/core/retrieval/search.py`
(the `hybrid_search` query function).

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
scales robust. `RetrievalMode.HYBRID` turns this on; it's a single Qdrant query.

**`hybrid_search(question, top_k)`** is the whole retrieval path in one function:

1. **Extract query entities** with spaCy and build an **entity filter** (`_entity_filter`):
   restrict to chunks whose stored `entities` overlap the query's. This sharpens precision
   when a query names something specific.
2. **Compute `fetch_k`** (`_fetch_k`): the candidate pool to pull *before* reranking. With
   reranking on, `fetch_k = min(top_k × RERANK_MULTIPLIER, RERANK_FETCH_CAP)` (default
   `top_k×4`, capped 100, floored at `top_k`); with it off, `fetch_k = top_k`.
3. **Hybrid search** for `fetch_k` candidates. If the entity filter matched nothing,
   **fall back** to an unfiltered hybrid search (`entity_fallback`).
4. Map results to dicts (`content`, `filename`, `chunk_index`, `chunk_length`,
   `source="vectorstore"`), skipping blanks. Log `docs_retrieved` with score stats.
5. **Rerank** the pool with the cross-encoder and keep `top_k` (§9). Reranking off → just
   trim to `top_k`.

> Bi-encoder vs cross-encoder: hybrid search is still **bi-encoder** — query and chunk are
> embedded separately and compared. That is fast but approximate. The reranker (§9) is a
> **cross-encoder** that reads query and chunk *together* for a sharper relevance judgment.
> We use the fast one to shortlist and the sharp one to re-order the shortlist.

`hybrid_search` is called from the `retrieve_documents` **tool** (§6) during serving, and
directly from the offline eval (§15) — the retrieval layer is the same in both.

---

## 6. The LangGraph agent — state, tools, topology

`src/core/agent/graph.py` builds the graph; `src/core/agent/state.py` defines the shared
state; `src/core/agent/nodes.py` holds the two node functions; `src/core/agent/tools.py`
holds the tools.

### 6.1 The shape: ReAct + corrective grading

A tool-calling LLM (the `agent` node) drives the loop. It either **calls a tool** or **writes
the final answer**. `tools_condition` (a LangGraph prebuilt) reads the agent's last message:
if it contains tool calls, route to `tools`; otherwise the agent answered, so route to `END`.

```
                         START
                           │
                           ▼
                  ┌────►  agent  ──────────────────┐   (ReAct brain: pick a tool, or answer)
                  │        │                        │
                  │   tools_condition               │ no tool calls
                  │        │ tool calls             ▼
                  │        ▼                        END
                  │      tools     (ToolNode runs retrieve_documents / web_search)
                  │        │
                  │        ▼
                  └──────  grade    (score docs, drop irrelevant, rewrite the tool message)
```

Edges (`graph.py`):
- entry: `agent`
- `agent` → **conditional** via `tools_condition`: `tools` if the agent emitted tool calls,
  else `END`.
- `tools` → `grade`
- `grade` → `agent` (loop back so the agent reasons over the graded result)

Each custom node is wrapped by `timed(...)` (`src/utils/node_timer.py`), which logs a
`node_complete` event with `duration_ms`. (`tools` is the prebuilt `ToolNode`, not wrapped.)

The graph is compiled with a checkpointer (`AsyncSqliteSaver` in production, a `MemorySaver`
in tests) — that's what makes it multi-turn (§12).

**Why a corrective grade step at all?** Plain ReAct would feed *raw* retrieval back to the
agent. The `grade` node sits between `tools` and `agent` and filters each retrieval to only
the chunks an LLM judged relevant. If nothing survives, the agent sees *"No relevant results
found."* — which is exactly the signal that makes it reach for `web_search` next. The
corrective filter is what turns "retrieved something" into "retrieved something useful".

### 6.2 State and the reducers

`AgentState` (`state.py`) is a small `TypedDict` (`total=False`) with four channels:

```python
class AgentState(TypedDict, total=False):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    documents: Annotated[list[dict], _add_or_reset_list]
    docs_retrieved_total: Annotated[int, _add_or_reset_int]
    web_search_used: bool
```

- **`messages`** — the conversation **and** the ReAct scratchpad in one list. The
  `add_messages` reducer appends each turn (human question, agent tool-call, tool result,
  agent answer…). The checkpointer persists this per `thread_id`, so it doubles as the
  multi-turn history — there is no separate `chat_history` field. `Required[...]` marks it as
  always-present inside the otherwise-optional `TypedDict`, while keeping the `add_messages`
  metadata that LangGraph reads for the reducer.
- **`documents`** — the graded-relevant docs for the **current** query, used for source
  metadata and metrics. Carries the `_add_or_reset_list` reducer.
- **`docs_retrieved_total`** — pre-grade count, for the precision metric. Carries
  `_add_or_reset_int`.
- **`web_search_used`** — did the `web_search` tool run this query (for the web-search-rate
  metric). Plain `bool`.

The two reducers do one job: **accumulate within a query, reset between queries.**

```python
def _add_or_reset_list(left, right):   # right is the node's returned value
    if right is None: return []        # explicit reset
    return left + right                # otherwise accumulate

def _add_or_reset_int(left, right):
    if right is None: return 0
    return left + right
```

Within one query the agent may retrieve several times (e.g. retrieve, then two web searches);
each `grade` pass adds its survivors to `documents` and its count to `docs_retrieved_total`.
At the **start** of each new query the stream handler sends `documents=None` and
`docs_retrieved_total=None`, which the reducers interpret as "reset to empty" so last turn's
docs don't bleed into this one. `messages` is deliberately **not** reset — that's the history
the checkpointer carries forward.

> **Per-request knobs are not in state.** `model` and `top_k` ride in the `RunnableConfig`'s
> `configurable` dict, not in `AgentState`. The agent node reads `model`; the
> `retrieve_documents` tool reads `top_k`. Keeping invocation-scoped parameters out of the
> evolving, persisted state is what keeps the state minimal (§13, §12).

### 6.3 The tools

`src/core/agent/tools.py` defines two tools, both declared
`@tool(response_format="content_and_artifact")`. That response format is the key trick: each
tool returns a **`(string, list[dict])`** pair — the **string** becomes the `ToolMessage`
content the model reads, and the **list of doc dicts** rides along as the `ToolMessage`'s
**`artifact`**. The `grade` node pulls the artifact back out to score relevance and build
source metadata, none of which has to be re-parsed from the model-facing text.

| Tool | What it does | Notes |
|---|---|---|
| `retrieve_documents(query, config)` | hybrid search over the user's uploaded docs (§5) | `top_k` is read from the injected `config` (`configurable.top_k`), defaulting to `RETRIEVE_TOP_K=5`. `config` is injected by `ToolNode`; the model never sees it. |
| `web_search(query)` | DuckDuckGo live web search | **Fails soft**: on any error it logs `web_search_tool_failed` and returns `("Web search failed.", [])`, so a search outage never breaks the request. |

`format_docs` renders docs for the model, labelling each `[Document: <filename>]` or
`[Web Search]`. `TOOLS = [retrieve_documents, web_search]` is the list bound to the agent and
wrapped by `ToolNode`.

---

## 7. Node by node

### Agent — `agent_node` (`nodes.py`)
The ReAct brain. It binds the tools to the LLM (`get_llm(model).bind_tools(TOOLS)`), prepends
the `AGENT_SYSTEM_PROMPT` to the running `messages`, and invokes once. The result is appended
to `messages`. Two outcomes:
- the response carries **tool calls** → `tools_condition` routes to `tools`;
- the response is **plain content** → that *is* the answer; `tools_condition` routes to `END`,
  and those tokens are what stream to the user (§10).

The model is per-request: `get_llm(config.configurable.model)`, falling back to `LLM_MODEL`.
Temperature is `get_llm`'s default of **0** (the system was previously 0.7 in a dedicated
generate step; there is no separate generate step now).

The system prompt (`prompts.py:AGENT_SYSTEM_PROMPT`) is what makes the loop behave: it tells
the agent to (1) call `retrieve_documents` first for almost any question, resolving vague
follow-ups like *"expand on that"* into standalone queries; (2) call `web_search` only when
the documents are insufficient or the question needs current/external facts; (3) stop calling
tools and answer once it has enough context, using only the retrieved context.

### Tools — `ToolNode(TOOLS)` (prebuilt)
LangGraph's prebuilt `ToolNode` executes whatever tool calls the agent emitted (it can run
more than one), injects the `RunnableConfig` into tools that ask for it, and appends one
`ToolMessage` per call — content for the model, `artifact` for us (§6.3).

### Grade — `grade_documents_node` (`nodes.py`)
The corrective step. It looks only at the `ToolMessage`s produced **since the last agent turn**
(`_recent_tool_messages`), so each pass grades just the fresh retrieval. For each:
- pull the docs from `tm.artifact`; count them into `docs_retrieved_total`;
- if it was the `web_search` tool, mark `web_search_used`;
- **batch-grade** the docs against the question (`grade_documents_batch`, §13) — one `yes/no`
  per doc;
- keep the `yes` docs, accumulate them into `documents`, and **rewrite the `ToolMessage` in
  place** (same `id`, so `add_messages` replaces rather than appends) with `format_docs` of
  the survivors — or the literal `"No relevant results found."` when none survive.

That rewrite is the whole point: the agent's next turn reasons over *graded* context, and an
empty result is the explicit nudge to try another tool. `web_search_used` is only ever set to
`True`, so an earlier retrieve pass's `False` can't clobber a later web pass within one query.

---

## 8. Walkthroughs: the two retrieval paths

**Documents-only (the common case).**
1. `agent` → calls `retrieve_documents` (query derived from the question).
2. `tools` → hybrid search returns chunks as a `ToolMessage` + artifact.
3. `grade` → enough chunks pass; it rewrites the tool message to the relevant subset and adds
   them to `documents`.
4. `agent` → has what it needs, writes the answer. `tools_condition` → `END`.

**Web-search fallback (the corrective path).** *(Observed live on an off-corpus question.)*
1. `agent` → calls `retrieve_documents`.
2. `tools` → returns chunks.
3. `grade` → **zero** relevant; rewrites the tool message to *"No relevant results found."*
4. `agent` → sees nothing useful, **calls `web_search`** on its own.
5. `tools` → web results; `grade` → relevant, added to `documents`, `web_search_used=True`.
6. `agent` → answers from the web context. (It may run `web_search` more than once if it
   judges it needs more — each call is graded and accumulated.)

Nothing routes this — the **agent decides** based on the graded observation. There is no
classifier and no parallel branch; the trajectory emerges from the ReAct loop.

---

## 9. Reranking (cross-encoder)

`src/core/retrieval/rerank.py`. Retrieval is tuned for recall (cast a wide net); the reranker
is tuned for precision (sharpen the order).

- **Model**: `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed's `TextCrossEncoder` (ONNX, CPU,
  no torch in the hot path). Loaded once and cached (`@lru_cache`).
- **Flow**: `hybrid_search` over-fetches `fetch_k` candidates → `rerank(query, docs, top_k)`
  scores each `(query, chunk)` pair *together* → sort by score → keep `top_k`.
- **`top_k` is the UI knob** (how many docs come back); **`fetch_k` is internal** (the pool to
  choose from, a multiple of `top_k`). The user always gets exactly `top_k`, just better-chosen.
- **Logging**: `documents_reranked` reports `candidates`, `returned`, and **`promoted`** — how
  many returned docs ranked *below* `top_k` in the raw hybrid order, i.e. hits reranking
  rescued. `promoted > 0` proves it changed the outcome.
- **Toggle**: `RERANK_ENABLED=false` reverts to raw hybrid ordering with no model load.
- **Warmup**: `rerank.warmup()` runs at startup (§3) so the first query doesn't pay the model
  load. No-op when reranking is disabled.
- **Deployment**: the model is **baked into the Docker image** at build time
  (`FASTEMBED_CACHE_PATH=/app/.model_cache`) so the first request doesn't pay a ~9s download.
  See §19.

---

## 10. Streaming (SSE)

`src/api/handlers/stream.py`. `POST /api/stream` returns `text/event-stream`.

The handler drives the graph with `agent.astream_events(..., version="v2")` and filters two
event kinds:
- `on_chat_model_stream` **where `metadata.langgraph_node == "agent"`** → forwards the token.
  This filter matters: the grader also calls the LLM and emits token events, but only the
  *agent* node's tokens should reach the user. The agent node both *decides tools* and *writes
  the answer*; tool-deciding turns carry no content, so in the normal case only the final
  answer yields tokens here.
- `on_chain_end` with `name == "LangGraph"` → the final graph state, used to extract
  `sources`/`sources_count` (from `documents`), `web_search_used`, and `docs_retrieved_total`.

Event shapes the client sees:
```
data: {"token": "partial text"}                                              # during the answer
data: {"done": true, "sources_count": N, "sources": [...], "session_id": "..."}  # success
data: {"error": "...", "done": true}                                         # on error
```
Headers disable proxy buffering (`X-Accel-Buffering: no`, `Cache-Control: no-cache`). After
the stream closes the handler records metrics (§14).

> **Known edge — preamble tokens.** Because the agent node *can* emit content alongside a tool
> call (e.g. *"Let me search the web for you!"* right before a `web_search` call), that
> narration is on the `agent` node and therefore streams to the user ahead of the real answer.
> It's harmless (the final answer is complete and correct) but it mixes process-narration into
> the output. The system prompt can be tuned to suppress it if undesired.

---

## 11. Guardrails

`src/core/guardrails.py`. **Input only** — there is no output guardrail.

- **Input (hard gate, before the graph).** `check_input` runs the question through
  **llm-guard**'s `scan_prompt` with two **local** HuggingFace scanners: `Toxicity` (harmful
  content) and `PromptInjection` (jailbreaks / system probing). The scan runs in a thread
  executor so it doesn't block the event loop. If **either** scanner flags the text, the
  request is refused immediately (a fixed refusal string) and the graph never runs.
- **No external provider.** Unlike the rest of the app, guardrails touch neither OpenAI nor
  OpenRouter — the models run in-process. They're loaded once (`@lru_cache` on
  `_get_scanners`) and primed at startup by `guardrails.warmup()` (§3); the models are **baked
  into the Docker image** so the first request skips a ~64s download (§19).
- **No output check.** Because the agent streams its answer token by token, there is no point
  at which a full response exists to screen before the user sees it. The previous
  buffer-and-correct design was dropped along with the dedicated generate step; output safety
  would require giving up streaming, which we don't.

> Note: the input scanners are not wrapped in a try/except, so a scanner *internal* error
> propagates rather than failing open. If you need fail-open behaviour (degrade safety but
> never block on a guardrails outage), that's a deliberate change to make in `check_input`.

---

## 12. Memory & multi-turn

The graph is compiled with a **checkpointer** (`graph.py`). State is keyed by `thread_id`,
which the stream handler sets to the request's `session_id`
(`config={"configurable": {"thread_id": session_id, ...}}`). Across turns with the same
`session_id`, the `messages` list accumulates (via the `add_messages` reducer) and the agent
sees the **whole conversation** on every turn.

**`messages` *is* the memory.** There is no separate `chat_history`. Each turn the stream
handler appends only the new `HumanMessage`; everything else — prior questions, the agent's
tool calls, tool results, prior answers — is reloaded from the checkpoint. That single list
feeds both:
- **Conversational memory** — the agent answers *"expand on that"* coherently because the
  prior turns are right there in `messages`.
- **Conversational retrieval** — there's no separate router rewriting follow-ups anymore; the
  agent itself, seeing the history, is instructed by its system prompt to resolve a
  context-dependent follow-up into a standalone `retrieve_documents` query before searching.

**Persistent checkpointer.** Production uses `AsyncSqliteSaver(aiosqlite.connect(path))`
(`langgraph-checkpoint-sqlite`), where `path = settings.checkpoints_db_path` (under
`DATA_DIR`, §13). The serving path is async (`astream_events`), so the checkpointer must be
async too — a sync `SqliteSaver` raises on the async API. It self-initializes its tables on
first use and is **built inside the FastAPI lifespan** (§3) because `AsyncSqliteSaver` grabs
the running event loop at construction. Tests inject a sync `MemorySaver` via
`build_graph(checkpointer=...)`, which works with the sync `.invoke` API they drive.

> **Durability caveat.** SQLite makes history survive a process restart *when the database
> file persists* — true under docker-compose (the `./data` volume, §19) but **not** on Cloud
> Run, whose local disk is ephemeral and not shared across instances. For durable,
> horizontally-scaled memory, point the checkpointer at a shared backend (e.g. Postgres). The
> graph code already takes an injectable checkpointer, so that's a one-line swap in `graph.py`.

---

## 13. Models, providers & configuration

Model roles:

| Role | Setting / value | Provider | Temp | Where |
|---|---|---|---|---|
| Agent (reason + answer) | `LLM_MODEL` (UI-overridable) | OpenRouter | 0 | `agent_node` |
| Grader | `CLASSIFIER_MODEL` | OpenRouter | 0 | `grade_documents_batch` |
| Embeddings | `EMBEDDING_MODEL` (`text-embedding-3-small`) | OpenAI | — | ingestion + retrieval |
| Reranker | `RERANK_MODEL` (MiniLM cross-encoder) | local ONNX | — | `rerank` |
| Guardrails | Toxicity + PromptInjection | local (HF, via llm-guard) | — | `guardrails` |
| Judge | `JUDGE_MODEL` (eval only) | OpenRouter | 0 | `evals/judges.py` |

**Config vs constants** (the project's rule):
- `src/config.py` (`Settings`, pydantic-settings) — anything the running app reads that can
  vary by environment. **Every field is overridable by an env var of the same name**; `.env`
  only needs the per-environment values (secrets, deployment URLs, toggles). Unset fields use
  their code default.
- Genuine code-fixed invariants live next to their use (e.g. `SPARSE_VECTOR_NAME` in
  `vectorstore.py`, `RETRIEVE_TOP_K` in `tools.py`, the OpenRouter base URL in `llm.py`).
- Eval-only values (thresholds, `K`, `JUDGE_MODEL`, judge prompts) live in `evals/`, never in
  the app package.

**`DATA_DIR`.** A single `DATA_DIR` (default `./data`) is the home for runtime SQLite files;
two derived properties hang off it — `metrics_db_path` (`monitoring`, §14) and
`checkpoints_db_path` (the conversation checkpointer, §12). Mounting one directory persists
both.

Other key settings: `RERANK_*` (§9), `QDRANT_MODE` (`local`|`cloud`, picks the URL/credentials),
`LLM_MAX_RETRIES` (§17), `RATE_LIMIT*` (§17). See `.env.example` for the full annotated list.

---

## 14. Monitoring — online telemetry

`src/core/monitoring/tracker.py` + `db.py`. This is **live production telemetry**, distinct
from offline evaluation (§15).

After each query the stream handler calls `MetricsTracker.record(QueryMetrics(...))`. The
tracker keeps running totals (thread-safe via a `Lock`) and **persists them to SQLite**
(`metrics_db_path`, under `DATA_DIR`), reloading on startup so stats survive restarts.
`GET /api/monitoring/stats` returns the aggregates, which the Streamlit UI shows:

| Stat | Meaning |
|---|---|
| `total_queries` | queries served |
| `web_search_rate` | fraction of queries where the agent used `web_search` |
| `avg_docs_retrieved` | mean candidates retrieved per query (pre-grade) |
| `avg_docs_relevant` | mean docs passing the grader |
| `avg_retrieval_precision` | relevant ÷ retrieved (corpus-wide) |
| `avg_latency_ms` | mean end-to-end latency |

---

## 15. Evaluation — offline quality gate

`evals/` (repo root, **not** part of the shipped app). Full details in `evals/README.md`;
the summary:

It scores the **real retrieval layer** (`hybrid_search`) — and, under `--full`, a generation
step driven by the eval's own `GENERATION_*` prompts — against a fixed labelled **golden set**
(`evals/golden.jsonl`) over a fixed **corpus** (`evals/corpus/`, ingested into an isolated
`documents_eval` Qdrant collection). It deliberately scores the retrieval layer **directly**,
not through the agent's tool loop, so the numbers measure retrieval quality rather than the
LLM's tool-choice behaviour. Three independent levels:

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
every log line) via `RequestLoggingMiddleware` (`src/api/middleware.py`). `/stream` and
`/upload` are rate-limited per IP and may return **429** (§17).

---

## 17. Error handling & resilience

| Failure | Handling | Where |
|---|---|---|
| Transient LLM API errors | `ChatOpenAI(max_retries=LLM_MAX_RETRIES)` | `llm.py` |
| Malformed structured output (grader) | `with_retry` (bounded retry on parse/validation) | `utils/retry.py`, used in `grading.py` |
| Web search outage | fail soft → empty docs, request continues | `web_search` tool |
| Entity filter over-narrows | fall back to unfiltered hybrid search | `search.py` |
| Empty / unsupported upload | typed exceptions → 400/500, temp file always cleaned up | `handlers/upload.py` |
| Request floods / runaway cost | per-IP rate limit (slowapi) → 429 + `Retry-After` | `api/rate_limit.py` |

### Rate limiting

The two endpoints that cost money or do real work — `/stream` (LLM + embeddings per call) and
`/upload` (parsing + embedding) — are rate-limited with **slowapi** (`api/rate_limit.py`). The
shared `Limiter` decorates those two routes (`@limiter.limit(rate_limit)`), so
`/monitoring/stats` and `/health` stay unmetered.

- **What it limits:** requests per client IP. The limit is the `RATE_LIMIT` string
  (e.g. `30/minute`); exceeding it returns **HTTP 429** with a `Retry-After` header, via the
  `_rate_limit_exceeded_handler` registered in `main.py`. A custom `key_func` reads the client
  from the first `X-Forwarded-For` entry (the real caller behind Cloud Run / a proxy), falling
  back to the socket address.
- **Where it lives:** in the *application* layer, because only the app knows what to meter
  (this endpoint vs that one). In a fuller deployment this is one of several layers — a
  CDN/WAF and the load balancer would shed coarse IP floods *before* they reach the app, and
  OpenRouter/OpenAI already rate-limit us upstream (their 429s are absorbed by `max_retries`).
- **Single-instance vs scale:** the limiter's `RATE_LIMIT_STORAGE_URI` defaults to `memory://`
  — correct on one instance, like the SQLite monitoring (§14) and conversation memory (§12).
  Cloud Run autoscales, so in-memory counters aren't shared and a client spread across N
  instances effectively gets N× the limit. The fix is **one config change**: point the URI at
  `redis://…` and slowapi shares counters across instances. Keyed on IP rather than user
  because the app has no auth. *(Chose the maintained library over a hand-rolled token bucket:
  it's the standard FastAPI tool and gives the Redis backend for free.)*

---

## 18. Project layout

```
src/
  api/            delivery layer: routes, handlers (stream/upload), middleware, DI, schemas, rate_limit
  core/
    agent/        the LangGraph agent
      graph.py    builds + compiles the graph (agent ⇄ tools → grade), owns the checkpointer
      state.py    AgentState + the reducers
      nodes.py    agent_node + grade_documents_node
      tools.py    retrieve_documents + web_search (@tool, content_and_artifact)
      grading.py  the batched yes/no document grader (structured-output LLM)
      prompts.py  agent system prompt, grader prompts, eval-only generation prompts
    retrieval/    hybrid search (search.py) + cross-encoder reranker (rerank.py)
    ingestion/    extract, chunk, enrich (spaCy), index, pipeline
    monitoring/   online telemetry (tracker + SQLite)
    guardrails.py llm-guard input scanners
    vectorstore.py  Qdrant setup + the hybrid store
    nlp.py, llm.py, exceptions.py
  utils/          logger, node timer, retry
  config.py       Settings (env-overridable)
  main.py         FastAPI app + lifespan wiring
evals/            offline evaluation (golden set, corpus, metrics) — NOT shipped
tests/            unit + integration
ui.py             Streamlit frontend
```

Principle: the **app package stays free of eval-only concerns**; `evals/` and `tests/` are dev
tooling (excluded from the Docker image via `.dockerignore`).

---

## 19. Deployment notes

- **Docker**: multi-stage build (`Dockerfile`). The runtime stage bakes **two** sets of models
  into the image so cold starts don't download them:
  - the reranker cross-encoder into `FASTEMBED_CACHE_PATH=/app/.model_cache` (skips ~9s);
  - llm-guard's Toxicity + PromptInjection models into `HF_HOME=/app/.hf_cache` (skips ~64s).
  Both bakes run *before* `COPY . .` and inline the model names (not imported from `src`) so the
  layers — and the downloads — stay cached across code changes. Keep the names in sync with
  `config.RERANK_MODEL` and `guardrails.py`.
- **Why not `/tmp`**: fastembed's default cache is `/tmp/fastembed_cache`, but Cloud Run mounts
  a fresh in-memory tmpfs over `/tmp` per instance — a model baked there would vanish at
  runtime. The persistent `/app/.model_cache` and `/app/.hf_cache` paths are the fix.
- **Runtime state**: `DATA_DIR` (`/app/data`) holds `metrics.db` and `checkpoints.db`.
  docker-compose mounts `./data:/app/data` so both persist locally; on Cloud Run the disk is
  ephemeral (see the §12 caveat).
- **Target**: GCP Cloud Run (Terraform in `terraform/gcp`). spaCy + onnxruntime + the
  reranker + llm-guard's models must fit in the instance memory; watch for OOM if you scale the
  corpus or models.
- **CI**: `.github/workflows/ci.yml` (ruff, format, mypy, pytest, docker build) and
  `eval.yml` (retrieval eval gate on push to main; needs `OPENAI_API_KEY`).
- **Config**: see `.env.example`. Secrets (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, Qdrant
  cloud creds) and deployment values go in the environment; everything else has a default.

```
make up             # boot Qdrant + API via docker compose
make dev            # run the API locally (uvicorn --reload)
make ui             # run the Streamlit UI
make test           # pytest
make eval           # full offline eval (local)
make eval-retrieval # retrieval+embeddings eval (what CI runs)
```
