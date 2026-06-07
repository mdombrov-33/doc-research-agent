# Memory

How the agent remembers what was said earlier in a conversation.

---

## The short version

Each browser session gets a unique `session_id`. That ID is passed to LangGraph as a
`thread_id`, which tells it which thread's state to load and save. The conversation lives in
the agent state's `messages` list, and LangGraph's checkpointer persists that list per thread.
There is no separate "chat history" field — `messages` *is* the history.

---

## `messages` is the memory

The agent state has one channel that carries the conversation (`src/core/agent/state.py`):

```python
class AgentState(TypedDict, total=False):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    ...
```

The `add_messages` reducer appends to this list every turn — the human question, the agent's
tool calls, the tool results, the agent's answer. Because the checkpointer persists the whole
list per `thread_id`, the next turn reloads everything that came before. The agent sees the
**entire conversation** on each turn, so it can answer a follow-up like *"expand on that"*
coherently.

Each new turn the stream handler appends **only** the new question; the rest is reloaded from
the checkpoint (`src/api/handlers/stream.py`):

```python
inputs = {
    "messages": [HumanMessage(content=request.question)],
    "documents": None,            # reset per-query accumulators (see reducers)
    "docs_retrieved_total": None,
    "web_search_used": False,
}
```

> Note `documents` / `docs_retrieved_total` are set to `None` here — their reducers read that
> as "reset to empty" so last turn's retrieved docs don't bleed into this one. `messages` is
> deliberately *not* reset; that's the part that carries over.

---

## The checkpointer

When the graph is compiled it's given a checkpointer (`src/core/agent/graph.py`):

```python
checkpointer = AsyncSqliteSaver(aiosqlite.connect(settings.checkpoints_db_path))
app = workflow.compile(checkpointer=checkpointer)
```

`AsyncSqliteSaver` (from `langgraph-checkpoint-sqlite`) stores each thread's state in a SQLite
file under `DATA_DIR`. Two details matter:

- **It must be async.** The serving path streams with `astream_events`, which uses the async
  checkpointer API. A sync `SqliteSaver` raises *"does not support async methods"* there.
- **It must be built inside the running event loop.** `AsyncSqliteSaver` grabs the loop at
  construction, so the graph is built in the FastAPI **lifespan** (`src/main.py`), not at
  import time.

Tests don't want a real database, so `build_graph` takes an **injectable** checkpointer and
the tests pass a sync `MemorySaver` (they drive the graph with `.invoke`, the sync API):

```python
return build_graph(checkpointer=MemorySaver())
```

---

## How sessions are isolated

Every request carries a `session_id` (`src/api/schemas.py`):

```python
class QueryRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ...
```

If the frontend doesn't send one, a fresh UUID is generated — a new session with no history.
Send the same `session_id` across requests and LangGraph loads the matching thread and
continues. The stream handler wires it as the `thread_id`:

```python
config = {"configurable": {"thread_id": request.session_id, "model": ..., "top_k": ...}}
async for event in agent.astream_events(inputs, config=config, version="v2"):
    ...
```

That `config` is what tells LangGraph which thread's state to load. (The same `configurable`
dict also carries the per-request `model` and `top_k` knobs — those are *not* in the persisted
state.)

---

## History-aware retrieval

Memory and retrieval are two different problems. The checkpointer gives the agent the whole
conversation, but a follow-up like *"the third one"* carries no search terms on its own — so a
naive retrieval call would fail. There's no separate router rewriting the query anymore; the
**agent itself** handles it. Its system prompt (`src/core/agent/prompts.py`) instructs it,
when a question refers to earlier conversation, to resolve the reference into a standalone
`retrieve_documents` query *before* searching. The agent has the history in `messages`, so it
has everything it needs to do that.

---

## Durability

SQLite makes history survive a **process restart** — *as long as the database file persists*.

- **docker-compose**: the `./data` volume is mounted, so `checkpoints.db` persists across
  restarts.
- **Cloud Run**: the local disk is ephemeral and not shared across instances, so a SQLite file
  there is lost on instance recycle and isn't seen by sibling instances.

For durable, horizontally-scaled memory, swap the checkpointer for a shared backend (e.g.
Postgres via `langgraph-checkpoint-postgres`). `build_graph` already accepts an injectable
checkpointer, so it's a one-line change in `graph.py`.
