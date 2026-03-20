# Memory

How the agent remembers what was said earlier in a conversation.

---

## The short version

Each browser session gets a unique `session_id`. That ID is passed to LangGraph as a `thread_id`, which tells it to load and save state for that specific thread. The conversation history lives inside the agent state as `chat_history` and gets appended to after every response.

---

## Two layers working together

There are actually two things doing memory here, and it's worth knowing what each one does:

**1. MemorySaver (LangGraph)**

When we compile the graph we pass it a checkpointer:

```python
app = workflow.compile(checkpointer=MemorySaver())
```

`MemorySaver` stores the entire agent state in memory, keyed by `thread_id`. So between requests, LangGraph can reload the state from the previous turn — including `chat_history`, the last question, retrieved documents, etc. This is what makes it multi-turn.

**2. chat_history (inside the state)**

`chat_history` is a plain list of messages in the agent state:

```python
chat_history: list[dict[str, str]]
```

Each entry is `{"role": "user" | "assistant", "content": "..."}`. The `generate_node` reads this and injects it into the LLM prompt so the model actually sees the prior conversation:

```python
messages = [
    {"role": "system", "content": ...},  # system prompt with doc context
    *chat_history,                        # all previous turns
    {"role": "user", "content": question} # current question
]
```

After generating a response, it appends the new turn and saves it back:

```python
updated_history = chat_history + [
    {"role": "user", "content": question},
    {"role": "assistant", "content": generation},
]
```

---

## How sessions are isolated

Every request includes a `session_id`:

```python
class QueryRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
```

If the frontend doesn't send one, a new UUID is generated — meaning it's a fresh session with no history. If it sends the same `session_id` across requests, LangGraph loads the matching thread and continues from where it left off.

In `stream.py` this is wired as:

```python
config = {"configurable": {"thread_id": request.session_id}}
async for event in agent.astream_events(inputs, config=config, version="v2"):
    ...
```

That `config` is what tells LangGraph which thread's state to load.

---

## What gets reset each turn

Not everything in the state carries over. Some fields are reset at the start of each request — these are the retrieval fields that are only relevant for the current query:

```python
# in router_node, these reset every turn:
"web_search": web_search,
"web_search_done": False,
"web_fallback_needed": False,
"question": result.rewritten_query,
"raw_documents": None,
"docs_retrieved_total": None,
```

`chat_history` is not reset — it just keeps growing.

---

## The limitation

`MemorySaver` is in-process memory. If the server restarts, all conversation history is gone. For production this would need a persistent checkpointer. LangGraph supports swapping `MemorySaver` out for other backends — options would include a database like Postgres, a cache like Redis, or object storage like S3. The interface stays the same, only the checkpointer changes.
