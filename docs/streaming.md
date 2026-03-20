# Streaming

How a question goes from the user typing it to tokens appearing on screen one by one.

---

## The short version

We use Server-Sent Events (SSE). The backend streams JSON lines over an open HTTP connection, and the frontend reads them as they arrive. Each token is its own message. At the very end, one final message carries metadata (sources, session id, etc.).

---

## Why SSE and not just `yield chunks`?

Plain chunked streaming works fine if all you're sending is text. But over the same connection we need to send different types of things:

- Tokens (the text itself, one by one)
- Sources metadata at the end (filenames, chunk indices, where it came from)
- A correction flag if NeMo flagged the output
- Errors

With raw chunked streaming the frontend gets a byte stream with no structure — it can't tell if what just arrived is a token, a source list, or an error. We'd end up inventing our own delimiter format anyway.

SSE gives us that structure for free. Every message is `data: {...}\n\n` so the client just parses JSON and checks what's inside. It immediately knows: is this a token? Is this the done event? Is `correction: true`?

WebSockets would also work but they're bidirectional — the client can send messages back mid-stream. We don't need that. SSE is unidirectional (server → client only), which is exactly the case here, and it's plain HTTP so there's nothing extra to set up.

---

## The flow end to end

```
User types question
      ↓
POST /api/stream
      ↓
NeMo input check → blocked? send refusal and close stream
      ↓ (if safe)
LangGraph agent starts running
      ↓
on_chat_model_stream events → each token → SSE line → client renders it
      ↓
on_chain_end event → grab sources metadata
      ↓
NeMo output check on full accumulated response
      ↓
Final SSE event: done (with sources) or correction
```

---

## Backend — how tokens get sent

Everything lives in `src/api/handlers/stream.py`. FastAPI's `StreamingResponse` wraps an async generator, which is the key piece:

```python
async def _token_generator(request: QueryRequest) -> AsyncGenerator[str, None]:
    ...
    async for event in agent.astream_events(inputs, config=config, version="v2"):
        if kind == "on_chat_model_stream" and langgraph_node == "generate":
            token = event["data"]["chunk"].content
            if token:
                accumulated.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
```

Each `yield` sends one SSE line to the client immediately. The format is always `data: {...}\n\n` — that's the SSE spec.

**Why `astream_events` and not `astream`?**

`astream` gives us full state snapshots after each node completes — too coarse for token-level streaming. `astream_events` gives us low-level events including `on_chat_model_stream`, which fires for every single token the LLM produces. We filter to only the `generate` node so we don't accidentally stream tokens from other LLM calls (like the grader or router).

**What events we actually care about:**

| Event                                       | When it fires  | What we do with it                          |
| ------------------------------------------- | -------------- | ------------------------------------------- |
| `on_chat_model_stream` from `generate` node | Every token    | Yield it to client, append to `accumulated` |
| `on_chain_end` from `LangGraph`             | Graph finishes | Pull sources metadata out of final state    |

Everything else (router events, retrieval events, etc.) is ignored.

---

## The SSE message format

Three types of messages the frontend can receive:

**Token (mid-stream):**

```json
{"token": "The"}
{"token": " answer"}
{"token": " is"}
```

**Done (happy path):**

```json
{"done": true, "sources_count": 3, "sources": [...], "session_id": "..."}
```

**Correction (output flagged by NeMo):**

```json
{
  "token": "I can only provide information related to document research...",
  "done": true,
  "correction": true
}
```

**Error:**

```json
{ "error": "something went wrong", "done": true }
```

---

## Frontend — how tokens get rendered

In `ui.py`, the `stream_query()` function is a generator that consumes the SSE stream:

```python
def stream_query(question: str, model: str | None, top_k: int):
    with requests.post(..., stream=True) as response:
        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = json.loads(line[6:])
            if "token" in data:
                yield data["token"]
            if data.get("done"):
                st.session_state.stream_meta = data
                return
```

Streamlit's `st.write_stream()` accepts a generator and renders each yielded value as it arrives. The final metadata (sources, etc.) is stashed in session state and rendered separately after the stream closes.

---

## Metadata and evaluation

While streaming, we track everything needed for the evaluation dashboard:

- `sources_count` — how many docs passed grading
- `sources_meta` — filename, chunk index, chunk length, source (vectorstore vs web)
- `web_search_triggered` — whether the web fallback ran
- `docs_retrieved_total` — total before grading
- `latency_ms` — full request duration from first token to done event

This gets recorded into the evaluation tracker right before the final SSE event is sent.
