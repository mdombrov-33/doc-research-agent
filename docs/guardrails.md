# Guardrails

Two checks wrap the agent — one before, one after.

## Input check

Runs before LangGraph. Two independent stages run **concurrently** (`asyncio.gather`)
to keep latency low:

1. **OpenAI Moderation API** — catches harmful, violent, or explicit content. Free, ~100ms.
2. **`gpt-5.4-mini` injection classifier** — catches prompt injection, jailbreaks, system probing.

If either fires, the refusal message is returned and LangGraph never runs.

> **Provider split:** the moderation call goes directly to OpenAI (`OPENAI_API_KEY`),
> while the injection classifier goes through OpenRouter (`OPENROUTER_API_KEY`) via
> `get_llm`. An input check therefore touches both providers.

## Output check

Runs after all tokens have been streamed and accumulated. OpenAI Moderation API only.

**Best-effort, not a hard block.** Because the response is streamed token-by-token, the
content has already reached the user by the time this check runs. So it serves two
purposes — a monitoring signal (logged as `guardrails_output_flagged`) and a client-side
correction flag — but it cannot guarantee unsafe content is never shown. Hard-blocking
would require buffering the full response before sending (giving up streaming), which we
deliberately don't do.

## Fail open

Both checks catch all exceptions and return safe (not flagged) on error. A guardrails
outage never blocks the agent.
