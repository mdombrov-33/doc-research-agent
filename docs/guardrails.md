# Guardrails

Two checks wrap the agent — one before, one after.

## Input check

Runs before LangGraph. Two stages in sequence:

1. **OpenAI Moderation API** — catches harmful, violent, or explicit content. Free, ~100ms.
2. **`gpt-5.4-mini` injection classifier** — catches prompt injection, jailbreaks, system probing. Runs only if moderation passes.

If either fires, the refusal message is returned immediately and LangGraph never runs.

## Output check

Runs after all tokens have been streamed and accumulated. OpenAI Moderation API only — checks the full response for policy violations. If flagged, a correction event replaces the streamed content on the client.

## Fail open

Both checks catch all exceptions and return `None` (safe) on error. A guardrails outage never blocks the agent.
