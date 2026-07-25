# Backlog — not grilled, not scheduled

Ideas from the original audit of `~/Projects/ai-daddy` and
`~/Projects/ai-engineering-from-scratch` that were *not* promoted into Tier 1 specs. Grill an
item into its own numbered spec before implementing it. Main audit sources:
`ai-daddy/06-retrieval-systems/14-production-rag-at-scale.md` (richest),
`08-memory-and-state/05-semantic-caching.md`, `13-reliability-and-safety/03-reliability-patterns.md`,
`10-document-processing/01-ocr-and-layout.md`, `15-ai-design-patterns/02-anti-patterns.md`,
`ai-engineering-from-scratch/phases/17-infrastructure-and-production/` (caching/routing docs;
its code is toy simulators — concept source only).

## Medium-term

- **Prompt-cache-friendly prompt layout** — byte-stable system-prompt prefix (no dynamic
  content), stable conversation prefix before per-turn evidence; verify via provider
  `cached_tokens`. OpenRouter passes provider caching through for supported models.
- **One bounded query reformulation before web fallback** (CRAG) — on first insufficient
  verdict, reformulate the query (told why it failed) and retrieve from documents once more;
  one extra hop, no loops.
- **VLM extraction fallback for scanned/visual PDFs** — per page, if text density is below a
  threshold, rasterize and transcribe to Markdown via a vision model; per-document page cap;
  also start populating `SourceCitation.page` (always empty today).
- **Provider resilience** — embeddings are a single direct OpenAI dependency on the hot path:
  degrade to sparse-only (BM25) retrieval on embedding failure; small in-process circuit
  breaker in front of chat/embedding calls. Full gateway/hedging rejected at this scale.
- **Prompt-injection input scanner** (llm-guard PromptInjection) — the question itself is
  unscanned today; watch Cloud Run memory and false positives.
- **Nightly sampled quality check on live traffic** — run judges inline post-stream on a
  sample, store scores only (preserves the no-content telemetry stance).

## Architectural bets

- **Multi-tenancy** — *decision already made* (see `00-decisions.md`): single-tenant,
  forward-compatible via the `namespace` field. Full build (auth, mandatory tenant filter on
  every query, tenant-keyed limits and cache partitioning) stays here until needed.
- **Model cascade for answer generation** — cheap default, escalate on borderline assessment
  or complex query; gated by online metrics (cheap-model drift, escalation-rate alarm).
  Requires cost telemetry (01) to prove ROI.
- **Hierarchical retrieval (document-summary index)** — only at real corpus scale.
- **Page-as-image retrieval (ColPali-style)** — track, don't build.
- **Degradation ladder** — explicit `DegradationLevel` (full → skip rerank → sparse-only →
  cache-only → error) once the rungs (cache, sparse fallback) exist.

## Considered and rejected

- Long-context instead of RAG; AI gateway (LiteLLM/Portkey); request hedging; fine-tuning;
  GraphRAG; static price table for cost telemetry; chunk-level embedding cache; classifier
  for conversational routing; selective cache invalidation via supporting-doc IDs.

## Anti-pattern check (from the audit)

Already avoided: God Prompt, unbounded agent loops, missing metadata, missing rate limiting,
vibes-based evaluation. Hit at audit time: **No Caching** (→ spec 02), single-provider
embeddings (→ resilience, above).
