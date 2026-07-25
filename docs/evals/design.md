# Evaluation Redesign

Status: agreed design; implementation in progress  
Branch: `improvements/evaluation-redesign`

## Purpose

Replace the existing hand-written evaluation harness with a reproducible benchmark built on
RAGAS, DeepEval, and `ir-measures`.

The benchmark protects one user-visible quality contract:

> Given a question and an uploaded corpus, the system produces a relevant answer supported by
> correctly cited evidence, uses web evidence when the corpus is insufficient, or abstains when
> neither source is sufficient.

Component metrics diagnose why that contract failed. They are not interchangeable product goals.

## Outcomes

Every evaluation case expects exactly one outcome:

- **Document-grounded answer** — claims are supported by the evaluation corpus.
- **Web-grounded answer** — the corpus is insufficient and fixed web evidence supports the answer.
- **Abstention** — neither document nor web evidence is sufficient.

The benchmark evaluates the decision to use web evidence, but not the quality of live internet
search. Live search remains an optional smoke check.

## Design principles

- Run the real ingestion, retrieval, and compiled graph paths.
- Execute the application once per case and evaluate that same captured run everywhere.
- Separate exact behavioral checks from LLM-judged quality metrics.
- Prefer framework metrics over locally implemented scoring formulas.
- Use fixed, reviewed data for meaningful regression comparisons.
- Report per-case and per-category results, not only global averages.
- Calibrate judges and thresholds against this benchmark rather than copying external numbers.
- Keep the system local-first, manually runnable, and inexpensive enough for a small project.

## Evaluation architecture

```text
Fact ledger
    |
    +--> fixed PDF/DOCX/TXT evaluation corpus
    +--> reviewed evaluation cases
                  |
                  v
       real ingestion into isolated Qdrant
                  |
                  v
       one compiled-graph execution per case
                  |
                  v
          normalized run record
             /        |        \
            v         v         v
     ir-measures    RAGAS    DeepEval
       ranking     RAG quality  system behavior
             \        |        /
              v       v       v
             JSON + Markdown report
```

## Benchmark design

### Fact ledger

The fact ledger is the authoritative synthetic world model. It defines facts, document
relationships, revisions, dates, deliberate conflicts, and the source documents in which each
fact appears.

Documents and expected answers are derived from the same ledger. They must not be generated
independently.

See
[ADR 0001](../adr/0001-use-a-fact-ledger-derived-evaluation-benchmark.md).

### Evaluation corpus

Initial target:

- Approximately 36 documents in six related document packs.
- A meaningful mix of PDF, DOCX, and TXT.
- Overlapping facts, plausible distractors, revisions, and temporal conflicts.
- Documents generated at authoring time, reviewed, and committed.

The current 17-document corpus will be retired. Its contents are not a compatibility constraint
for the new benchmark.

### Evaluation cases

Initial target: approximately 60 cases.

Coverage must include:

- Single-document factual questions.
- Multi-document synthesis.
- Comparisons.
- Temporal and version-sensitive questions.
- Near-neighbour distractors.
- Insufficient-corpus cases that require web evidence.
- Cases where the correct result is abstention.
- Approximately six two-turn conversational follow-ups.

Counts may change during review. Coverage is the constraint, not an exact total.

Each accepted case contains:

- Stable case ID and coverage tags.
- User input or two-turn conversation.
- Expected outcome.
- Reference answer when an answer is expected.
- Reference passages.
- Expected tool calls.
- Graded document relevance judgments.
- Fixed web results for web-grounded and abstention scenarios.

Document relevance uses three grades:

- `2` — required evidence.
- `1` — supporting evidence.
- `0` — irrelevant and normally omitted from labels.

Generated chunk IDs are not benchmark labels. Chunking is a variable the benchmark must be able
to compare.

### Authoring and acceptance

Synthetic generation happens only during benchmark authoring. RAGAS may propose evaluation cases,
but it does not overwrite accepted data.

Every accepted case receives one manual review for:

- Question clarity.
- Correct expected outcome.
- Correct required and supporting evidence.
- Reference-answer agreement with the fact ledger.
- Unique coverage value.

## Application execution

The runner:

1. Creates or resets an isolated evaluation collection.
2. Ingests the fixed files through the real ingestion pipeline.
3. Runs the compiled graph once per case.
4. Supplies deterministic web fixtures when the graph requests web search.
5. Captures a normalized run record.
6. Adapts the record to `ir-measures`, RAGAS, and DeepEval.

A run record contains:

- Case and conversation input.
- Retrieved contexts and stable document identities in retrieval order.
- Final response and outcome.
- Tool calls and arguments.
- Resolved citations.
- Generator, embedding, reranker, chunking, prompt, and framework configuration.
- Timing, token usage when available, and evaluation errors.

Evaluator failure is reported separately from application failure. A broken judge must not be
misreported as a bad application response.

## Metrics

### Retrieval ranking with `ir-measures`

Calculate established information-retrieval metrics from graded document judgments:

- Precision@10.
- Recall@10, with required evidence as the recall target.
- Reciprocal rank.
- Mean average precision.
- NDCG@10.

These replace the formulas in the current `evals/ranking.py`.

### RAG quality with RAGAS

Initial scorecard:

- Context precision.
- Context recall.
- Faithfulness.
- Factual correctness.
- Response relevancy.

RAGAS evaluates retrieved passages and generated answers. Exact document ranking remains the
responsibility of `ir-measures`.

### System behavior with DeepEval

Initial scorecard:

- Exact expected outcome.
- Strict tool correctness.
- Citation faithfulness for answered cases.
- Conversation completeness for multi-turn cases.

Generic planning metrics are out of scope. The graph is bounded and has no open-ended planning
policy to evaluate.

## Models

### Canonical generator

The accepted baseline protects one canonical application configuration, not every model exposed
in the UI.

Before selecting it, run a one-time comparison over approximately 15 representative cases:

- Claude Haiku 4.5.
- GPT-5.4 Mini.
- GPT-5.6 Luna.

The least expensive model that satisfies the quality contract becomes the default and canonical
benchmark target. Model comparison is an experiment, not a permanent test matrix.

### Judge

RAGAS and DeepEval share one explicit, pinned judge configuration. The judge should not be one of
the generator candidates when practical.

Every report records:

- Judge model and provider.
- Framework versions.
- Judge prompts or metric versions.
- Sampling and embedding configuration.

The judge must pass calibration cases before its benchmark scores are accepted.

## Calibration and regression policy

Do not start with copied absolute thresholds.

1. Build obvious positive and negative calibration examples.
2. Verify expected judge behavior.
3. Run and manually inspect the complete reviewed benchmark.
4. Accept that run as the first baseline.
5. Repeat enough cases to observe ordinary judge variance.
6. Introduce regression tolerances only after variance is known.

Wrong outcomes and structurally invalid citations are exact case failures. LLM-judged scores are
initially diagnostic and are reported by category and case.

## Execution modes

### Deterministic CI

Normal CI runs:

- Fact-ledger, case-schema, artifact, and cross-reference validation.
- Existing graph workflow contracts as ordinary integration tests.
- Unit tests for adapters and report handling.

It does not make paid judge calls.

### Full benchmark

The complete benchmark runs:

- Locally through one documented command.
- Through an optional manually dispatched GitHub workflow.

It does not run nightly and does not initially block every pull request.

### Model-selection experiment

A separate command runs the representative subset against generator candidates and produces a
cost/quality comparison. It is not part of routine regression testing.

## Reports and baselines

Each full run produces:

- A machine-readable JSON report.
- A concise Markdown summary.
- Per-case results and evaluator reasons.
- Category aggregates.
- Application and evaluator configuration.
- Comparison with the accepted baseline.
- Estimated or observed evaluation cost when available.

Commit the accepted baseline summary and its metadata. Keep detailed transient reports local or
publish them as GitHub workflow artifacts.

Do not require Confident AI, Langfuse, Braintrust, or another hosted platform.

## Production boundary

Runtime operational telemetry remains separate and may continue to track outcomes, retrieval
counts, web use, errors, and latency.

The redesign does not:

- Send production questions to RAGAS or DeepEval.
- Add LLM-judged production sampling.
- Add evaluation-specific user-data storage.
- Add a hosted observability dependency.

## Replacement scope

Retire the current:

- Corpus and `golden.jsonl`.
- Custom ranking module.
- Custom LLM judges.
- Embedding-separation guard.
- Offline and live-graph eval runners.
- Eval-specific unit tests and push-to-main eval workflow.

Retain:

- Deterministic compiled-graph behavior tests.
- Production ingestion, retrieval, graph, citation, and monitoring code.

The retained graph tests become normal integration tests rather than a separately branded eval
tier.

## Planned implementation slices

1. Replace dependencies and create the benchmark schemas, validation, and command structure.
2. Build the fact ledger, corpus authoring path, web fixtures, and draft cases.
3. Review and accept the synthetic benchmark.
4. Implement shared graph execution and normalized run capture.
5. Integrate `ir-measures`, RAGAS, and DeepEval.
6. Add calibration, reports, baseline comparison, and model-selection mode.
7. Simplify CI, add the manual workflow, update project documentation, and establish the first
   accepted baseline.

Implementation status is tracked in [progress.md](progress.md).

## References

- [RAGAS documentation](https://docs.ragas.io/)
- [DeepEval RAG evaluation](https://deepeval.com/docs/getting-started-rag)
- [DeepEval agent evaluation](https://deepeval.com/docs/getting-started-agents)
- [`ir-measures` documentation](https://ir-measur.es/en/latest/)
