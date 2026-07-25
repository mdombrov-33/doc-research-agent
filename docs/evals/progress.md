# Evaluation Redesign Progress

Last updated: 2026-07-25  
Branch: `improvements/evaluation-redesign`  
Overall status: milestone 1 complete; synthetic benchmark authoring is next

The authoritative design is [design.md](design.md).

## Milestones

| Milestone | Status | Exit condition |
|---|---|---|
| 0. Design and terminology | Complete | Quality contract, boundaries, metrics, and data strategy agreed |
| 1. Dependencies and structure | Complete | New packages resolve on Python 3.13; schemas and validation command exist |
| 2. Synthetic benchmark draft | Pending | Fact ledger, approximately 36 files, web fixtures, and approximately 60 draft cases exist |
| 3. Benchmark review | Pending | Every accepted case has passed the one-time manual review |
| 4. Shared application runner | Pending | Each case produces one normalized run record from the real graph |
| 5. Evaluator integration | Pending | `ir-measures`, RAGAS, and DeepEval consume the shared record |
| 6. Reports and workflows | Pending | JSON/Markdown reports, baseline comparison, CI validation, and manual workflow work |
| 7. Model selection and baseline | Pending | Canonical generator selected and first reviewed baseline accepted |

## Completed

- Created and switched to `improvements/evaluation-redesign`.
- Audited the existing evaluation harness, tests, workflows, dependencies, and documentation.
- Agreed the document/web/abstention quality contract.
- Defined distinct responsibilities for RAGAS, DeepEval, and `ir-measures`.
- Agreed on a fixed, reviewed, fact-ledger-derived synthetic benchmark.
- Added evaluation terminology to `CONTEXT.md`.
- Added
  [ADR 0001](../adr/0001-use-a-fact-ledger-derived-evaluation-benchmark.md).
- Wrote the evaluation redesign document.
- Pinned RAGAS 0.4.3, DeepEval 4.1.1, and `ir-measures` 0.4.3.
- Proved deterministic metric calls from all three frameworks on Python 3.13.
- Added strict fact-ledger, evaluation-case, web-fixture, and normalized-run schemas.
- Added cross-reference and corpus-artifact validation.
- Added `make eval-validate` as the first command in the replacement harness.
- Disabled DeepEval's transitive pytest rerun plugin; the project does not use retries and the
  plugin opens a local coordination socket during test startup.

## Next slice

Milestone 2:

1. Choose the six related document-pack themes and define the facts, revisions, conflicts, and
   distractors each pack must cover.
2. Add the authoring path that renders corpus documents from the fact ledger.
3. Generate the approximately 36 draft PDF, DOCX, and TXT artifacts.
4. Generate approximately 60 draft cases and deterministic web fixtures.
5. Run schema, cross-reference, and artifact validation over the complete draft.

The draft will not become the accepted benchmark until milestone 3's manual review.

## Decision log

| Decision | Result |
|---|---|
| Primary contract | Supported document answer, supported web answer, or abstention |
| Live web | Fixed fixtures in benchmark; optional live smoke only |
| Framework split | RAGAS for RAG quality; DeepEval for system behavior |
| Ranking metrics | `ir-measures`, not custom formulas |
| Benchmark source | New synthetic data; current corpus is retired |
| Synthetic source of truth | Fact ledger first |
| Benchmark stability | Generated at authoring time, reviewed, fixed, and version-controlled |
| Initial scale | Approximately 36 documents and 60 cases |
| Conversation scope | Approximately six two-turn cases |
| Evidence labels | Stable document IDs and reference passages, not generated chunk IDs |
| Relevance | Required `2`, supporting `1`, irrelevant `0` |
| Application execution | One real graph execution feeds every evaluator |
| Judge configuration | One pinned judge shared by RAGAS and DeepEval |
| Generator selection | One-time 15-case comparison; cheapest qualifying model wins |
| Thresholds | Calibrate from project data; do not import arbitrary values |
| Routine CI | Deterministic validation and graph contracts only |
| Full evaluation | Local or manually dispatched; no nightly run |
| Reporting | Local JSON/Markdown and workflow artifacts |
| Hosted services | None required |
| Production evaluation | LLM-judged production sampling is out of scope |
| Existing graph contracts | Retained as ordinary integration tests |

## Open items

- Exact judge model after compatibility and calibration checks.
- Themes and fact structure for the six document packs.
- Final case counts after deduplication and review.
- Calibration examples and observed judge variance.
- Baseline scores and eventual regression tolerances.

## Verification log

### Milestone 1

- Dependency resolution: RAGAS 0.4.3, DeepEval 4.1.1, and `ir-measures` 0.4.3 resolved and
  installed under Python 3.13.
- Framework smoke tests: deterministic RAGAS exact match, DeepEval tool correctness, and
  `ir-measures` Precision@10 calls passed without provider credentials.
- Focused tests: `10 passed`.
- Static checks: Ruff passed; mypy passed for the new evaluation modules.
- Material issue: DeepEval's transitive `pytest-rerunfailures` plugin tried to open a localhost
  socket during pytest startup. It is disabled because this repository does not use automatic
  test reruns.
