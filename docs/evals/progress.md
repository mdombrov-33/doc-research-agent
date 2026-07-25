# Evaluation Redesign Progress

Last updated: 2026-07-25  
Branch: `improvements/evaluation-redesign`  
Overall status: milestone 2 in progress; first document pack validates

The authoritative design is [design.md](design.md).

## Milestones

| Milestone | Status | Exit condition |
|---|---|---|
| 0. Design and terminology | Complete | Quality contract, boundaries, metrics, and data strategy agreed |
| 1. Dependencies and structure | Complete | New packages resolve on Python 3.13; schemas and validation command exist |
| 2. Synthetic benchmark draft | In progress | Fact ledger, approximately 36 files, web fixtures, and approximately 60 draft cases exist |
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
- Defined six related Northstar Labs document packs and their intended retrieval challenges.
- Added fact and document revision metadata to the ledger schema.
- Added `make eval-author` to render PDF, DOCX, and TXT artifacts from the ledger.
- Added artifact-content validation so committed files cannot silently drift from ledger
  passages.
- Generated the People policies vertical slice: six documents and ten draft cases, including one
  two-turn conversation.
- Isolated PyMuPDF at an `Any` boundary in artifact validation because its editor-visible
  `Document` annotations do not match the runtime API.

## Next slice

Continue milestone 2:

1. Expand the fact ledger across Product, Customer, Security, Facilities, and Finance.
2. Render the remaining approximately 30 PDF, DOCX, and TXT artifacts.
3. Use RAGAS-assisted generation to propose the remaining cases, then normalize them into the
   project schema.
4. Add deterministic web fixtures and document/web/abstention cases.
5. Deduplicate to approximately 60 useful draft cases and validate the complete draft.

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

### Milestone 2 vertical slice

- Authoring: six corpus artifacts rendered from the fact ledger across PDF, DOCX, and TXT.
- Reproducibility: rerendering the same ledger produces byte-identical artifacts.
- Validation: `Validated 6 documents, 10 cases, and 0 web fixtures.`
- Focused tests: `11 passed`.
- Artifact extraction: all three generated formats were read through the production extraction
  path in tests.
- Chunk-shape check: the pack mixes four short one-chunk documents with two longer two-chunk
  documents under the current production chunker.
- Full verification: Ruff, formatting, mypy, and the complete pytest suite passed.
- Material issue: PyMuPDF's legacy and canonical imports produced inconsistent editor type
  information. The PDF reader now uses distinct DOCX/PDF variables and limits `Any` to the
  opened third-party PDF object.
