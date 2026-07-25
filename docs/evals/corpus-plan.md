# Synthetic Corpus Plan

The evaluation corpus describes Northstar Labs, a fictional software company. Keeping every
document pack inside one organization creates realistic vocabulary overlap and permits
cross-pack questions without introducing real-company facts.

## Document packs

| Pack | Primary challenges | Planned documents |
|---|---|---:|
| People policies | Superseded policies, regional addenda, conversational follow-ups | 6 |
| Product lifecycle | Similar product names, versioned specifications, release notes | 6 |
| Customer commitments | Plan comparisons, contract exceptions, SLA calculations | 6 |
| Security operations | Playbook versus incident evidence, timelines, remediation status | 6 |
| Facilities and sustainability | Site comparisons, dated targets, numeric synthesis | 6 |
| Finance and planning | Quarterly changes, budgets, forecasts, board decisions | 6 |

Each pack should contain two PDF, two DOCX, and two TXT artifacts unless the content gives a good
reason to vary the mix.

Document lengths must vary deliberately. Each pack should contain short document-level
distractors and longer multi-section documents that cross the application's current chunk
boundary. A benchmark made only of one-chunk documents would not meaningfully exercise passage
retrieval or context precision.

## People policies vertical slice

The first draft pack proves the authoring and validation path before the remaining thirty
documents are created.

| Document | Format | Benchmark role |
|---|---|---|
| 2024 employee handbook | TXT | Superseded values and temporal distractor |
| 2026 employee handbook | PDF | Current company-wide policy |
| Distributed work policy | DOCX | Remote-work rules and numeric limits |
| Regional benefits addendum | PDF | Location-specific comparison |
| Family leave guide | DOCX | Role comparison and return-to-work detail |
| Manager people FAQ | TXT | Supporting evidence and near-duplicate phrasing |

The pack contains ten draft cases: current-policy facts, policy changes over time,
multi-document comparisons, and one two-turn follow-up. These cases remain drafts until the
manual review milestone.
