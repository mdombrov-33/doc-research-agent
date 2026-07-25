# Tier 1 progress

Session-reset entry point. Specs live beside this file; cross-cutting decisions in
`00-decisions.md`; ungrilled ideas in `backlog.md`. Statuses: todo / in progress / done.

| Item | Status | Note |
| --- | --- | --- |
| 01 cost telemetry | done | tokens captured; cost NULL — LangChain drops it on streamed calls |
| 02 answer cache | done | Qdrant two-layer cache; first-turn document_answer only; corpus_version invalidation |
| 03 ingestion dedupe | done | file-level SHA-256; duplicate:true skips re-embed; no corpus_version bump on dedupe |
| 04 chunk headers | done | filename header in page_content; eval-retrieval held at ceiling (no regression) |
| 05 conversational path | todo | |
| 06 rerank floor | todo | sweep before enabling |
