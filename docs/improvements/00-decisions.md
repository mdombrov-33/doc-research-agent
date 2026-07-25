# Cross-cutting decisions (grilled 2026-07-25)

Settled in the grilling session that produced these specs. They bind all items.

- **Scope**: Tier 1 only (items 01–06). Tier 2–3 remain ideas in `backlog.md`.
- **Tenancy**: the app is declared **single-tenant, forward-compatible**. One shared corpus,
  global answer cache. Every new key structure (cache entries, corpus version) carries a
  `namespace` field hardcoded to `"default"` so a real `tenant_id` can be added additively
  later, without a migration.
- **No ADRs**: decisions live in these spec files.
- **Docs are part of every item's definition of done**: `docs/architecture.md`, `README.md`,
  and any affected doc under `docs/` are updated in the same slice as the code.
- **Implementation order** = file numbering: 01 telemetry → 02 cache → 03 dedupe →
  04 headers → 05 conversational → 06 rerank floor. Telemetry lands first so the cache's
  savings are measurable.
- **Workflow**: direct on main, one item at a time, small reviewable slices, commit message
  provided then wait, push only when asked.
- **Corpus is append-only** (no delete endpoint exists); several designs below rely on this.
- Track status in `progress.md`; that file is the session-reset entry point.
