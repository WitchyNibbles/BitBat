# BitBat Reliability and Timeline Evolution

## Current State

- **Shipped version:** v1.0 (2026-02-24)
- **Milestone result:** D1/D2/D3 delivered and verified
- **Release acceptance command:** `make test-release`

## What This Is

BitBat is a local-first BTC prediction application with CLI, API, autonomous monitoring, and a Streamlit dashboard. v1.0 stabilized monitoring + schema behavior, repaired and expanded the timeline experience, and standardized Streamlit compatibility guardrails for repeatable release verification.

## Core Value

A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.

## Requirements

### Validated

- ✓ SCHE-01/02/03: Runtime schema compatibility, startup preflight, and idempotent upgrades are in place.
- ✓ MON-01/02/03: Monitor DB runtime stability and actionable fault diagnostics are enforced.
- ✓ TIM-01/02/03/04/05: Timeline reliability and UX expansion are delivered and regression-covered.
- ✓ GUI-01/02/03: Deprecated `use_container_width` usage removed and compatibility checks enforced.
- ✓ QUAL-01/02/03: D1/D2/D3 release gates implemented with canonical acceptance workflow.
- ✓ API-01/02: API/readiness surfaces are aligned with runtime schema/timeline semantics.

### Active (Next Milestone Candidates)

- [ ] ANLY-01: Timeline supports model-vs-model comparative overlays.
- [ ] ANLY-02: Timeline supports exportable segment reports with annotation metadata.
- [ ] OPER-01: Optional database backend migration path beyond SQLite for higher concurrency.
- [ ] Define v1.1 milestone goals and acceptance criteria.

### Out of Scope

- Full dashboard redesign across all pages.
- Major model strategy replacement.
- Multi-tenant auth/permissions redesign.

## Context

v1.0 was delivered through 8 phases (21 plans, 63 tasks) and finalized with release-level verification. The runtime now has explicit schema/readiness contracts, timeline semantics are deterministic and filter-safe, and Streamlit compatibility regressions are prevented by automated checks.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as brownfield stabilization plus targeted enhancement (scope C) | Existing app surfaces were valuable but unstable in critical paths | ✓ Shipped in v1.0 |
| Timeline target is T2 (improve, not just restore) | Timeline quality was central to operator usability | ✓ Shipped in v1.0 |
| Done criteria are D1/D2/D3 | Technical acceptance gates needed to be explicit and verifiable | ✓ Enforced by phase gates + `make test-release` |
| Prioritize DB/runtime correctness before visual polish | Reliability blockers had to be eliminated before UX expansion | ✓ Executed in phase order 1→8 |

## Next Milestone Goals

1. Define v1.1 scope and requirement priorities.
2. Create fresh milestone requirements and roadmap phases.
3. Decide whether to keep phases in-place or archive raw phase directories.

---
*Last updated: 2026-02-24 after v1.0 milestone completion*
