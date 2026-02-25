# BitBat Reliability and Timeline Evolution

## Current State

- **Shipped version:** v1.0 (2026-02-25 final closure)
- **Milestone result:** D1/D2/D3 delivered and verified, post-audit timeline readability gaps closed
- **Release acceptance command:** `make test-release`

## Current Milestone: v1.1 UI-First Simplification

**Goal:** Reduce BitBat to the views that are actively used and harden the UI by removing/retiring broken surfaces.

**Target features:**
- Keep only `Quick Start`, `Settings`, `Performance`, `About`, and `System` as supported views.
- Retire non-used/broken views (`Alerts`, `Analytics`, `History`, `Backtest`, `Pipeline`) from primary navigation.
- Ensure removed or legacy routes fail gracefully with user-facing guidance rather than tracebacks.

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

### Active (v1.1 Scope)

- [ ] Streamlit navigation and entry points expose only the five supported views.
- [ ] Broken views no longer crash runtime (`KeyError: confidence`, pipeline import error, backtest indexing error).
- [ ] Legacy/non-supported view access is handled safely (retired page notice or redirect).
- [ ] UI copy and internal links no longer reference retired pages.
- [ ] Regression tests enforce the simplified UI surface contract.

### Out of Scope

- Full dashboard redesign across all pages.
- Major model strategy replacement.
- Multi-tenant auth/permissions redesign.

## Context

v1.0 was delivered through 9 phases (24 plans, 72 tasks), including a post-audit closure phase for timeline readability and comparison clarity. The runtime now has explicit schema/readiness contracts, timeline semantics/readability are regression-tested, and Streamlit compatibility regressions are prevented by automated checks.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as brownfield stabilization plus targeted enhancement (scope C) | Existing app surfaces were valuable but unstable in critical paths | ✓ Shipped in v1.0 |
| Timeline target is T2 (improve, not just restore) | Timeline quality was central to operator usability | ✓ Shipped in v1.0 |
| Done criteria are D1/D2/D3 | Technical acceptance gates needed to be explicit and verifiable | ✓ Enforced by phase gates + `make test-release` |
| Prioritize DB/runtime correctness before visual polish | Reliability blockers had to be eliminated before UX expansion | ✓ Executed in phase order 1→8 |

## Next Milestone Goals

1. Ship a stable, minimal UI surface aligned with actual operator usage.
2. Remove dependence on currently broken advanced pages from normal runtime.
3. Lock simplified behavior with regression gates before considering advanced-page reintroduction.

---
*Last updated: 2026-02-25 after starting v1.1 milestone*
