# BitBat Reliability and Timeline Evolution

## Current State

- **Shipped version:** v1.1 (2026-02-25)
- **Milestone result:** UI-first simplification shipped with a five-view supported surface, retirement-safe legacy routes, and regression/smoke release gates.
- **Release acceptance command:** `make test-release`

## What This Is

BitBat is a local-first BTC prediction platform with CLI, API, autonomous monitoring, and a Streamlit dashboard. After v1.1, the primary operator UI is intentionally reduced to reliable core views with retired advanced paths guarded behind user-facing guidance.

## Core Value

A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.

## Requirements

### Validated

- ✓ SCHE-01/02/03: Runtime schema compatibility, startup preflight, and idempotent upgrades.
- ✓ MON-01/02/03: Monitor runtime fault handling and diagnostics.
- ✓ TIM-01/02/03/04/05: Timeline reliability, UX expansion, readability, and comparison clarity.
- ✓ GUI-01/02/03: Streamlit compatibility and width-contract hardening.
- ✓ API-01/02: API/readiness alignment with runtime semantics.
- ✓ QUAL-01/02/03: D1/D2/D3 release verification contracts.
- ✓ UIF-01/02/03: Supported Streamlit surface constrained to Quick Start, Settings, Performance, About, and System.
- ✓ STAB-01/02/03: Runtime crash paths removed (`confidence`, pipeline import, backtest indexing).
- ✓ RET-01/02: Retired-page UX and supported-surface guidance aligned.
- ✓ QUAL-04/05/06: Simplified UI contract, crash-signature guards, and supported-view smoke coverage release-wired.

### Active (Next Milestone Planning Queue)

- [ ] ANLY-01: Evaluate whether any advanced analytics surfaces should be reintroduced based on validated demand.
- [ ] ANLY-02: Reintroduce backtest workflows only with robust dataset/model/runtime guards.
- [ ] ANLY-03: Rebuild pipeline UX against current modeling/evaluation APIs if advanced workflows return.
- [ ] OPER-01: Evaluate optional non-SQLite backend path for higher concurrency environments.

### Out of Scope

- Full dashboard redesign.
- Major model strategy replacement.
- Reintroducing advanced views without explicit operator demand and verification contracts.

## Context

v1.1 completed in 3 phases (10-12), 8 plans, and 24 tasks. The runtime surface is now intentionally minimal, retired advanced routes fail safely, and release acceptance includes dedicated phase-level coverage for supported-surface, stability, and smoke behavior.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as brownfield stabilization plus targeted enhancement (scope C) | Existing app surfaces were valuable but unstable in critical paths | ✓ Shipped in v1.0 |
| Done criteria are D1/D2/D3 | Technical acceptance gates needed to be explicit and verifiable | ✓ Enforced by canonical `make test-release` |
| Prioritize UI-first simplification for v1.1 | Operator value concentrated in five views; advanced views were broken | ✓ Shipped in v1.1 |
| Retire Backtest/Pipeline routes instead of patching brittle imports in place | Fastest safe path to remove user-facing tracebacks while preserving future rebuild option | ✓ Shipped in v1.1 |
| Lock milestone behavior with dedicated phase gates plus release wiring | Prevent regressions when future work touches Streamlit/runtime contracts | ✓ Shipped in v1.1 |

## Next Milestone Goals

1. Define v1.2 requirements from real operator workflows and current usage evidence.
2. Decide whether advanced interfaces should remain retired or return behind explicit contracts.
3. Preserve D1/D2/D3 release acceptance as the non-negotiable shipping gate.

<details>
<summary>Archived v1.1 kickoff notes</summary>

- Goal: reduce BitBat to actively used views and retire broken surfaces safely.
- Targeted runtime failures: missing `confidence`, pipeline import failure, backtest indexing crash.

</details>

---
*Last updated: 2026-02-25 after completing v1.1 milestone*
