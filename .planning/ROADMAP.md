# Roadmap: BitBat Reliability and Timeline Evolution

## Milestones

- ✅ **v1.0 Reliability and Timeline Evolution** — Phases 1-9 shipped on 2026-02-25 ([roadmap archive](milestones/v1.0-ROADMAP.md), [requirements archive](milestones/v1.0-REQUIREMENTS.md), [audit archive](milestones/v1.0-MILESTONE-AUDIT.md)).
- 🚧 **v1.1 UI-First Simplification** — Phases 10-12 planned (reduce UI surface to the operator-used views and retire broken advanced pages safely).

## v1.1 Planned Phases

### Phase 10: Supported Surface Pruning
**Goal:** Keep BitBat focused on the five actively used views and remove non-core pages from normal navigation.
**Depends on:** v1.0 verified baseline
**Requirements:** [UIF-01, UIF-02, UIF-03, RET-02]
**Plans:** 3 planned

Success criteria:
1. Sidebar and page discovery expose only Quick Start, Settings, Performance, About, and System.
2. Home quick actions and internal links route only to supported views.
3. Non-supported pages are no longer reachable from the default runtime surface.
4. About/help copy no longer references retired advanced pages.

### Phase 11: Runtime Stability and Retirement Guards
**Goal:** Remove current crash paths from app startup and legacy route access behavior.
**Depends on:** Phase 10
**Requirements:** [STAB-01, STAB-02, STAB-03, RET-01]
**Plans:** 3 planned

Success criteria:
1. `streamlit/app.py` no longer raises `KeyError: 'confidence'` when data is partial.
2. Normal UI startup path does not fail from pipeline-only imports.
3. Users cannot hit the current backtest indexing crash from standard navigation.
4. Legacy route access yields user-facing guidance or redirect instead of traceback.

### Phase 12: Simplified UI Regression Gates
**Goal:** Lock the simplified UI contract and crash fixes with automated verification.
**Depends on:** Phase 11
**Requirements:** [QUAL-04, QUAL-05, QUAL-06]
**Plans:** 2 planned

Success criteria:
1. Automated tests enforce the supported-page contract and guard against retired-page link regressions.
2. Regression tests cover the reported failure signatures via guard/retirement behavior.
3. Smoke coverage confirms the five supported views render cleanly.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 10. Supported Surface Pruning | v1.1 | 0/3 | Not started | - |
| 11. Runtime Stability and Retirement Guards | v1.1 | 0/3 | Not started | - |
| 12. Simplified UI Regression Gates | v1.1 | 0/2 | Not started | - |

## Next

- Start execution with `$gsd-plan-phase 10` (or `$gsd-discuss-phase 10` first).
