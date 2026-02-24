# Project Research Summary

**Project:** BitBat Reliability and Timeline Evolution
**Domain:** Brownfield ML monitoring/dashboard stabilization
**Researched:** 2026-02-24
**Confidence:** HIGH

## Executive Summary

This project is a brownfield stabilization and enhancement effort, not a greenfield build. The highest-leverage path is to re-establish schema/runtime correctness first, then harden timeline data contracts, then deliver targeted UX improvements and compatibility cleanup. Reliability must be treated as a product feature: if monitor cycles fail or timeline semantics are ambiguous, user trust collapses regardless of model quality.

The recommended approach is a migration-first architecture with explicit schema compatibility checks, a dedicated timeline read-model adapter, and a one-pass Streamlit API modernization (`use_container_width` -> `width=`). This sequence minimizes blast radius while delivering visible operator value quickly.

Key risks are recurring schema drift, silent failure behavior in monitor/UI paths, and regression churn from missing targeted tests. These are mitigated through phased hardening and acceptance gates tied to D1/D2/D3.

## Key Findings

### Recommended Stack

The existing stack is appropriate for this scope: Python + SQLAlchemy + Streamlit + FastAPI + SQLite/parquet artifacts. The main gap is migration discipline for brownfield schema evolution.

**Core technologies:**
- **SQLAlchemy + SQLite:** operational state and prediction history; add migration/version checks.
- **Streamlit:** timeline/UI surface; modernize to `width=` API to remove deprecation debt.
- **Pytest + quality tooling:** enforce D1/D2/D3 via regression tests and CI.

### Expected Features

**Must have (table stakes):**
- Monitor runs without DB schema runtime failures (D1).
- Timeline renders predictions + realized outcomes correctly with filters and confidence context (D2).
- No Streamlit `use_container_width` deprecation warnings (D3).

**Should have (competitive):**
- Timeline overlays that make prediction quality quickly legible.
- Actionable failure banners/diagnostics.

**Defer (v2+):**
- Advanced comparative timeline/reporting workflows.

### Architecture Approach

Use a compatibility-first architecture: run schema preflight/migrations before monitor cycles, isolate timeline query normalization from page rendering, and keep UI modernization as a cross-page consistency sweep.

**Major components:**
1. **Schema/Migration boundary** - guarantees DB/model alignment before runtime operations.
2. **Monitor + validator services** - produce and realize predictions reliably.
3. **Timeline read-model + UI renderer** - transforms operational records into stable chart/table presentation.

### Critical Pitfalls

1. **ORM/schema drift** - prevented by migration/versioning and startup checks.
2. **Timeline assumptions on ideal data** - prevented by normalized timeline adapters.
3. **Silent exception swallowing** - prevented by critical-path failure surfacing.
4. **Deprecation debt accumulation** - prevented by one-pass API migration and regression guard.
5. **Fixes without tests** - prevented by D1/D2/D3 acceptance tests in CI.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Schema Compatibility Recovery
**Rationale:** Monitor stability is blocked by DB mismatch.
**Delivers:** Versioned schema compatibility/migration path and successful monitor runs.
**Addresses:** D1 baseline.
**Avoids:** repeated `OperationalError` outages.

### Phase 2: Monitor Runtime Hardening
**Rationale:** After schema recovery, failure semantics and DB interactions must be robust.
**Delivers:** Hardened monitor/predictor/validator DB paths with better diagnostics.
**Uses:** existing autonomous architecture.
**Implements:** critical-path error handling discipline.

### Phase 3: Timeline Data Contract + UX Core
**Rationale:** Timeline correctness depends on stable normalized data flows.
**Delivers:** Correct timeline rendering with filters, confidence, and pending/realized semantics.
**Implements:** read-model adapter and rendering contract tests.

### Phase 4: Streamlit Compatibility Sweep
**Rationale:** Deprecation cleanup is broad and should be isolated.
**Delivers:** Global `use_container_width` removal and warning-free GUI interactions.
**Addresses:** D3.

### Phase 5: Regression Guardrails
**Rationale:** Prevent re-breakage.
**Delivers:** Targeted tests and CI checks for D1/D2/D3; runtime smoke checks.

### Phase 6: Timeline Enhancements (T2 expansion)
**Rationale:** Scope C includes feature additions after stabilization.
**Delivers:** richer overlays/annotations and operator context improvements.

### Phase 7: API and Monitoring Surface Alignment
**Rationale:** Ensure API/UI/monitor share consistent semantics and fields.
**Delivers:** aligned schemas, docs, and endpoint behavior under new timeline model.

### Phase 8: Final Verification and Release Readiness
**Rationale:** comprehensive depth requires explicit UAT closure.
**Delivers:** end-to-end verification for D1/D2/D3 and readiness checklist.

### Phase Ordering Rationale

- DB correctness must precede all timeline/UI work.
- Timeline correctness must precede timeline enhancements.
- Compatibility cleanup and regression guards should land before wider scope-C features.
- Final phases consolidate cross-surface consistency and release confidence.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** migration strategy details for existing local SQLite states.
- **Phase 6:** best visual/interaction patterns for timeline overlays in Streamlit.

Phases with standard patterns (skip research-phase):
- **Phase 4:** Streamlit deprecation replacement sweep.
- **Phase 5:** regression test and CI guardrail setup.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Existing stack is already deployed and suitable for scope |
| Features | HIGH | User-defined D1/D2/D3 and T2/C scope are explicit |
| Architecture | HIGH | Existing modular boundaries support phased stabilization |
| Pitfalls | HIGH | Current failures directly match known brownfield failure patterns |

**Overall confidence:** HIGH

### Gaps to Address

- Exact migration strategy for already-mutated local DB variants in the field.
- Exact timeline enhancement interaction details to be finalized during phase planning.

## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md`
- `.planning/codebase/STACK.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/CONCERNS.md`
- Runtime code in `src/bitbat/autonomous/` and `streamlit/`

### Secondary (MEDIUM confidence)
- Existing docs under `docs/` and workflow assumptions in current code comments/config

### Tertiary (LOW confidence)
- None used for decision-critical claims

---
*Research completed: 2026-02-24*
*Ready for roadmap: yes*
