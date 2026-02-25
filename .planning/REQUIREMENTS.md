# Requirements: BitBat UI-First Simplification

**Defined:** 2026-02-25
**Status:** v1.1 in progress (Phases 10-11 complete, Phase 12 pending)
**Core Value:** A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.

## v1.1 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### Supported UI Surface

- [x] **UIF-01**: Streamlit sidebar navigation exposes only these supported views: Quick Start, Settings, Performance, About, and System.
- [x] **UIF-02**: Home page quick actions and internal links route only to supported views.
- [x] **UIF-03**: Non-supported pages (`Alerts`, `Analytics`, `History`, `Backtest`, `Pipeline`) are removed from the normal runtime surface.

### Runtime Stability

- [x] **STAB-01**: The home app handles missing prediction fields (including `confidence`) without raising `KeyError`.
- [x] **STAB-02**: Normal UI startup path does not crash from pipeline-only imports (including `classification_metrics` import failures).
- [x] **STAB-03**: Users cannot encounter the current backtest indexing runtime crash from the standard UI flow.

### Retired View UX

- [x] **RET-01**: Accessing a retired page path yields a user-friendly retirement notice or redirect, not a traceback.
- [x] **RET-02**: About/help copy references only the supported view set and current navigation model.

### Quality & Verification

- [ ] **QUAL-04**: Automated tests enforce the supported-page contract and reject reintroduction of retired-page links in core UI surfaces.
- [ ] **QUAL-05**: Automated regressions cover the reported failure signatures (`KeyError: confidence`, pipeline import crash, backtest indexing crash) via guards or retirement behavior.
- [ ] **QUAL-06**: Smoke tests verify that Quick Start, Settings, Performance, About, and System render without runtime exceptions.

## v1.2+ Requirements (Deferred)

Deferred to a later milestone after the UI-first baseline is stable.

### Advanced Interfaces

- **ANLY-01**: Reintroduce advanced analytics views with validated operator demand and stable contracts.
- **ANLY-02**: Reintroduce backtest workflows with robust dataset/model guards.
- **ANLY-03**: Rebuild pipeline UI against current modeling APIs and evaluation contracts.

### Operations

- **OPER-01**: Optional database backend migration path beyond SQLite for higher concurrency.

## Out of Scope

Explicitly excluded for v1.1 to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Full product redesign | Goal is simplification and stabilization of current UI, not full redesign |
| New model strategy research | Current milestone is UI surface reliability, not model experimentation |
| Rebuilding all advanced/technical pages now | User value is concentrated in five core views; advanced pages are deferred |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| UIF-01 | Phase 10 | Complete |
| UIF-02 | Phase 10 | Complete |
| UIF-03 | Phase 10 | Complete |
| RET-02 | Phase 10 | Complete |
| STAB-01 | Phase 11 | Complete |
| STAB-02 | Phase 11 | Complete |
| STAB-03 | Phase 11 | Complete |
| RET-01 | Phase 11 | Complete |
| QUAL-04 | Phase 12 | Pending |
| QUAL-05 | Phase 12 | Pending |
| QUAL-06 | Phase 12 | Pending |

**Coverage:**
- v1.1 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-25*
*Last updated: 2026-02-25 after Phase 11 completion*
