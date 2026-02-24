# Requirements: BitBat Reliability and Timeline Evolution

**Defined:** 2026-02-24
**Core Value:** A reliable prediction system where operators can trust that monitoring runs without DB failures and the timeline shows clear prediction vs. outcome history.

## v1 Requirements

Requirements for this release. Each maps to exactly one roadmap phase.

### Schema & Data Compatibility

- [x] **SCHE-01**: Existing local databases can be upgraded safely so `prediction_outcomes` contains required runtime columns (including `predicted_price`) without manual table recreation.
- [x] **SCHE-02**: Application startup validates schema compatibility and surfaces actionable errors when migration/preconditions are missing.
- [x] **SCHE-03**: Schema upgrade path is idempotent (safe to run multiple times) and preserves existing prediction history.

### Monitoring Runtime Stability

- [x] **MON-01**: Monitoring cycles run without SQLAlchemy `OperationalError` caused by missing prediction columns.
- [x] **MON-02**: Prediction write/read paths in monitor + validator behave consistently across active freq/horizon settings.
- [x] **MON-03**: Critical monitor DB failures are surfaced with operator-actionable diagnostics (not silently swallowed).

### Timeline Experience (T2)

- [x] **TIM-01**: Timeline page renders recent and historical prediction records reliably from operational data.
- [x] **TIM-02**: Timeline clearly distinguishes pending vs realized predictions and shows realized outcome alignment.
- [x] **TIM-03**: Timeline exposes confidence and direction context per prediction event.
- [x] **TIM-04**: Timeline supports practical filtering (at least freq/horizon/date window) without breaking rendering.
- [x] **TIM-05**: Timeline supports improved visualization for predicted vs realized behavior (overlay/comparison view).

### GUI Compatibility & UX Hygiene

- [x] **GUI-01**: All Streamlit UI code replaces deprecated `use_container_width=True` with `width='stretch'`.
- [x] **GUI-02**: All Streamlit UI code replaces deprecated `use_container_width=False` with `width='content'`.
- [x] **GUI-03**: Primary GUI workflows execute without deprecation warnings related to width arguments.

### Quality & Verification

- [x] **QUAL-01**: Automated tests cover DB schema compatibility + monitor stability regressions for D1.
- [x] **QUAL-02**: Automated tests cover timeline data/render behavior for D2.
- [x] **QUAL-03**: Automated checks prevent reintroduction of `use_container_width` usage for D3.

### API / Surface Alignment

- [x] **API-01**: API and GUI timeline-consumed fields remain semantically aligned after schema/timeline changes.
- [x] **API-02**: Health/status surfaces accurately reflect readiness when schema compatibility is not satisfied.

## v2 Requirements

Deferred to later milestones.

### Advanced Analytics

- **ANLY-01**: Timeline supports model-vs-model comparative overlays.
- **ANLY-02**: Timeline supports exportable segment reports with annotation metadata.

### Operations

- **OPER-01**: Optional database backend migration path beyond SQLite for higher concurrency environments.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full dashboard redesign across all pages | Not required to satisfy D1/D2/D3 goals |
| New trading strategy/model objective overhaul | Current scope is stability + timeline UX, not model research |
| Multi-tenant auth/permissions redesign | No requirement signal for this milestone |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SCHE-01 | Phase 1 | Complete |
| SCHE-02 | Phase 1 | Complete |
| SCHE-03 | Phase 2 | Complete |
| MON-01 | Phase 3 | Complete |
| MON-02 | Phase 4 | Complete |
| MON-03 | Phase 3 | Complete |
| TIM-01 | Phase 5 | Complete |
| TIM-02 | Phase 5 | Complete |
| TIM-03 | Phase 6 | Complete |
| TIM-04 | Phase 6 | Complete |
| TIM-05 | Phase 6 | Complete |
| GUI-01 | Phase 7 | Complete |
| GUI-02 | Phase 7 | Complete |
| GUI-03 | Phase 7 | Complete |
| QUAL-01 | Phase 8 | Complete |
| QUAL-02 | Phase 8 | Complete |
| QUAL-03 | Phase 8 | Complete |
| API-01 | Phase 4 | Complete |
| API-02 | Phase 2 | Complete |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-24*
*Last updated: 2026-02-24 after Phase 8 execution*
