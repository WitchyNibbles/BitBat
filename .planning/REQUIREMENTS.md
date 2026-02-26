# Requirements: BitBat Autonomous Monitor Alignment and Metrics Integrity

**Defined:** 2026-02-26
**Status:** v1.3 requirements defined
**Core Value:** A reliable prediction system where operators can trust that monitoring outputs
correspond to real, active prediction flows for the configured runtime pair.

## v1.3 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### Runtime Alignment and Startup Safety

- [x] **ALGN-01**: Operator can start monitoring with an explicit config source and see the
  resolved `freq/horizon` pair before the first cycle runs.
- [x] **ALGN-02**: Operator receives a startup-blocking, actionable error when no model artifact
  exists for the resolved `freq/horizon` pair.
- [x] **ALGN-03**: Operator can verify heartbeat metadata includes config source, `freq`, and
  `horizon` for the running process.
- [x] **SCHE-04**: Operator can run monitor status/snapshot commands without runtime SQL errors
  because schema compatibility checks cover required `performance_snapshots` columns.

### Monitoring Signal Integrity

- [x] **MON-04**: Operator can distinguish "no prediction generated", "predictions pending
  realization", and "realized metrics available" states from one cycle summary payload.
- [x] **MON-05**: Operator can run monitor status and view total, unrealized, and realized
  prediction counts for the active pair.
- [x] **MON-06**: Operator can identify missing-model root cause from a single monitor cycle log
  line without inspecting traceback output.

### Quality and Regression Protection

- [ ] **QUAL-07**: Automated tests fail if monitor startup silently runs with unresolved
  runtime/model mismatch.
- [ ] **QUAL-08**: Automated tests fail if cycle/status payloads regress to ambiguous all-zero
  metrics without explicit state reasoning.
- [ ] **QUAL-09**: Automated tests fail if schema compatibility contracts omit runtime-required
  `performance_snapshots` columns.

## v1.4+ Requirements (Deferred)

Deferred until monitor alignment and metrics integrity are complete.

### Advanced Modeling

- **MICR-01**: Add optional microstructure/LOB feature pipeline for short-horizon signal
  experiments.
- **PORT-01**: Add multi-asset portfolio-level forecasting and allocation workflow.
- **EA-01**: Add EA-driven policy optimization under strict anti-overfitting controls.

## Out of Scope

| Feature | Reason |
|---------|--------|
| New model family expansion in v1.3 | Observability and runtime correctness are prerequisite gates |
| Full dashboard redesign | Milestone is monitor runtime and diagnostics hardening |
| Promotion-gate policy redesign | Existing v1.2 promotion safeguards remain authoritative |
| Multi-asset execution support | Deferred until single-asset monitor alignment is stable |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ALGN-01 | Phase 17 | Complete |
| ALGN-02 | Phase 17 | Complete |
| ALGN-03 | Phase 17 | Complete |
| SCHE-04 | Phase 17 | Complete |
| MON-04 | Phase 18 | Complete |
| MON-05 | Phase 18 | Complete |
| MON-06 | Phase 18 | Complete |
| QUAL-07 | Phase 19 | Pending |
| QUAL-08 | Phase 19 | Pending |
| QUAL-09 | Phase 19 | Pending |

**Coverage:**
- v1.3 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-26*
*Last updated: 2026-02-26 after Phase 17 execution and verification*
