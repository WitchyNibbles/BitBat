# Roadmap: BitBat Reliability and Timeline Evolution

## Overview

This roadmap stabilizes the existing BitBat runtime first, then improves timeline capabilities, then locks in regression guardrails so reliability stays intact. The sequence prioritizes D1 (monitor DB stability), then D2 (timeline correctness and enhancement), then D3 (Streamlit compatibility hygiene), finishing with verification-ready quality gates.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Schema Contract Baseline** - Establish DB compatibility baseline and required columns. (Completed 2026-02-24)
- [x] **Phase 2: Migration Safety & Startup Readiness** - Make upgrades idempotent and enforce preflight checks. (Completed 2026-02-24)
- [x] **Phase 3: Monitor Runtime Error Elimination** - Remove monitor OperationalError paths and surface critical failures. (Completed 2026-02-24)
- [x] **Phase 4: Monitor Flow Consistency & API Alignment** - Align monitor prediction flow semantics across surfaces. (completed 2026-02-24)
- [x] **Phase 5: Timeline Core Reliability** - Restore stable timeline rendering with pending/realized correctness. (completed 2026-02-24)
- [ ] **Phase 6: Timeline UX Expansion (T2)** - Add richer confidence/filter/overlay timeline behavior.
- [ ] **Phase 7: Streamlit Compatibility Sweep** - Remove deprecated width API usage globally.
- [ ] **Phase 8: Regression Gates & Release Verification** - Lock in D1/D2/D3 via automated checks and end-to-end verification.

## Phase Details

### Phase 1: Schema Contract Baseline
**Goal**: Define and implement minimal schema compatibility for prediction runtime fields.
**Depends on**: Nothing (first phase)
**Requirements**: [SCHE-01, SCHE-02]
**Success Criteria** (what must be TRUE):
  1. Existing local databases can be brought to a schema state containing runtime-required prediction columns.
  2. Application startup performs schema compatibility checks before monitor cycles begin.
  3. Missing schema preconditions are surfaced with actionable operator-facing errors.
**Plans**: 3/3 plans complete

Plans:
- [x] 01-01-PLAN.md — Audit current model/table expectations vs on-disk SQLite states.
- [x] 01-02-PLAN.md — Implement compatibility/migration baseline for required prediction columns.
- [x] 01-03-PLAN.md — Add startup schema preflight and failure messaging.

### Phase 2: Migration Safety & Startup Readiness
**Goal**: Ensure migration path is repeatable, non-destructive, and deployment-safe.
**Depends on**: Phase 1
**Requirements**: [SCHE-03, API-02]
**Success Criteria** (what must be TRUE):
  1. Schema upgrade path is idempotent and preserves existing prediction history.
  2. Health/readiness signals reflect schema incompatibility accurately.
  3. Re-running migration or startup checks does not corrupt operational state.
**Plans**: 2 plans

Plans:
- [x] 02-01: Harden migration/idempotency behavior for repeated executions.
- [x] 02-02: Wire schema readiness into health/status surfaces.

### Phase 3: Monitor Runtime Error Elimination
**Goal**: Remove monitor DB runtime failures and improve critical-path failure visibility.
**Depends on**: Phase 2
**Requirements**: [MON-01, MON-03]
**Success Criteria** (what must be TRUE):
  1. Monitoring cycles run without `OperationalError` tied to missing prediction columns.
  2. Critical DB failures in monitor flow are no longer silently swallowed.
  3. Operator-visible diagnostics identify failing monitor step and remediation path.
**Plans**: 3 plans

Plans:
- [x] 03-01: Patch monitor/predictor DB interactions for robust missing-column handling.
- [x] 03-02: Refactor exception boundaries in monitor critical paths.
- [x] 03-03: Add structured logging and diagnostics for monitor DB faults.

### Phase 4: Monitor Flow Consistency & API Alignment
**Goal**: Align prediction field semantics between monitor writes and API/UI reads.
**Depends on**: Phase 3
**Requirements**: [MON-02, API-01]
**Success Criteria** (what must be TRUE):
  1. Prediction write/read behavior is consistent across active freq/horizon pairs.
  2. API fields consumed by timeline/UI are semantically aligned with monitor data.
  3. Cross-surface prediction records are internally consistent for confidence/direction/outcome fields.
**Plans**: 2 plans

Plans:
- [x] 04-01: Normalize monitor write semantics for prediction fields and dimensions.
- [x] 04-02: Align API/read models with updated prediction schema semantics.

### Phase 5: Timeline Core Reliability
**Goal**: Restore stable timeline rendering with accurate pending/realized behavior.
**Depends on**: Phase 4
**Requirements**: [TIM-01, TIM-02]
**Success Criteria** (what must be TRUE):
  1. Timeline reliably renders both recent and historical prediction records.
  2. Pending and realized predictions are visually and semantically distinguished.
  3. Timeline remains functional when optional fields are null or delayed.
**Plans**: 3 plans

Plans:
- [x] 05-01: Build/upgrade timeline read-model normalization layer. (Completed 2026-02-24)
- [x] 05-02: Repair timeline rendering for mixed pending/realized datasets. (Completed 2026-02-24)
- [x] 05-03: Validate timeline behavior across representative data fixtures. (Completed 2026-02-24)

### Phase 6: Timeline UX Expansion (T2)
**Goal**: Add richer timeline analysis context without regressing reliability.
**Depends on**: Phase 5
**Requirements**: [TIM-03, TIM-04, TIM-05]
**Success Criteria** (what must be TRUE):
  1. Timeline events show confidence and direction context clearly.
  2. Users can filter timeline by practical controls (freq/horizon/date window).
  3. Predicted vs realized behavior is compared via improved overlay/visualization.
**Plans**: 3 plans

Plans:
- [ ] 06-01: Implement confidence/direction timeline presentation improvements.
- [ ] 06-02: Implement and validate timeline filtering controls.
- [ ] 06-03: Implement predicted-vs-realized comparison overlays.

### Phase 7: Streamlit Compatibility Sweep
**Goal**: Remove deprecated Streamlit width API usage across GUI surfaces.
**Depends on**: Phase 6
**Requirements**: [GUI-01, GUI-02, GUI-03]
**Success Criteria** (what must be TRUE):
  1. `use_container_width=True` usages are replaced with `width='stretch'`.
  2. `use_container_width=False` usages are replaced with `width='content'`.
  3. Primary GUI workflows run without width deprecation warnings.
**Plans**: 2 plans

Plans:
- [ ] 07-01: Apply width API migration across Streamlit app/pages/widgets.
- [ ] 07-02: Verify warning-free execution across primary GUI interaction flows.

### Phase 8: Regression Gates & Release Verification
**Goal**: Enforce D1/D2/D3 through automated checks and final validation.
**Depends on**: Phase 7
**Requirements**: [QUAL-01, QUAL-02, QUAL-03]
**Success Criteria** (what must be TRUE):
  1. Automated tests cover schema compatibility + monitor stability for D1.
  2. Automated tests cover timeline rendering/data behavior for D2.
  3. Automated checks prevent reintroduction of deprecated Streamlit width APIs for D3.
**Plans**: 3 plans

Plans:
- [ ] 08-01: Add/expand D1-focused regression tests.
- [ ] 08-02: Add/expand D2-focused timeline behavior tests.
- [ ] 08-03: Add D3 guard checks and run end-to-end acceptance pass.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Schema Contract Baseline | 3/3 | Complete | 2026-02-24 |
| 2. Migration Safety & Startup Readiness | 2/2 | Complete | 2026-02-24 |
| 3. Monitor Runtime Error Elimination | 3/3 | Complete | 2026-02-24 |
| 4. Monitor Flow Consistency & API Alignment | 2/2 | Complete    | 2026-02-24 |
| 5. Timeline Core Reliability | 3/3 | Complete    | 2026-02-24 |
| 6. Timeline UX Expansion (T2) | 0/3 | Not started | - |
| 7. Streamlit Compatibility Sweep | 0/2 | Not started | - |
| 8. Regression Gates & Release Verification | 0/3 | Not started | - |
