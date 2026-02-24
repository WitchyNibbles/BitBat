---
phase: 04-monitor-flow-consistency-api-alignment
plan: "01"
subsystem: database
tags: [monitor, prediction-semantics, validator, sqlite]
requires:
  - phase: 03-03
    provides: Structured monitor DB diagnostics and critical-path error propagation
provides:
  - Predictor persistence now stores explicit predicted_return/predicted_price semantics
  - Validator correctness aligns to realized return sign when return forecasts exist
  - Regression coverage for monitor DB diagnostic surfacing at DB, validator, and agent boundaries
affects: [04-02, api-predictions, timeline-widgets, monitor-metrics]
tech-stack:
  added: []
  patterns: [explicit-prediction-semantics, sign-based-correctness, monitor-db-error-contract]
key-files:
  created:
    - .planning/phases/04-monitor-flow-consistency-api-alignment/04-01-SUMMARY.md
  modified:
    - src/bitbat/autonomous/db.py
    - src/bitbat/autonomous/predictor.py
    - src/bitbat/autonomous/validator.py
    - tests/autonomous/test_db.py
    - tests/autonomous/test_validator.py
    - tests/autonomous/test_agent_integration.py
key-decisions:
  - "Persist monitor outputs as predicted_return/predicted_price first-class fields to avoid implicit probability fallback semantics."
  - "Compute realized correctness from return-sign agreement when forecast return exists so validator and persistence stay coherent."
patterns-established:
  - "Predictor write path emits explicit return/price semantics per freq/horizon record."
  - "Validator realization correctness prefers return-sign parity and falls back to direction matching only when needed."
requirements-completed: [MON-02]
duration: 34 min
completed: 2026-02-24
---

# Phase 04 Plan 01: Monitor Flow Consistency Summary

**Canonical monitor persistence semantics now flow through predictor writes and validator realization correctness.**

## Performance

- **Duration:** 34 min
- **Started:** 2026-02-24T14:00:00Z
- **Completed:** 2026-02-24T14:34:00Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Predictor persistence now writes `predicted_return` and `predicted_price` explicitly instead of relying on inferred confidence fields.
- Validator correctness logic now aligns to sign agreement between predicted and realized returns for deterministic realization semantics.
- Monitor DB diagnostic coverage now validates surfaced remediation details through DB helper, validator, and agent integration paths.

## Task Commits

1. **Task 1: Introduce canonical prediction field normalization in autonomous DB boundary** - `6a08c80` (feat)
2. **Task 2: Align realization correctness persistence with validator semantics** - `27019cf` (fix)
3. **Task 3: Add regression tests for cross-dimension monitor consistency** - `b6a1547` (test)

## Files Created/Modified
- `.planning/phases/04-monitor-flow-consistency-api-alignment/04-01-SUMMARY.md` - Plan summary and traceability metadata.
- `src/bitbat/autonomous/db.py` - Added monitor DB fault classification and schema-remediation diagnostics used by monitor boundaries.
- `src/bitbat/autonomous/predictor.py` - Persisted explicit predicted return/price fields and wrapped DB boundaries with classified errors.
- `src/bitbat/autonomous/validator.py` - Aligned correctness semantics to return-sign agreement and propagated classified DB failures.
- `tests/autonomous/test_db.py` - Added runtime DB classification regression coverage.
- `tests/autonomous/test_validator.py` - Added validator DB failure surfacing regression coverage.
- `tests/autonomous/test_agent_integration.py` - Added monitor runtime DB failure propagation coverage.

## Decisions Made
- Treated return/price as the canonical monitor prediction contract so downstream API/GUI consumers can avoid fabricated confidence defaults.
- Preserved fallback direction comparison for legacy rows without `predicted_return` to keep historical compatibility intact.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- API and GUI surfaces can now consume explicit monitor semantics without inferring confidence from legacy probability fields.
- Cross-surface alignment work can proceed on top of deterministic correctness and persistence behavior.

## Self-Check: PASSED

- Verified key files from summary metadata exist on disk.
- Verified `git log --grep="04-01"` contains plan task commits.
- Verified targeted monitor consistency test suite passes.

---
*Phase: 04-monitor-flow-consistency-api-alignment*
*Completed: 2026-02-24*
