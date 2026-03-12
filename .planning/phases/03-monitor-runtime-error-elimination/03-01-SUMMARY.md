---
phase: 03-monitor-runtime-error-elimination
plan: "01"
subsystem: database
tags: [sqlalchemy, monitor-runtime, schema-compat, diagnostics]
requires:
  - phase: 02-02
    provides: Schema readiness signals and non-mutating compatibility diagnostics
provides:
  - Shared monitor DB fault classification with remediation guidance
  - Predictor/validator runtime DB boundaries that raise structured monitor errors
  - Runtime regression coverage for missing-column monitor fault handling
affects: [03-02, 03-03, monitor-cli, monitoring-agent]
tech-stack:
  added: []
  patterns: [structured-monitor-db-failures, schema-remediation-diagnostics]
key-files:
  created:
    - .planning/phases/03-monitor-runtime-error-elimination/03-01-SUMMARY.md
  modified:
    - src/bitbat/autonomous/db.py
    - src/bitbat/autonomous/predictor.py
    - src/bitbat/autonomous/validator.py
    - tests/autonomous/test_db.py
    - tests/autonomous/test_validator.py
key-decisions:
  - "Centralize monitor DB failure classification in `classify_monitor_db_error` so predictor/validator do not hand-roll exception parsing."
  - "Attach schema remediation commands to runtime missing-column failures to keep MON-01 diagnostics actionable."
patterns-established:
  - "Monitor-critical DB paths now raise `MonitorDatabaseError` with step/detail/remediation metadata."
  - "Runtime DB error regressions are tested at both helper and validator boundary levels."
requirements-completed: [MON-01]
duration: 18 min
completed: 2026-02-24
---

# Phase 03 Plan 01: Monitor Runtime Error Elimination Summary

**Structured runtime DB fault classification for predictor/validator paths with schema-aware remediation diagnostics**

## Performance

- **Duration:** 18 min
- **Started:** 2026-02-24T14:48:00Z
- **Completed:** 2026-02-24T15:06:00Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Added a shared `MonitorDatabaseError` contract and `classify_monitor_db_error` helper in the autonomous DB layer.
- Hardened predictor and validator runtime DB boundaries to raise classified errors instead of uncategorized DB exceptions.
- Added regression tests verifying runtime missing-column faults expose monitor-step and remediation details.

## Task Commits

Task-level commits were not created in this run because the workspace already contained unrelated in-progress modifications.

## Files Created/Modified
- `.planning/phases/03-monitor-runtime-error-elimination/03-01-SUMMARY.md` - Plan execution summary and traceability metadata.
- `src/bitbat/autonomous/db.py` - Added monitor DB fault classification contract and schema-remediation helpers.
- `src/bitbat/autonomous/predictor.py` - Wrapped monitor-critical DB reads/writes with classified runtime error propagation.
- `src/bitbat/autonomous/validator.py` - Wrapped unrealized prediction fetch and realization update paths with classified DB failures.
- `tests/autonomous/test_db.py` - Added helper-level regression coverage for missing-column DB fault classification.
- `tests/autonomous/test_validator.py` - Added validator-level runtime DB fault surfacing test.

## Decisions Made
- Kept runtime DB classification logic close to `AutonomousDB` so all monitor components share one remediation strategy.
- Treated missing-column runtime failures as actionable schema compatibility diagnostics with explicit `--audit`/`--upgrade` guidance.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Agent, CLI, and script boundaries can now consume structured monitor DB failures instead of raw exceptions.
- Runtime DB fault metadata is available for cross-surface diagnostics in Phase 03-02 and 03-03.

## Self-Check: PASSED

- Verified key files exist.
- Verified targeted regression tests pass.
- No unresolved issues recorded.

---
*Phase: 03-monitor-runtime-error-elimination*
*Completed: 2026-02-24*
