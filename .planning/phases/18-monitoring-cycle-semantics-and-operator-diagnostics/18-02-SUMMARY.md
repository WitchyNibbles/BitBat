---
phase: 18-monitoring-cycle-semantics-and-operator-diagnostics
plan: "02"
subsystem: monitor-status-lifecycle-counts
tags: [monitor-status, autonomous-db, lifecycle-counts, operator-diagnostics]
requires:
  - phase: 18-monitoring-cycle-semantics-and-operator-diagnostics
    provides: Cycle-state semantics and no-prediction diagnostics contract from Plan 01
provides:
  - Pair-scoped DB helper for total/unrealized/realized prediction lifecycle counts
  - `bitbat monitor status` output lines with explicit lifecycle count semantics
  - Regression coverage for active-pair and no-snapshot status count behavior
affects: [autonomous-db, monitor-cli, autonomous-tests, cli-tests, v1.3-phase18]
tech-stack:
  added: []
  patterns:
    - Monitor status count semantics must come from one canonical DB helper
    - Status output must remain explicit even when no performance snapshot exists
key-files:
  created: []
  modified:
    - src/bitbat/autonomous/db.py
    - src/bitbat/cli.py
    - tests/autonomous/test_db.py
    - tests/test_cli.py
key-decisions:
  - Added a pair-scoped `AutonomousDB.get_prediction_counts` helper as the canonical source for lifecycle counts.
  - Kept `Pending validations` in status output, but now defined it explicitly as unrealized prediction count.
  - Added active-pair and zero-count regressions to prevent cross-pair leakage and no-snapshot ambiguity.
patterns-established:
  - Lifecycle count semantics should be pair-scoped by `freq/horizon` for all operator status surfaces.
  - Status regressions should include mixed-pair fixtures and zero-snapshot scenarios.
requirements-completed: [MON-05]
duration: 3 min
completed: 2026-02-26
---

# Phase 18 Plan 02: Monitor Status Lifecycle Counts Summary

**Monitor status now reports explicit total, unrealized, and realized prediction counts for the selected runtime pair.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-26T14:27:57+01:00
- **Completed:** 2026-02-26T14:30:08+01:00
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Implemented a canonical pair-scoped DB count helper for prediction lifecycle states.
- Updated `monitor status` output to print explicit total/unrealized/realized counts from DB data.
- Added regression coverage for mixed-pair filtering and no-snapshot count visibility.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add pair-scoped prediction count helper(s) to autonomous DB layer** - `3e40dbb` (feat)
2. **Task 2: Update monitor status CLI to print total/unrealized/realized counts** - `d62160e` (feat)
3. **Task 3: Add MON-05 regression tests for active-pair count semantics** - `77d7d6d` (test)

## Files Created/Modified

- `src/bitbat/autonomous/db.py` - added `get_prediction_counts` helper returning pair-scoped lifecycle counts.
- `src/bitbat/cli.py` - monitor status now surfaces total/unrealized/realized counts plus pending validations from DB helper output.
- `tests/autonomous/test_db.py` - added pair-scoped and empty-pair lifecycle count regression tests.
- `tests/test_cli.py` - added monitor status regressions for no-snapshot and active-pair filtered count outputs.

## Decisions Made

- Count semantics are sourced from DB rows for the selected pair rather than inferred from snapshots.
- Monitor status remains coherent without snapshots by printing count lines before snapshot diagnostics.
- Active-pair filtering is test-locked to prevent accidental cross-pair aggregation.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- Plan 18-03 can now reuse stable reason/count semantics to propagate concise root-cause diagnostics to heartbeat and operator logs.
- Phase verification can assert MON-04 + MON-05 contracts through both payload and CLI status surfaces.

## Self-Check: PASSED

- `poetry run pytest tests/autonomous/test_db.py -q -k "count or unrealized or realized or monitor_status"` -> 2 passed, 4 deselected
- `poetry run pytest tests/test_cli.py -q -k "monitor_status and (total or unrealized or realized or snapshot)"` -> 3 passed, 27 deselected
- `poetry run pytest tests/autonomous/test_db.py tests/test_cli.py -q -k "monitor and status and (count or unrealized or realized)"` -> 4 passed, 34 deselected

---
*Phase: 18-monitoring-cycle-semantics-and-operator-diagnostics*
*Completed: 2026-02-26*
