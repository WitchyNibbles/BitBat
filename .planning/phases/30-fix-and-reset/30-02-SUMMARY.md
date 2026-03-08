---
phase: 30-fix-and-reset
plan: 02
subsystem: testing
tags: [cli, click, system-reset, diagnostic-tests, tdd]

# Dependency graph
requires:
  - phase: 30-01
    provides: Three root-cause bugs fixed in train.py, infer.py, validator.py
provides:
  - Four diagnostic tests inverted from "bug present" to "bug fixed" assertions
  - bitbat system reset --yes CLI command deleting data/, models/, autonomous.db
  - Three unit tests for reset command (happy path, abort on n, missing dirs)
affects:
  - 30-03
  - phase-31

# Tech tracking
tech-stack:
  added: []
  patterns: [TDD RED-GREEN for CLI commands, skip guards for live-artifact tests]

key-files:
  created: []
  modified:
    - tests/diagnosis/test_pipeline_stage_trace.py
    - src/bitbat/cli.py
    - tests/test_cli.py

key-decisions:
  - "30-02: Diagnostic tests guard via pytest.skip when DB/model absent; pre-fix DB data causes failure — expected until operator runs system reset"
  - "30-02: system reset deletes data/ and models/ via shutil.rmtree; autonomous.db deleted separately only if outside data_dir"
  - "30-02: Path.is_relative_to() used (Python 3.11 project) for autonomous.db containment check"

patterns-established:
  - "Skip guards on live-artifact tests: skip if file absent, fail if file present with wrong state — this is correct test behavior"
  - "CLI lifecycle commands grouped under system group with explicit --yes flag for destructive operations"

requirements-completed: [FIXR-02]

# Metrics
duration: 8min
completed: 2026-03-08
---

# Phase 30 Plan 02: Fix Validation and System Reset Summary

**Four diagnostic tests inverted to "bug fixed" assertions; bitbat system reset --yes CLI command deletes data/, models/, and autonomous.db for clean-slate restart**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-08T17:50:00Z
- **Completed:** 2026-03-08T17:58:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- All four Phase 29 diagnostic tests renamed and assertions inverted from "bug present" to "bug fixed" (multi:softprob, flat_count > 0, zero_count < 50, accuracy > 0.33)
- Added `bitbat system reset --yes` command under new `system` CLI group — deletes data/, models/, autonomous.db
- TDD: wrote three failing tests first, then implemented command, confirmed GREEN; 649 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Invert Phase 29 diagnostic tests** - `f199536` (test)
2. **Task 2: Add system reset CLI command and tests** - `e4912c6` (feat)

## Files Created/Modified
- `tests/diagnosis/test_pipeline_stage_trace.py` - Four tests inverted to "bug fixed" assertions with updated names and docstrings
- `src/bitbat/cli.py` - Added system group and system_reset command using shutil.rmtree
- `tests/test_cli.py` - Three new reset tests: happy path, abort on n, missing dirs

## Decisions Made
- Diagnostic tests skip when model/DB absent but fail when DB contains pre-fix predictions — this is correct behavior; operator runs `system reset` before retraining to clear stale DB
- `Path.is_relative_to()` used for autonomous.db containment check (Python 3.11, safe to use)
- `shutil.rmtree(ignore_errors=True)` for resilient directory deletion

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `test_serving_direction_is_balanced` fails on CI if `data/autonomous.db` exists with pre-fix predictions (no flat class). This is expected: the DB was written before Phase 30 fixes. Operator must run `bitbat system reset --yes` to clear it before retraining. The test is correct.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- FIXR-02 satisfied: system reset command ready for operator use
- Operator should run `bitbat system reset --yes` then retrain to populate fresh predictions
- Phase 30 Plan 03 can now add DB-dependent tests against a freshly-trained system

---
*Phase: 30-fix-and-reset*
*Completed: 2026-03-08*
