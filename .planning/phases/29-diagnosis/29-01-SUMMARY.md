---
phase: 29-diagnosis
plan: 01
subsystem: testing

tags: [xgboost, sqlite3, pytest, diagnosis, tdd]

requires: []
provides:
  - "4 pipeline stage trace tests confirming reg:squarederror bug, direction bias, zero-return corruption, accuracy collapse"
  - "Structural test gating ROOT_CAUSE.md creation (RED until Plan 02)"
  - "tests/diagnosis/ package"
affects: [29-02, 30-fix]

tech-stack:
  added: []
  patterns:
    - "Diagnosis-first TDD: write tests confirming bugs before writing any fix code"
    - "Skip-graceful test pattern: tests skip cleanly when data/model files absent in CI"

key-files:
  created:
    - tests/diagnosis/__init__.py
    - tests/diagnosis/test_pipeline_stage_trace.py
    - tests/docs/test_root_cause_exists.py
  modified: []

key-decisions:
  - "Tests assert bugs ARE present (RED after Phase 30 fix, not now) — Phase 30 inverts these assertions"
  - "DB queries use sqlite3 stdlib only, no ORM, for maximum isolation from src/bitbat"
  - "test_root_cause_md_exists intentionally FAILS (RED) — gates Plan 02 and Phase 30"

patterns-established:
  - "Diagnostic TDD: test confirms bug exists first, becomes regression guard after fix"
  - "CI skip guards via pytest.skip when data/model files absent"

requirements-completed: [DIAG-01, DIAG-02]

duration: 1min
completed: 2026-03-08
---

# Phase 29 Plan 01: Diagnosis Test Harness Summary

**4 pipeline trace tests confirming reg:squarederror regression bug + direction bias + zero-return corruption + accuracy collapse, plus ROOT_CAUSE.md gate test in RED state**

## Performance

- **Duration:** 1 min
- **Started:** 2026-03-08T11:03:59Z
- **Completed:** 2026-03-08T11:05:18Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- tests/diagnosis/ package with 4 tests all PASSING — confirms all 4 bugs are live and real
- tests/docs/test_root_cause_exists.py in RED state — gates any Phase 30 fix code on ROOT_CAUSE.md being committed first
- All tests skip gracefully in CI where data/model files are absent

## Task Commits

Each task was committed atomically:

1. **Task 1: Pipeline stage trace tests** - `011b049` (test)
2. **Task 2: ROOT_CAUSE.md structural test** - `d5a47f3` (test)

## Files Created/Modified

- `tests/diagnosis/__init__.py` - Package init (empty)
- `tests/diagnosis/test_pipeline_stage_trace.py` - 4 diagnostic tests confirming bugs 1-3 + accuracy collapse
- `tests/docs/test_root_cause_exists.py` - 2 structural tests gating ROOT_CAUSE.md creation

## Decisions Made

- Tests assert bugs ARE present (they pass now, should fail after Phase 30 fix) — each docstring notes "After Phase 30 fix, this assertion should be inverted"
- DB queries use only sqlite3 stdlib — no src/bitbat imports — for full isolation
- test_root_cause_md_exists intentionally left in RED state — this is the correct behavior per DIAG-02

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 29 Plan 02 (ROOT_CAUSE.md creation) is unblocked — the structural test in RED state is ready to verify it
- When ROOT_CAUSE.md is committed with all required sections, both tests in test_root_cause_exists.py will PASS
- Phase 30 fix work must not begin until ROOT_CAUSE.md gates are GREEN

---
*Phase: 29-diagnosis*
*Completed: 2026-03-08*
