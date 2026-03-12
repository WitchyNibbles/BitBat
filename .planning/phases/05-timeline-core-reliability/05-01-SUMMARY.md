---
phase: 05-timeline-core-reliability
plan: "01"
subsystem: ui
tags: [timeline, normalization, sqlite, streamlit, regression]
requires:
  - phase: 04-02
    provides: API and widget prediction semantics aligned to return/price contract
provides:
  - Canonical timeline read-model normalization for mixed legacy and return/price-first rows
  - Explicit status semantics (`prediction_status`, `is_realized`) for timeline consumers
  - Regression coverage for schema-tolerant timeline reads and status summary expectations
affects: [05-02, quick-start-timeline, timeline-rendering]
tech-stack:
  added: []
  patterns: [status-first-read-model, schema-tolerant-select, nullable-confidence]
key-files:
  created:
    - .planning/phases/05-timeline-core-reliability/05-01-SUMMARY.md
  modified:
    - src/bitbat/gui/timeline.py
    - tests/gui/test_timeline.py
    - tests/gui/test_complete_gui.py
key-decisions:
  - "Normalize timeline status at the read-model boundary so downstream UI code avoids ad hoc null checks."
  - "Keep confidence nullable when probabilities are absent to avoid misleading synthetic 0% values."
patterns-established:
  - "Timeline rows are normalized to a stable contract before rendering or metric summarization."
  - "Timeline query path is schema-tolerant (legacy columns missing) but deterministic in ordering and filtering."
requirements-completed: [TIM-01]
duration: 3 min
completed: 2026-02-24
---

# Phase 05 Plan 01: Timeline Read-Model Normalization Summary

**Timeline reads now normalize mixed prediction schemas into a deterministic status contract used consistently by tests and UI consumers.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-24T16:14:59+01:00
- **Completed:** 2026-02-24T16:17:07+01:00
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Added canonical normalization in `get_timeline_data` with explicit `prediction_status`, `is_realized`, normalized `correct`, and nullable `confidence` semantics.
- Hardened timeline query behavior for mixed schemas using table/column introspection, timestamp fallback, and deterministic ordering.
- Added regression tests for mixed legacy/new rows, legacy schema compatibility, timestamp fallback behavior, and status summary correctness.

## Task Commits

1. **Task 1: Implement canonical timeline row normalization helpers** - `5b1c221` (feat)
2. **Task 2: Harden timeline query path for mixed schema payload reliability** - `4608f46` (fix)
3. **Task 3: Add regression fixtures for normalized timeline data behavior** - `7379842` (test)

## Files Created/Modified
- `.planning/phases/05-timeline-core-reliability/05-01-SUMMARY.md` - Plan summary metadata and execution trace.
- `src/bitbat/gui/timeline.py` - Added normalized read-model, schema-tolerant SQL path, status metrics helper, and status-aware figure semantics.
- `tests/gui/test_timeline.py` - Added mixed-schema timeline fixture assertions and legacy compatibility coverage.
- `tests/gui/test_complete_gui.py` - Added timeline status summary regression checks aligned to normalized semantics.

## Decisions Made
- Status semantics (`pending`, `realized_correct`, `realized_wrong`) are computed centrally in the timeline module instead of inferred in page-level UI code.
- Query reliability favors fail-closed behavior (empty normalized frame) when timeline prerequisites are missing.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Timeline consumers can now rely on normalized status and confidence semantics.
- Wave 2 can focus on rendering behavior, sparse-price fallback, and Quick Start metric alignment.

## Self-Check: PASSED

- Verified key files from summary metadata exist on disk.
- Verified `git log --grep="05-01"` contains task commits.
- Verified timeline normalization regression command passed:
  - `poetry run pytest tests/gui/test_timeline.py tests/gui/test_complete_gui.py -q -k "timeline or prediction or status"`

---
*Phase: 05-timeline-core-reliability*
*Completed: 2026-02-24*
