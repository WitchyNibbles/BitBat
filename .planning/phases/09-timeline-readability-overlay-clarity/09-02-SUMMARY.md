---
phase: 09-timeline-readability-overlay-clarity
plan: "02"
subsystem: gui
tags: [timeline, comparison, overlay, quick-start, defaults]
requires:
  - phase: 09-timeline-readability-overlay-clarity
    provides: Readability-first base timeline from 09-01
provides:
  - Explicit opt-in comparison figure (`build_timeline_comparison_figure`) decoupled from base timeline
  - Quick Start default comparison-off behavior for readability-first initial view
  - Regression tests covering comparison availability and default control behavior
affects: [timeline-rendering, quick-start, timeline-tests]
tech-stack:
  added: []
  patterns: [opt-in-comparison-mode, comparison-view-decoupling]
key-files:
  created:
    - .planning/phases/09-timeline-readability-overlay-clarity/09-02-SUMMARY.md
  modified:
    - src/bitbat/gui/timeline.py
    - streamlit/pages/0_Quick_Start.py
    - tests/gui/test_timeline.py
    - tests/gui/test_complete_gui.py
key-decisions:
  - "Keep primary timeline focused on event readability; expose return comparison as a dedicated opt-in figure."
  - "Set Quick Start comparison default to off to reduce cognitive load on first view."
patterns-established:
  - "Comparison analytics and event timeline can evolve independently through separate figure builders."
  - "Default UI state for heavy analytics views is tested explicitly in integration coverage."
requirements-completed: [TIM-05]
duration: 18 min
completed: 2026-02-25
---

# Phase 09 Plan 02: Opt-In Comparison Mode and Default Behavior

**Return comparison now uses an explicit opt-in path, preserving default timeline readability while maintaining TIM-05 capabilities.**

## Performance

- **Duration:** 18 min
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- Added `build_timeline_comparison_figure` to render predicted-vs-realized analytics in a dedicated comparison chart.
- Updated Quick Start timeline controls so comparison is off by default and enabled explicitly by the operator.
- Added timeline and integration regressions for comparison figure semantics and default control behavior.

## Task Commits

1. **Task 1: Implement explicit comparison rendering path decoupled from default event readability** - `add86a4` (feat)
2. **Task 2: Change Quick Start controls to readability-first defaults and explicit comparison intent** - `c211cf6` (feat)
3. **Task 3: Add integration regressions for default comparison state and toggled comparison behavior** - `ce48bb3` (test)

## Files Created/Modified
- `.planning/phases/09-timeline-readability-overlay-clarity/09-02-SUMMARY.md` - Plan execution summary.
- `src/bitbat/gui/timeline.py` - Added comparison figure API and shared overlay-trace composition helper.
- `streamlit/pages/0_Quick_Start.py` - Switched comparison default to off and rendered comparison as opt-in chart.
- `tests/gui/test_timeline.py` - Added comparison figure and default-overlay absence assertions.
- `tests/gui/test_complete_gui.py` - Added quick-start default-state and comparison availability integration tests.

## Decisions Made
- Retained legacy `show_overlay=True` path in `build_timeline_figure` for compatibility, while moving dashboard UX to dedicated comparison rendering.
- Used source-level integration assertion in `test_complete_gui` to lock default state in Quick Start.

## Deviations from Plan

None - executed as planned.

## Issues Encountered
None.

## User Setup Required
None.

## Next Phase Readiness
- Wave 3 can now focus on phase-level readability closure gates and D2 alignment.

## Self-Check: PASSED

- `poetry run pytest tests/gui/test_timeline.py -q -k "timeline and (overlay or pending or mismatch or comparison)"` -> 4 passed
- `poetry run pytest tests/gui/test_complete_gui.py -q -k "timeline and (overlay or default or filter or summary)"` -> 6 passed
- `poetry run pytest tests/gui/test_timeline.py tests/gui/test_complete_gui.py -q -k "timeline and (overlay or comparison or default)"` -> 7 passed
- `poetry run pytest tests/gui/test_timeline.py tests/gui/test_complete_gui.py -q` -> 49 passed

---
*Phase: 09-timeline-readability-overlay-clarity*
*Completed: 2026-02-25*
