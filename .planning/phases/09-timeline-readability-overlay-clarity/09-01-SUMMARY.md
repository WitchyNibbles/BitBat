---
phase: 09-timeline-readability-overlay-clarity
plan: "01"
subsystem: gui
tags: [timeline, readability, markers, plotly, regression]
requires: []
provides:
  - Readability-first marker rendering by semantic trace grouping instead of per-event traces
  - Preserved confidence/status hover semantics using grouped trace customdata payloads
  - Dense-history readability regression tests for timeline marker composition
affects: [timeline-rendering, timeline-tests]
tech-stack:
  added: []
  patterns: [semantic-marker-grouping, readability-proxy-testing]
key-files:
  created:
    - .planning/phases/09-timeline-readability-overlay-clarity/09-01-SUMMARY.md
  modified:
    - src/bitbat/gui/timeline.py
    - tests/gui/test_timeline.py
key-decisions:
  - "Grouped marker traces by direction/status to reduce default chart clutter while preserving per-point hover details."
  - "Added dense-data readability proxy tests (trace fan-out bound + full point coverage) so interpretability regressions fail fast."
patterns-established:
  - "Timeline markers can scale with data density by consolidating into semantic traces."
  - "Readability regressions are enforced through deterministic fixture-based assertions."
requirements-completed: [TIM-03]
duration: 15 min
completed: 2026-02-25
---

# Phase 09 Plan 01: Primary Timeline Readability Refactor

**Default timeline readability is improved by semantic marker grouping and dense-data regression coverage.**

## Performance

- **Duration:** 15 min
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- Replaced one-trace-per-event marker rendering with grouped marker traces by direction/status semantics.
- Preserved hover-level confidence/predicted/actual/status detail using grouped trace `customdata`.
- Added dense-history readability tests to guarantee bounded marker-trace fan-out while retaining full data-point coverage.

## Task Commits

1. **Task 1: Replace per-row marker trace creation with semantic marker grouping** - `af2514b` (feat)
2. **Task 2: Apply readability-first visual hierarchy in the primary timeline figure** - `23f8728` (style)
3. **Task 3: Add TIM-03 readability regressions for dense-history rendering** - `bb59496` (test)

## Files Created/Modified
- `.planning/phases/09-timeline-readability-overlay-clarity/09-01-SUMMARY.md` - Plan execution summary.
- `src/bitbat/gui/timeline.py` - Marker grouping helpers and readability-oriented figure styling.
- `tests/gui/test_timeline.py` - Updated marker semantics assertions and dense-readability regression tests.

## Decisions Made
- Use grouped marker traces to lower default chart complexity while maintaining semantic fidelity.
- Validate readability through deterministic proxy metrics rather than visual-only manual checks.

## Deviations from Plan

None - plan executed as intended.

## Issues Encountered
None.

## User Setup Required
None.

## Next Phase Readiness
- Wave 2 can now rework comparison behavior on top of a cleaner primary timeline baseline.

## Self-Check: PASSED

- `poetry run pytest tests/gui/test_timeline.py -q -k "timeline and (status_marker_semantics or confidence_exact_percent or readability)"` -> 3 passed
- `poetry run pytest tests/gui/test_timeline.py -q -k "timeline and (figure or marker or status)"` -> 8 passed
- `poetry run pytest tests/gui/test_timeline.py -q -k "timeline and (readability or dense or marker)"` -> 3 passed
- `poetry run pytest tests/gui/test_timeline.py -q` -> 24 passed

---
*Phase: 09-timeline-readability-overlay-clarity*
*Completed: 2026-02-25*
