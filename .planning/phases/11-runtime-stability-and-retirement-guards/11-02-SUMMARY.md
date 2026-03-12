---
phase: 11-runtime-stability-and-retirement-guards
plan: "02"
subsystem: ui
tags: [streamlit, retired-routes, route-guards, stability]
requires:
  - phase: 11-runtime-stability-and-retirement-guards
    provides: Missing-confidence hardening for active home flow
provides:
  - Shared retirement notice component for legacy route entrypoints
  - Safe Backtest and Pipeline retired shells without brittle technical imports
  - Regression checks for legacy route guard behavior
affects: [retired-pages, legacy-routes, gui-regressions]
tech-stack:
  added: []
  patterns:
    - Shared helper-driven retirement UX across legacy page scripts
    - Lightweight route guard shells instead of loading advanced modules
key-files:
  created:
    - streamlit/retired_pages/_retired_notice.py
  modified:
    - streamlit/retired_pages/8_🎯_Backtest.py
    - streamlit/retired_pages/9_🔬_Pipeline.py
    - streamlit/retired_pages/README.md
    - tests/gui/test_complete_gui.py
key-decisions:
  - "Retired Backtest/Pipeline routes should render guidance immediately and avoid any advanced imports"
  - "Centralize retired-route navigation targets in one helper to prevent route drift"
patterns-established:
  - "Retired entry scripts should be thin wrappers around shared guidance helpers"
requirements-completed: [STAB-02, STAB-03, RET-01]
duration: 24 min
completed: 2026-02-25
---

# Phase 11 Plan 02: Legacy Route Retirement Guard Summary

**Shared retired-page guidance with lightweight Backtest/Pipeline shells that remove traceback-prone legacy imports**

## Performance

- **Duration:** 24 min
- **Started:** 2026-02-25T18:21:00Z
- **Completed:** 2026-02-25T18:45:00Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Added a shared retired-route notice helper with direct navigation to supported pages only.
- Replaced retired Backtest and Pipeline scripts with minimal wrappers that avoid advanced module imports.
- Added regression tests that lock legacy-route notice behavior and guard against reintroducing brittle imports.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add shared retirement-notice helper for legacy UI entrypoints** - `b67d524` (feat)
2. **Task 2: Replace Backtest/Pipeline retired scripts with safe legacy shells** - `68161da` (fix)
3. **Task 3: Add RET-01/STAB regression assertions for legacy route behavior** - `cffe749` (test)

**Plan metadata:** not committed (`.planning/` is gitignored in this repository)

## Files Created/Modified
- `streamlit/retired_pages/_retired_notice.py` - Shared retirement notice renderer and supported-page navigation actions.
- `streamlit/retired_pages/8_🎯_Backtest.py` - Retired shell that displays guidance without backtest-engine imports.
- `streamlit/retired_pages/9_🔬_Pipeline.py` - Retired shell that displays guidance without pipeline/evaluation imports.
- `streamlit/retired_pages/README.md` - Added Phase 11 retirement guard documentation.
- `tests/gui/test_complete_gui.py` - Added retired-route guard regressions for helper wiring and import safety.

## Decisions Made
- Kept retirement behavior explicit at script entrypoint level so legacy links fail safely and predictably.
- Centralized supported destinations in one helper to keep retired-route UX consistent.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Legacy route crash signatures are neutralized by retirement shells.
- Ready for Plan 11-03 completion-gate and release-contract wiring.

---
*Phase: 11-runtime-stability-and-retirement-guards*
*Completed: 2026-02-25*
