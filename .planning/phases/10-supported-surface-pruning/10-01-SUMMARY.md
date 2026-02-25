---
phase: 10-supported-surface-pruning
plan: "01"
subsystem: gui
tags: [streamlit, ui-surface, runtime-scope, simplification]
requires:
  - phase: 10-supported-surface-pruning
    provides: Supported-page-only runtime surface contract
provides:
  - Runtime surface pruned to five supported operator views
  - Retired advanced pages preserved outside active runtime page discovery
  - Runtime scope tests locking simplified page inventory and width literal contract
affects: [streamlit-pages, gui-runtime-contracts]
tech-stack:
  added: []
  patterns: [runtime-surface-pruning, test-locked-ui-inventory]
key-files:
  created:
    - .planning/phases/10-supported-surface-pruning/10-01-SUMMARY.md
    - streamlit/retired_pages/README.md
  modified:
    - tests/gui/test_streamlit_width_compat.py
  moved:
    - streamlit/pages/5_🔔_Alerts.py -> streamlit/retired_pages/5_🔔_Alerts.py
    - streamlit/pages/6_📊_Analytics.py -> streamlit/retired_pages/6_📊_Analytics.py
    - streamlit/pages/7_📅_History.py -> streamlit/retired_pages/7_📅_History.py
    - streamlit/pages/8_🎯_Backtest.py -> streamlit/retired_pages/8_🎯_Backtest.py
    - streamlit/pages/9_🔬_Pipeline.py -> streamlit/retired_pages/9_🔬_Pipeline.py
key-decisions:
  - "Moved non-core pages out of streamlit/pages to preserve code while removing them from runtime discovery."
  - "Replaced broad runtime page-count assertions with explicit five-view inventory checks."
patterns-established:
  - "Runtime surface is now contract-tested as an explicit page inventory, not an open-ended page count."
requirements-completed: [UIF-01, UIF-03]
duration: 18 min
completed: 2026-02-25
---

# Phase 10 Plan 01: Supported Surface Pruning

**BitBat runtime page discovery now focuses on the five supported operator views.**

## Performance

- **Duration:** 18 min
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- Pruned non-core Streamlit pages (`Alerts`, `Analytics`, `History`, `Backtest`, `Pipeline`) from active `streamlit/pages` runtime discovery.
- Preserved retired pages in `streamlit/retired_pages/` for reference/future reintroduction.
- Updated runtime scope compatibility tests to enforce explicit five-view inventory and compatible width literal usage.

## Task Commits

1. **Task 1: Implement supported-page inventory and prune non-core runtime pages** - `db2a1fb` (feat)
2. **Task 2: Update runtime scope checks to reflect the supported five-view contract** - `31b5ba2` (test)
3. **Task 3: Preserve Streamlit width compatibility behavior after surface pruning** - `31b5ba2` (test)

## Files Created/Modified
- `.planning/phases/10-supported-surface-pruning/10-01-SUMMARY.md` - Plan execution summary.
- `streamlit/retired_pages/README.md` - Notes on retired Streamlit pages.
- `streamlit/retired_pages/*.py` - Moved non-core pages out of active runtime discovery.
- `tests/gui/test_streamlit_width_compat.py` - Runtime surface contract updated for five supported views.

## Decisions Made
- Keep retired pages in-repo but out of `streamlit/pages` so runtime navigation is simplified without deleting legacy code.
- Treat supported runtime page inventory as an explicit contract in tests.

## Deviations from Plan

- Task 2 and Task 3 landed in the same test commit because width-literal checks are coupled with runtime-scope assertions after page pruning.

## Issues Encountered
- Existing runtime width literal assertion required both `stretch` and `content`; after pruning, active pages only used `stretch`.
- Resolved by changing the assertion to enforce allowed literals subset + required `stretch`.

## User Setup Required
None.

## Next Phase Readiness
- Wave 2 (Plan 10-02) can proceed to align home navigation/copy with simplified supported surface.

## Self-Check: PASSED

- `poetry run pytest tests/gui/test_streamlit_width_compat.py -q` -> 4 passed

---
*Phase: 10-supported-surface-pruning*
*Completed: 2026-02-25*
