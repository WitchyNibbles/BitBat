---
phase: 11-runtime-stability-and-retirement-guards
plan: "01"
subsystem: ui
tags: [streamlit, sqlite, schema-compat, confidence]
requires:
  - phase: 10-supported-surface-pruning
    provides: Stable five-view runtime surface for targeted hardening
provides:
  - Schema-tolerant latest prediction payload with optional confidence handling
  - Home rendering guardrails for partial prediction data
  - Regression tests for missing-confidence and legacy-schema paths
affects: [streamlit-home, widgets, gui-regressions]
tech-stack:
  added: []
  patterns:
    - Dynamic column selection from sqlite schema metadata
    - Defensive optional-field rendering in Streamlit home cards
key-files:
  created: []
  modified:
    - src/bitbat/gui/widgets.py
    - streamlit/app.py
    - tests/gui/test_widgets.py
    - tests/gui/test_complete_gui.py
key-decisions:
  - "Use PRAGMA-based dynamic select construction so legacy prediction schemas remain readable"
  - "Treat confidence as optional UI context and render n/a fallback when unavailable"
patterns-established:
  - "Prediction payloads should include stable keys even when source columns are absent"
requirements-completed: [STAB-01]
duration: 31 min
completed: 2026-02-25
---

# Phase 11 Plan 01: Home Missing-Confidence Hardening Summary

**Schema-tolerant prediction payload handling with confidence fallback rendering for Streamlit home cards**

## Performance

- **Duration:** 31 min
- **Started:** 2026-02-25T17:50:00Z
- **Completed:** 2026-02-25T18:21:00Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- Reworked latest-prediction loading to adapt to available SQLite columns instead of assuming a fixed schema.
- Removed direct `latest_pred["confidence"]` access in home rendering and added readable fallback copy.
- Added regression tests for confidence derivation and legacy prediction schemas missing confidence-related columns.

## Task Commits

Each task was committed atomically:

1. **Task 1: Make latest prediction payload include safe confidence fallback semantics** - `2f04657` (fix)
2. **Task 2: Guard home rendering against missing confidence and other optional fields** - `6c1bb1a` (fix)
3. **Task 3: Add STAB-01 regression tests for missing-confidence paths** - `278354d` (test)

**Plan metadata:** not committed (`.planning/` is gitignored in this repository)

## Files Created/Modified
- `src/bitbat/gui/widgets.py` - Added schema-aware latest prediction query construction and confidence fallback derivation.
- `streamlit/app.py` - Replaced hard key indexing with safe accessors and fallback confidence copy.
- `tests/gui/test_widgets.py` - Added confidence and legacy-schema coverage for `get_latest_prediction`.
- `tests/gui/test_complete_gui.py` - Added source-contract checks for safe confidence rendering behavior.

## Decisions Made
- Used dynamic SQL projection based on `PRAGMA table_info(prediction_outcomes)` to avoid query failures on older schemas.
- Kept `confidence` optional in UI payload and presentation to prevent startup/home crashes when missing.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Home crash signature (`KeyError: 'confidence'`) is blocked by payload + UI guards.
- Ready for Plan 11-02 legacy route retirement guard implementation.

---
*Phase: 11-runtime-stability-and-retirement-guards*
*Completed: 2026-02-25*
