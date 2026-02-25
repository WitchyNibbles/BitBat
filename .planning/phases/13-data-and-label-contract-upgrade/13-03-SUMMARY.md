---
phase: 13-data-and-label-contract-upgrade
plan: "03"
subsystem: labeling
tags: [triple-barrier, dataset-mode, cli, contracts]
requires:
  - phase: 13-data-and-label-contract-upgrade
    provides: Shared return/direction label contract from Plan 13-02
provides:
  - Deterministic triple-barrier labeling helpers (take-profit, stop-loss, timeout)
  - Optional `build_xy` label-mode wiring for triple-barrier outputs
  - CLI options to explicitly enable triple-barrier mode without changing defaults
affects: [labeling, dataset-build, feature-contracts, cli, v1.2-phase13]
tech-stack:
  added: []
  patterns:
    - Dataset label semantics are explicit through `label_mode` instead of implicit assumptions
    - Contract validation enforces mode-specific allowed label vocabularies
key-files:
  created:
    - src/bitbat/labeling/triple_barrier.py
    - tests/labeling/test_triple_barrier.py
  modified:
    - src/bitbat/labeling/__init__.py
    - src/bitbat/dataset/build.py
    - src/bitbat/contracts.py
    - src/bitbat/cli.py
    - tests/dataset/test_build_xy.py
    - tests/test_cli.py
key-decisions:
  - "Kept `return_direction` as the default label mode and added `triple_barrier` as an explicit opt-in mode."
  - "Reused dataset schema columns (`label`, `r_forward`) while validating label vocabulary by `label_mode` to avoid default-path breakage."
  - "Exposed barrier thresholds through CLI flags so operators can run trading-aligned experiments without code changes."
patterns-established:
  - "Optional labeling modes must preserve baseline defaults and be toggled explicitly from CLI."
  - "Triple-barrier labels are deterministic first-hit outcomes with timeout fallback under the configured horizon."
requirements-completed: [LABL-01]
duration: 11 min
completed: 2026-02-25
---

# Phase 13 Plan 03: Optional Triple-Barrier Label Mode Summary

**Triple-barrier labels are now available as an optional dataset mode, while default return/direction behavior remains stable.**

## Performance

- **Duration:** 11 min
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Implemented deterministic triple-barrier label generation with explicit take-profit/stop-loss/timeout outcomes.
- Added optional `label_mode` routing in `build_xy` and mode-aware contract checks for allowed label classes.
- Wired `features build` CLI options for barrier mode activation and threshold control.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement deterministic triple-barrier labeling utilities** - `27f621a` (feat)
2. **Task 2: Add optional triple-barrier mode to dataset build path** - `8071736` (feat)
3. **Task 3: Wire CLI control for optional barrier labeling mode** - `ee2d9ca` (feat)

## Files Created/Modified

- `src/bitbat/labeling/triple_barrier.py` - triple-barrier event labeling implementation.
- `tests/labeling/test_triple_barrier.py` - deterministic coverage for event paths and first-hit ordering.
- `src/bitbat/labeling/__init__.py` - package exports for triple-barrier helpers.
- `src/bitbat/dataset/build.py` - optional `label_mode` and barrier threshold integration.
- `src/bitbat/contracts.py` - mode-aware label vocabulary validation.
- `src/bitbat/cli.py` - `features build` options for label mode and barrier thresholds.
- `tests/dataset/test_build_xy.py` - compatibility test for default vs triple-barrier dataset outputs.
- `tests/test_cli.py` - CLI tests for default and explicit triple-barrier mode routing.

## Decisions Made

- Chose `return_direction` and `triple_barrier` as explicit mode names with default behavior unchanged.
- Preserved existing dataset/training interfaces by keeping `build_xy` return shape and feature columns stable.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- Phase 13 requirement set is complete (DATA-01, DATA-02, LABL-01).
- v1.2 can proceed to Phase 14 baseline model and retraining cadence work.

## Self-Check: PASSED

- `poetry run pytest tests/labeling/test_triple_barrier.py -q` -> 4 passed
- `poetry run pytest tests/dataset/test_build_xy.py -q -k "triple or barrier or label_mode or compatibility"` -> 1 passed, 1 deselected
- `poetry run pytest tests/test_cli.py -q -k "features and (label or triple or barrier)"` -> 2 passed, 12 deselected

---
*Phase: 13-data-and-label-contract-upgrade*
*Completed: 2026-02-25*
