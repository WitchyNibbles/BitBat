---
phase: 15-cost-aware-walk-forward-evaluation
plan: "01"
subsystem: evaluation
tags: [walk-forward, purge, embargo, leakage-control, cli]
requires:
  - phase: 14-baseline-models-and-retraining-cadence
    provides: Rolling walk-forward windows and fold diagnostics metadata
provides:
  - Purge-aware walk-forward split generation with deterministic horizon handling
  - Fold metadata that records purge/embargo settings for leakage audits
  - CLI/config controls for purge and embargo behavior in model CV runs
affects: [dataset-splits, model-cv, walk-forward-evaluation, config, v1.2-phase15]
tech-stack:
  added: []
  patterns:
    - Leakage controls are first-class split parameters surfaced from CLI/config
    - Fold metadata carries split-control settings for deterministic diagnostics
key-files:
  created: []
  modified:
    - src/bitbat/dataset/splits.py
    - src/bitbat/model/walk_forward.py
    - src/bitbat/cli.py
    - src/bitbat/config/default.yaml
    - tests/dataset/test_splits.py
    - tests/model/test_walk_forward.py
    - tests/test_cli.py
key-decisions:
  - "Modeled purge and embargo as explicit bar exclusions before each test window for deterministic overlap control."
  - "Kept backward compatibility by defaulting CLI embargo behavior to 1 bar when no override is provided."
patterns-established:
  - "Model CV leakage controls must resolve from CLI overrides first, then model.cv defaults."
  - "Walk-forward fold metadata must include purge and embargo settings for auditability."
requirements-completed: [EVAL-01]
duration: 4 min
completed: 2026-02-26
---

# Phase 15 Plan 01: Purge/Embargo Walk-Forward Controls Summary

**Walk-forward split generation now enforces explicit purge/embargo leakage controls and exposes deterministic configuration through model CV CLI/config paths.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-26T07:08:30Z
- **Completed:** 2026-02-26T07:12:46Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Added purge-aware split controls with optional horizon-derived purge sizing for overlapping label windows.
- Propagated leakage-control metadata (embargo/purge bars) into fold-level walk-forward outputs.
- Added model CV options/config defaults for purge/embargo/label-horizon controls with regression-verified passthrough.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add purge-aware split generation utilities for overlapping label horizons** - `47e2bc2` (feat)
2. **Task 2: Integrate purge/embargo semantics into walk-forward evaluation execution** - `a07bd92` (feat)
3. **Task 3: Expose purge/embargo controls through model CV CLI and defaults** - `4c9d165` (feat)

## Files Created/Modified

- `src/bitbat/dataset/splits.py` - purge/embargo split controls and horizon-aware purge derivation.
- `src/bitbat/model/walk_forward.py` - fold window metadata now includes leakage-control settings.
- `src/bitbat/cli.py` - model CV accepts and resolves purge/embargo/label-horizon options.
- `src/bitbat/config/default.yaml` - default model CV leakage-control settings.
- `tests/dataset/test_splits.py` - purge and horizon-based split leakage regression tests.
- `tests/model/test_walk_forward.py` - fold metadata leakage-control contract test.
- `tests/test_cli.py` - CLI leakage-control passthrough regression test.

## Decisions Made

- Used positional bar exclusions from test-window boundaries to keep purge/embargo behavior deterministic across schedules.
- Preserved existing behavior by defaulting to one-bar embargo when no explicit configuration is provided.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- EVAL-01 split-control requirements are now implemented and regression covered.
- Phase 15 Plan 02 can build cost attribution on top of these deterministic fold semantics.

## Self-Check: PASSED

- `poetry run pytest tests/dataset/test_splits.py -q -k "walk_forward and (embargo or purge or leakage)"` -> 4 passed
- `poetry run pytest tests/model/test_walk_forward.py -q -k "walk_forward or fold or embargo or purge"` -> 14 passed
- `poetry run pytest tests/test_cli.py -q -k "model and cv and (purge or embargo or walk_forward)"` -> 1 passed, 18 deselected

---
*Phase: 15-cost-aware-walk-forward-evaluation*
*Completed: 2026-02-26*
