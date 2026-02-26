---
phase: 14-baseline-models-and-retraining-cadence
plan: "02"
subsystem: retraining
tags: [rolling-windows, walk-forward, retrainer, continuous-training, cli]
requires:
  - phase: 14-baseline-models-and-retraining-cadence
    provides: Dual baseline family training and artifact persistence from Plan 14-01
provides:
  - Duration-driven rolling train/backtest window schedule generation
  - Window-aware retraining orchestration for autonomous retrainer and continuous trainer
  - CLI/config controls for train/backtest window cadence with deterministic defaults
affects: [dataset-splits, walk-forward, autonomous-retraining, cli, config, v1.2-phase14]
tech-stack:
  added: []
  patterns:
    - Window cadence is explicit and configurable instead of implicit one-shot spans
    - Retraining metadata carries window parameters for auditability and replay
key-files:
  created: []
  modified:
    - src/bitbat/dataset/splits.py
    - src/bitbat/model/walk_forward.py
    - src/bitbat/autonomous/continuous_trainer.py
    - src/bitbat/autonomous/retrainer.py
    - src/bitbat/cli.py
    - src/bitbat/config/default.yaml
    - tests/model/test_walk_forward.py
    - tests/autonomous/test_retrainer.py
    - tests/test_cli.py
key-decisions:
  - "Generated rolling CV windows as explicit --windows tuples so retraining uses deterministic folds with existing CLI compatibility."
  - "Applied explicit train/backtest window bars in continuous retraining to remove implicit 80/20 split assumptions."
patterns-established:
  - "Retraining command orchestration must include explicit window tuples and window config metadata."
  - "CLI rolling window options may auto-generate windows when explicit --windows are not provided."
requirements-completed: [MODL-02]
duration: 3 min
completed: 2026-02-26
---

# Phase 14 Plan 02: Rolling Retraining Cadence Summary

**Retraining and CV flows now run on explicit rolling train/backtest windows configured from CLI/config instead of implicit one-shot ranges.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-26T06:48:19Z
- **Completed:** 2026-02-26T06:51:31Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Added deterministic duration-based rolling window generation utilities for train/backtest cycles.
- Wired window cadence metadata into walk-forward and autonomous retraining paths for reproducibility.
- Added CLI/config controls to generate rolling windows without manual timestamp surgery while keeping legacy defaults intact.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add explicit rolling-window schedule utilities for train/backtest cycles** - `52c4b55` (feat)
2. **Task 2: Integrate windowed retraining cadence into autonomous retrainer and walk-forward paths** - `63269e4` (feat)
3. **Task 3: Expose window controls through config/CLI with backward-compatible defaults** - `8be39d4` (feat)

## Files Created/Modified

- `src/bitbat/dataset/splits.py` - duration-driven rolling window schedule generator.
- `src/bitbat/model/walk_forward.py` - fold-level window metadata output for downstream retraining audit.
- `src/bitbat/autonomous/continuous_trainer.py` - explicit train/backtest window-bar split and metadata persistence.
- `src/bitbat/autonomous/retrainer.py` - rolling CV window tuple generation and window-config metadata wiring.
- `src/bitbat/cli.py` - rolling window CLI options and auto-generated window scheduling.
- `src/bitbat/config/default.yaml` - default window cadence settings for model CV and retraining flows.
- `tests/model/test_walk_forward.py` - generated window behavior and fold-ordering regression coverage.
- `tests/autonomous/test_retrainer.py` - retrainer CV command window tuple regression coverage.
- `tests/test_cli.py` - CLI rolling-window generation regression coverage.

## Decisions Made

- Reused existing `--windows` CLI interface for retrainer orchestration so windowed cadence can ship without breaking command compatibility.
- Made generated rolling windows opt-in unless duration controls are provided, preserving old one-window behavior by default.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- Rolling window semantics are now shared across CV and retraining orchestration paths.
- Phase 14 Plan 03 can now attach regime/drift diagnostics to the same fold/window metadata.

## Self-Check: PASSED

- `poetry run pytest tests/model/test_walk_forward.py -q -k "window or rolling or fold"` -> 5 passed, 8 deselected
- `poetry run pytest tests/autonomous/test_retrainer.py tests/model/test_walk_forward.py -q -k "retrain or window or rolling"` -> 5 passed, 11 deselected
- `poetry run pytest tests/test_cli.py -q -k "model and (cv or train) and (window or rolling or retrain)"` -> 1 passed, 16 deselected

---
*Phase: 14-baseline-models-and-retraining-cadence*
*Completed: 2026-02-26*
