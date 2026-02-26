---
phase: 14-baseline-models-and-retraining-cadence
plan: "01"
subsystem: model
tags: [xgboost, random-forest, baseline, persistence, cli]
requires:
  - phase: 13-data-and-label-contract-upgrade
    provides: Leakage-safe feature/target dataset contract for model training
provides:
  - Shared tree-baseline training interface for XGBoost and RandomForest
  - Family-aware artifact persistence with stable canonical paths
  - CLI model train/cv family selection across xgb, random_forest, and both
affects: [model-training, model-persistence, cli, v1.2-phase14]
tech-stack:
  added: []
  patterns:
    - Baseline families share one feature/target contract and seed semantics
    - Model artifacts are family-tagged with stable family-specific filenames
key-files:
  created: []
  modified:
    - src/bitbat/model/train.py
    - src/bitbat/model/persist.py
    - src/bitbat/cli.py
    - tests/model/test_train.py
    - tests/model/test_persist.py
    - tests/test_cli.py
key-decisions:
  - "Preserved default xgb behavior while adding explicit random_forest and both-family execution paths."
  - "Kept top-level cv_summary keys for backward compatibility and added per-family aggregates under family_metrics."
patterns-established:
  - "Baseline families must persist through canonical family-aware artifact helpers."
  - "CLI family selection defaults to config model.baseline_family, falling back to xgb."
requirements-completed: [MODL-01]
duration: 4 min
completed: 2026-02-26
---

# Phase 14 Plan 01: Dual Baseline Families Summary

**XGBoost and RandomForest now train, persist, and run CV from one shared baseline contract with family-selectable CLI flows.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-26T06:41:36Z
- **Completed:** 2026-02-26T06:45:35Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added unified tree-baseline training APIs that support `xgb` and `random_forest` with deterministic seed behavior.
- Added family-aware persistence helpers with stable artifact paths and metadata sidecars for reproducible model comparisons.
- Extended `bitbat model cv` and `bitbat model train` with `--family` selection and per-family CV reporting.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add RandomForest baseline training alongside XGBoost under a shared interface** - `bb2a3a8` (feat)
2. **Task 2: Implement family-aware artifact persistence for comparable baseline outputs** - `044acd7` (feat)
3. **Task 3: Wire CLI model family selection for baseline train/cv entrypoints** - `ff9e17a` (feat)

## Files Created/Modified

- `src/bitbat/model/train.py` - shared baseline training interface and deterministic random-forest trainer.
- `src/bitbat/model/persist.py` - family-aware save/load helpers, stable artifact path helpers, and metadata sidecar writes.
- `src/bitbat/cli.py` - model `--family` controls and per-family CV/train execution.
- `tests/model/test_train.py` - baseline family training and deterministic random-forest coverage.
- `tests/model/test_persist.py` - xgb/random_forest persistence roundtrip and canonical artifact-path helper coverage.
- `tests/test_cli.py` - CLI family selection tests for CV and train commands.

## Decisions Made

- Kept default xgb train/cv paths intact for backward compatibility, with optional family expansion via `--family`.
- Standardized family comparison output in `metrics/cv_summary.json` under `family_metrics` while retaining top-level `average_rmse` and `average_mae`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- Baseline family selection and artifact comparability are in place for rolling-window retraining work.
- Phase 14 Plan 02 can now focus on explicit retraining/backtest window cadence integration.

## Self-Check: PASSED

- `poetry run pytest tests/model/test_train.py -q -k "xgb or random or baseline"` -> 4 passed
- `poetry run pytest tests/model/test_persist.py tests/model/test_train.py -q -k "persist or random or xgb"` -> 7 passed
- `poetry run pytest tests/test_cli.py -q -k "model and (train or cv) and (baseline or random or family)"` -> 2 passed, 14 deselected

---
*Phase: 14-baseline-models-and-retraining-cadence*
*Completed: 2026-02-26*
