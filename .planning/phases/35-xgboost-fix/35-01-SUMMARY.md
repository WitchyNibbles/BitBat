---
phase: 35-xgboost-fix
plan: 01
subsystem: model-core
tags: [xgboost, classification, walk-forward, optimization, tdd]

requires:
  - phase: 30-fix-and-reset
    provides: Corrected XGBoost training/inference baseline
provides:
  - classification-aware walk-forward evaluation for label targets
  - classification-aware hyperparameter optimization and PR-AUC summaries
  - reusable multiclass probability metrics with stable label ordering
affects: [35-02, model-selection, evaluation]

tech-stack:
  added: []
  patterns:
    - adaptive classification-vs-regression model evaluation based on target shape
    - probability-first XGBoost metrics for label-driven selection flows

key-files:
  created: []
  modified:
    - src/bitbat/model/evaluate.py
    - src/bitbat/model/walk_forward.py
    - src/bitbat/model/optimize.py
    - tests/model/test_walk_forward.py
    - tests/model/test_optimize.py

key-decisions:
  - "Walk-forward and optimizer paths detect label targets and switch to multi:softprob automatically"
  - "Regression fallback remains supported for existing numeric-target fixtures and compatibility tests"
  - "Classification summaries keep compatibility fields like RMSE while adding PR-AUC and log-loss evidence"

patterns-established:
  - "Model-selection code should infer objective mode from the target instead of hardcoding regression"
  - "Probability metrics must encode labels against the model column order before scoring"

requirements-completed: [DEBT-04]

duration: 35min
completed: 2026-03-12
---

# Phase 35 Plan 01: XGBoost Fix Summary

**The model-core evaluation stack now matches the saved XGBoost classifier instead of silently reverting to regression during CV and optimization**

## Performance

- **Duration:** 35 min
- **Completed:** 2026-03-12T16:47:26Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added [classification_probability_metrics](/home/eimi/projects/ai-btc-predictor/src/bitbat/model/evaluate.py) with stable label-order handling and normalized probability scoring
- Updated [walk_forward.py](/home/eimi/projects/ai-btc-predictor/src/bitbat/model/walk_forward.py) to train `multi:softprob` for label targets, emit class probabilities, and report `mean_pr_auc` / `mean_logloss`
- Updated [optimize.py](/home/eimi/projects/ai-btc-predictor/src/bitbat/model/optimize.py) so label-driven optimization uses PR-AUC-based scoring while numeric fixtures still use RMSE
- Added regression coverage in [test_walk_forward.py](/home/eimi/projects/ai-btc-predictor/tests/model/test_walk_forward.py) and [test_optimize.py](/home/eimi/projects/ai-btc-predictor/tests/model/test_optimize.py) for the classification objective, probability outputs, and PR-AUC guardrail

## Task Commits

1. **Task 1-2: Add classification-mode core tests and implement adaptive walk-forward/optimizer behavior** - `a0ab91e` (feat)

## Verification

- `poetry run pytest tests/model/test_walk_forward.py tests/model/test_optimize.py tests/model/test_train.py tests/model/test_infer.py -x`
- `poetry run ruff check src/bitbat/model/walk_forward.py src/bitbat/model/optimize.py src/bitbat/model/evaluate.py tests/model/test_walk_forward.py tests/model/test_optimize.py`

## Decisions Made

- Preserved regression-mode evaluation for numeric targets so older fixtures and internal compatibility tests stay valid
- Represented classification fold predictions as both directional scores and raw probability columns so downstream reporting can keep numeric compatibility fields
- Locked the PR-AUC guardrail with a deterministic separable-data walk-forward test instead of leaving it as a manual expectation

## Deviations from Plan

None.

## Next Phase Readiness

- Plan 35-01 is complete and verified
- CLI CV and optimization can now switch to label targets without changing the model-core API again
- Operator-facing summary output can add PR-AUC evidence on top of the new core objective mode

---
*Phase: 35-xgboost-fix*
*Completed: 2026-03-12*
