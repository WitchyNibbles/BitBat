---
phase: 35-xgboost-fix
plan: 02
subsystem: cli-model-selection
tags: [cli, xgboost, classification, reporting, tdd]

requires:
  - phase: 35-xgboost-fix
    plan: 01
    provides: Classification-aware model core
provides:
  - CLI CV and optimization wired to label-driven XGBoost evaluation
  - PR-AUC-aware operator summaries and JSON payloads
  - regression-fixture-compatible XGBoost prediction helper behavior
affects: [operator-workflow, cv-summary, optimization-summary]

tech-stack:
  added: []
  patterns:
    - CLI family-specific target routing
    - compatibility-preserving summary extension for classification metrics

key-files:
  created: []
  modified:
    - src/bitbat/cli/_helpers.py
    - src/bitbat/cli/commands/model.py
    - tests/test_cli.py
    - tests/model/test_phase5_complete.py

key-decisions:
  - "CLI model cv loads labels whenever XGBoost participates, while RandomForest stays on r_forward"
  - "Optimization output reports best_pr_auc for classification mode and keeps best_score as the minimized search metric"
  - "The CLI helper still accepts 1-D XGBoost regression-style outputs so older fixtures do not break"

patterns-established:
  - "CLI reporting should branch on objective_mode rather than assuming RMSE-only output"
  - "Shared prediction helpers can derive a directional score from class probabilities while exposing raw probabilities on demand"

requirements-completed: [DEBT-04]

duration: 25min
completed: 2026-03-12
---

# Phase 35 Plan 02: XGBoost Fix Summary

**The operator-facing `model cv` and `model optimize` flows now exercise the same classification objective as training and inference**

## Performance

- **Duration:** 25 min
- **Completed:** 2026-03-12T16:47:26Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Extended [_predict_baseline](/home/eimi/projects/ai-btc-predictor/src/bitbat/cli/_helpers.py) so XGBoost probability matrices can be consumed either as raw class probabilities or as a derived directional score
- Updated [model.py](/home/eimi/projects/ai-btc-predictor/src/bitbat/cli/commands/model.py) so XGBoost CV uses `label`, RandomForest keeps `r_forward`, and optimization runs against labels
- Added PR-AUC, log-loss, and objective-mode fields to the CV/optimization summaries without removing the existing compatibility fields
- Added CLI and integration coverage in [test_cli.py](/home/eimi/projects/ai-btc-predictor/tests/test_cli.py) and [test_phase5_complete.py](/home/eimi/projects/ai-btc-predictor/tests/model/test_phase5_complete.py)

## Task Commits

1. **Task 1-2: Add CLI classification tests and wire label-driven CV/optimization output** - `b930ba4` (feat)

## Verification

- `poetry run pytest tests/test_cli.py tests/model/test_phase5_complete.py tests/model/test_train.py tests/model/test_infer.py -x`
- `poetry run ruff check src/bitbat/cli/_helpers.py src/bitbat/cli/commands/model.py tests/test_cli.py tests/model/test_phase5_complete.py`

## Decisions Made

- Kept the JSON summaries backward-compatible by retaining `average_rmse`, `average_mae`, and `best_score` while adding classification-specific fields
- Treated 1-D XGBoost outputs as a compatibility fallback so tests and fixtures that stub regression-like boosters still work
- Standardized CLI messaging around `objective_mode` instead of baking metric names into every call site

## Deviations from Plan

None.

## Next Phase Readiness

- Plan 35-02 is complete and verified
- Phase 35 now has end-to-end coverage from model core through CLI/operator summaries
- Milestone v1.6 is ready for closeout or audit

---
*Phase: 35-xgboost-fix*
*Completed: 2026-03-12*
