---
phase: 30-fix-and-reset
plan: 01
subsystem: model
tags: [xgboost, multi:softprob, classification, inference, validation, tau]

requires:
  - phase: 29-diagnosis
    provides: ROOT_CAUSE.md identifying three compounding bugs (train objective, infer decoding, validator tau)

provides:
  - XGBoost classification training via multi:softprob objective with DIRECTION_CLASSES encoding
  - 3-class argmax direction decoding in infer.py with p_flat probability output
  - PredictionValidator with functional constructor tau from config
  - cli.py model_train uses label column (not r_forward) for XGBoost target
  - cli.py batch_run handles predicted_return=None without crashing

affects:
  - 30-02-retrain (retrain pipeline depends on corrected train.py/infer.py)
  - 31-validate (accuracy measurement depends on corrected validator.py tau)
  - any phase using predict_bar (output schema changed: p_flat added, predicted_return=None)

tech-stack:
  added: []
  patterns:
    - DIRECTION_CLASSES dict used for consistent string-to-int label encoding across train/infer
    - INT_TO_DIRECTION reverse map for argmax decoding in predict_bar
    - TDD red-green per task: failing tests written before implementation

key-files:
  created: []
  modified:
    - src/bitbat/model/train.py
    - src/bitbat/model/infer.py
    - src/bitbat/autonomous/validator.py
    - src/bitbat/cli.py
    - tests/model/test_train.py
    - tests/model/test_infer.py
    - tests/model/test_assert_guards.py
    - tests/autonomous/test_validator.py

key-decisions:
  - "DIRECTION_CLASSES identical in train.py and infer.py — guarded by test_direction_classes_consistent_across_modules"
  - "predict_bar returns predicted_return=None; p_flat added to output; directional_confidence kept but not called"
  - "cli model_train now uses require_label=True so label column survives ensure_feature_contract filtering"
  - "validator tau loads from get_runtime_config() or load_config() fallback; constructor tau overrides"
  - "test_assert_guards.py updated to use direction labels for fit_xgb test (was using float y)"

patterns-established:
  - "All XGBoost training paths must use direction label Series (up/down/flat) not float r_forward"
  - "predict_bar output always includes p_flat; predicted_return is None for classification model"

requirements-completed:
  - FIXR-01

duration: 10min
completed: 2026-03-08
---

# Phase 30 Plan 01: Fix Three Root-Cause Bugs Summary

**XGBoost retrained with multi:softprob 3-class objective, argmax direction decoding, and validator tau wired from config — eliminating all three compounding bugs from ROOT_CAUSE.md**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-08T12:53:16Z
- **Completed:** 2026-03-08T13:03:48Z
- **Tasks:** 3 (+ 1 deviation auto-fix)
- **Files modified:** 8

## Accomplishments

- Bug 1 fixed: train.py uses `multi:softprob`, `num_class=3`, `mlogloss` with DIRECTION_CLASSES label encoding
- Bug 2 fixed: infer.py `predict_bar` uses `np.argmax(probs[0])` + INT_TO_DIRECTION; p_flat added; predicted_return=None
- Bug 3 fixed: validator.py `PredictionValidator.__init__` reads tau from config instead of hardcoding 0.0
- All 650 tests pass; ruff check and lint-imports green

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix train.py — multi:softprob classification objective** - `1cb3dc9` (feat)
2. **Task 2: Fix infer.py — 3-class argmax direction decoding** - `017db9a` (feat)
3. **Task 3: Fix validator.py — wire constructor tau from config** - `f720dc4` (feat)
4. **Deviation: cli.py require_label fix** - `d3a7982` (fix)

## Files Created/Modified

- `/home/eimi/projects/ai-btc-predictor/src/bitbat/model/train.py` - DIRECTION_CLASSES constant, multi:softprob objective, label encoding
- `/home/eimi/projects/ai-btc-predictor/src/bitbat/model/infer.py` - DIRECTION_CLASSES, INT_TO_DIRECTION, numpy argmax decoding, p_flat output
- `/home/eimi/projects/ai-btc-predictor/src/bitbat/autonomous/validator.py` - tau wired from constructor/config
- `/home/eimi/projects/ai-btc-predictor/src/bitbat/cli.py` - require_label=True in model_train; .get() for predicted_return crash sites
- `/home/eimi/projects/ai-btc-predictor/tests/model/test_train.py` - 3 new tests: objective, output shape, constant consistency; updated existing tests
- `/home/eimi/projects/ai-btc-predictor/tests/model/test_infer.py` - 2 new tests: 3-class direction, p values sum to 1
- `/home/eimi/projects/ai-btc-predictor/tests/model/test_assert_guards.py` - fit_xgb test updated to use direction labels
- `/home/eimi/projects/ai-btc-predictor/tests/autonomous/test_validator.py` - 2 new tests: constructor tau, config tau default

## Decisions Made

- `DIRECTION_CLASSES = {"up": 0, "down": 1, "flat": 2}` defined identically in both train.py and infer.py; consistency enforced by `test_direction_classes_consistent_across_modules`
- `predict_bar` no longer calls `directional_confidence()` (function left in place for backward compat); predicted_return returns None
- `cli.py model_train` changed from `require_label=False` to `require_label=True` so the `label` column survives `ensure_feature_contract` column filtering
- Random forest path unchanged — still uses `r_forward` float y

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_assert_guards.py fit_xgb test used float y Series**
- **Found during:** Task 2 (full model suite run)
- **Issue:** `test_fit_xgb_returns_booster_type` created float `y` via `rng.normal()`, which now raises `IntCastingNaNError` when `fit_xgb` tries `y.map(DIRECTION_CLASSES).astype(int)` on floats
- **Fix:** Added `_make_direction_data()` helper using `rng.choice(["up","down","flat"])`, updated test to use it
- **Files modified:** `tests/model/test_assert_guards.py`
- **Verification:** `poetry run pytest tests/model/ -x -q` — 83 passed
- **Committed in:** `017db9a` (Task 2 commit)

**2. [Rule 1 - Bug] cli.py model_train dropped label column via require_label=False**
- **Found during:** Full suite run after Task 3
- **Issue:** `_load_feature_dataset` with `require_label=False` causes `ensure_feature_contract` to exclude `label` from `ordered` columns; subsequent `dataset["label"]` access raises `KeyError: 'label'`
- **Fix:** Changed `require_label=False` to `require_label=True` in model_train's `_load_feature_dataset` call
- **Files modified:** `src/bitbat/cli.py`
- **Verification:** `test_cli_model_train_family_both` passes; full suite 650 passed
- **Committed in:** `d3a7982` (deviation fix commit)

---

**Total deviations:** 2 auto-fixed (Rule 1 bugs)
**Impact on plan:** Both fixes required for correctness. No scope creep.

## Issues Encountered

- numpy import in infer.py initially had `# noqa: F401` suppression (planned for Task 2 use); removed the noqa when the argmax code was added in the same task
- `test_fit_baseline_supports_both_families` parametrized test needed updating to pass direction labels for xgb and float labels for random_forest (handled in Task 1)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All three root-cause bugs from ROOT_CAUSE.md are fixed and unit-tested
- 650 tests pass; ruff and lint-imports clean
- Phase 30-02 (retrain pipeline / reset) can proceed with the corrected training objective
- Phase 31 (validate accuracy recovery) can proceed with corrected validator tau
- Note: existing saved model artifacts (`models/` directory) were trained with the old regression objective and must be retrained in Phase 30-02 before accuracy metrics are meaningful

---
*Phase: 30-fix-and-reset*
*Completed: 2026-03-08*
