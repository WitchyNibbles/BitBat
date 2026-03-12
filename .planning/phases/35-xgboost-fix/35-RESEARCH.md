# Phase 35: XGBoost Fix - Research

**Researched:** 2026-03-12  
**Domain:** XGBoost objective alignment across training, CV, and optimization  
**Confidence:** HIGH

## Summary

The primary `fit_xgb()` training path is already classification-based: `src/bitbat/model/train.py` uses `multi:softprob`, `num_class=3`, and `mlogloss`, and `src/bitbat/model/infer.py` already interprets XGBoost outputs as three-class probabilities. The remaining mismatch is in the walk-forward and optimization stack plus the CLI CV/optimization commands that still feed continuous return targets into XGBoost-oriented evaluation paths.

Remaining runtime objective mismatches:

- `src/bitbat/model/walk_forward.py` still trains XGBoost with `reg:squarederror`
- `src/bitbat/model/optimize.py` still trains XGBoost with `reg:squarederror`
- `src/bitbat/cli/commands/model.py` still runs `model cv` / `model optimize` against `r_forward` for the XGBoost path
- `src/bitbat/cli/_helpers.py::_predict_baseline()` assumes XGBoost returns a 1-D regression output, not a `(n, 3)` probability matrix

Existing tests already confirm the training/inference baseline:

- `tests/model/test_train.py` checks that `fit_xgb()` produces `multi:softprob`
- `tests/model/test_infer.py` checks probability-shape inference behavior
- `tests/diagnosis/test_pipeline_stage_trace.py` documents the original objective bug and expects the saved config objective to be `multi:softprob`

The missing piece for Phase 35 is to make the XGBoost **evaluation and selection** path classification-aware while preserving the existing regression fallback for numeric-target workflows and test fixtures. That suggests an adaptive implementation:

1. When the target is the direction label (`up` / `down` / `flat`), use classification training with `multi:softprob`
2. When the target is numeric, preserve the regression-mode behavior for legacy/internal fixtures
3. Add explicit PR-AUC and probability-based metrics for the classification path
4. Update CLI CV/optimization to load labels for the XGBoost path while leaving RandomForest on `r_forward`

## Recommended Architecture

### 1. Dual-mode walk-forward / optimizer core

Make `WalkForwardValidator` and `HyperparamOptimizer` infer mode from the target:

- label/string target → classification mode
- numeric target → regression mode

This keeps old regression-based tests and fixtures valid while fixing the real runtime XGBoost path.

### 2. Classification metrics for XGBoost selection

For classification mode, the important outputs are:

- class probabilities
- predicted direction
- PR-AUC (macro or one-vs-rest aggregate)
- directional accuracy
- log loss

`rmse` / `mae` can remain available for regression mode only. CLI summaries may still keep compatibility fields, but Phase 35 should add explicit PR-AUC reporting rather than pretending the classification path is regression.

### 3. CLI `model cv` / `model optimize` must load labels for XGBoost

Current CLI CV and optimization commands still load datasets with `require_label=False` and use `r_forward` for XGBoost. That bypasses the classification target entirely. The runtime fix should:

- require `label` when XGBoost is selected
- keep `r_forward` for RandomForest/backtest-derived numeric metrics
- handle 2-D XGBoost probability predictions explicitly in the CLI helper layer

## Risks

- CLI summary/test expectations currently reference `average_rmse`; those will need careful compatibility handling or targeted test updates
- Backtest/risk logic expects a numeric signal; classification probabilities likely need a derived directional score such as `p_up - p_down`
- Some tests intentionally construct regression XGBoost boosters as compatibility fixtures; runtime code should support that where appropriate rather than deleting those fixtures wholesale

## Validation Strategy

- Add classification-mode tests for `WalkForwardValidator` and `HyperparamOptimizer`
- Add CLI tests that prove `model cv` / `model optimize` consume labels for the XGBoost path and emit PR-AUC in summaries
- Re-run existing model train/infer tests to confirm saved artifact objective and probability outputs remain correct
- Keep regression-mode walk-forward/optimizer tests green to prove the adaptive fallback still works
