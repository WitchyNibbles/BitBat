---
status: complete
phase: 25-critical-correctness-remediation
source: [25-01-SUMMARY.md, 25-02-SUMMARY.md, 25-03-SUMMARY.md, 25-04-SUMMARY.md]
started: 2026-03-07T00:00:00Z
updated: 2026-03-07T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Full test suite passes (626 tests)
expected: Running `poetry run pytest` (or `make test`) completes with 626 tests, 0 failures, 0 errors.
result: issue
reported: "all tests passed but there are some warnings: XGBoost UserWarning 'Parameters: { n_estimators } are not used' from tests/analytics/test_explainer.py (11 warnings)"
severity: cosmetic

### 2. Retrainer subprocess no longer passes --tau
expected: In `src/bitbat/autonomous/retrainer.py`, the subprocess command for `features build` does not contain `--tau`. Running `grep -n "\-\-tau" src/bitbat/autonomous/retrainer.py` returns no matches in the build command.
result: pass

### 3. CV metric key is mean_directional_accuracy
expected: In `src/bitbat/cli.py`, the cv_summary.json aggregate key is `mean_directional_accuracy` (not `average_balanced_accuracy`). Running `grep "mean_directional_accuracy" src/bitbat/cli.py` shows at least one match. The retrainer reader (`_read_cv_score` in retrainer.py) also references `mean_directional_accuracy` as the primary key.
result: pass

### 4. regression_metrics() is a pure function
expected: `src/bitbat/model/evaluate.py` exports two separate functions: `regression_metrics()` (pure computation, no file writes) and `write_regression_metrics()` (explicit I/O). Running `grep "def regression_metrics\|def write_regression_metrics" src/bitbat/model/evaluate.py` shows both definitions.
result: pass

### 5. Leakage guardrail test file exists with 3 tests
expected: `tests/features/test_leakage.py` exists and contains 3 test functions: a PR-AUC guardrail test (training on random labels yields score < 0.7), a no-future-timestamps test, and an OBV no-lookahead test. Running `pytest tests/features/test_leakage.py -v` passes all 3.
result: pass

### 6. API defaults sourced from config (not hardcoded 1h/4h)
expected: The API routes no longer hardcode `Query("1h")` or `Query("4h")`. `src/bitbat/api/defaults.py` exists and `src/bitbat/api/routes/predictions.py` imports from it. Running `grep -r '"1h"\|"4h"' src/bitbat/api/routes/` returns no matches with Query defaults.
result: issue
reported: "metrics.py still has hardcoded value: all_preds = db.get_recent_predictions(session, \"1h\", \"4h\", days=30, realized_only=False) at lines 93-95"
severity: major

### 7. OBV fold-aware function exists
expected: `src/bitbat/features/price.py` contains `obv_fold_aware()` function alongside the original `obv()`. `src/bitbat/dataset/build.py` contains a `fold_boundaries` parameter in `_generate_price_features()`. Running `grep "obv_fold_aware\|fold_boundaries" src/bitbat/features/price.py src/bitbat/dataset/build.py` confirms both.
result: pass

## Summary

total: 7
passed: 5
issues: 2
pending: 0
skipped: 0

## Gaps

- truth: "Full test suite completes with 0 failures and 0 warnings"
  status: failed
  reason: "User reported: all tests passed but there are some warnings: XGBoost UserWarning 'Parameters: { n_estimators } are not used' from tests/analytics/test_explainer.py (11 warnings)"
  severity: cosmetic
  test: 1
  artifacts: []
  missing: []

- truth: "All API routes source freq/horizon defaults from config, no hardcoded 1h/4h"
  status: failed
  reason: "User reported: metrics.py still has hardcoded value: all_preds = db.get_recent_predictions(session, \"1h\", \"4h\", days=30, realized_only=False) at lines 93-95"
  severity: major
  test: 6
  artifacts: []
  missing: []
