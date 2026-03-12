---
phase: 28-activate-fold-aware-obv
plan: "01"
subsystem: autonomous/continuous-trainer
tags: [leakage, obv, fold-aware, continuous-training, wiring]
dependency_graph:
  requires: [phase-25-fold-aware-obv]
  provides: [LEAK-02-production-activation]
  affects: [src/bitbat/autonomous/continuous_trainer.py]
tech_stack:
  added: []
  patterns: [fold-boundary-wiring, behavioral-test]
key_files:
  created:
    - tests/dataset/test_fold_boundaries_wiring.py
  modified:
    - src/bitbat/autonomous/continuous_trainer.py
decisions:
  - fold_boundaries passed as [self.train_window_bars] (raw bar index before dropna) — conservative, acceptable NaN-offset
  - inference paths (predictor.py, batch cli) intentionally left at fold_boundaries=None
  - behavioral test placed in tests/dataset/ using public generate_price_features (no underscore)
metrics:
  duration: 165s
  completed: 2026-03-08
  tasks_completed: 2
  files_changed: 2
---

# Phase 28 Plan 01: Activate Fold-Aware OBV Summary

**One-liner:** Wired fold_boundaries=[self.train_window_bars] into ContinuousTrainer._do_retrain() and added a behavioral test confirming obv_fold_aware() is exercised, closing the LEAK-02 production activation gap.

## What Was Built

Phase 25 implemented `obv_fold_aware()` and added the `fold_boundaries` parameter to `generate_price_features()`. The parameter existed and the logic was correct, but no production code path was passing it. This plan closes that gap by wiring `fold_boundaries=[self.train_window_bars]` into `ContinuousTrainer._do_retrain()`, which is the only retraining path where OBV leakage is directly addressable at feature-generation time.

## Changes

### src/bitbat/autonomous/continuous_trainer.py (line 167-171)

Changed `generate_price_features(prices, enable_garch=self.enable_garch, freq=self.freq)` to:

```python
features = generate_price_features(
    prices,
    enable_garch=self.enable_garch,
    freq=self.freq,
    fold_boundaries=[self.train_window_bars],
)
```

### tests/dataset/test_fold_boundaries_wiring.py (new file)

Behavioral test with a single test function:
- `test_generate_price_features_fold_boundaries_changes_obv_second_segment`: generates 100 bars of synthetic OHLCV, calls `generate_price_features` with and without `fold_boundaries=[50]`, asserts OBV values are identical for positions `[:50]` and differ for positions `[50:]`.

## Test Results

- New test: 1 passed
- Targeted suite (test_obv_fold_aware + tests/dataset/): 17 passed
- Full suite: 638 passed, 0 failed
- Ruff lint (modified files): clean

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] E501 line-length violation in test docstring**
- **Found during:** Task 2 (ruff check)
- **Issue:** Test docstring at line 34 was 101 characters, exceeding the 100-char limit
- **Fix:** Shortened docstring from "...must differ from standard OBV." to "...must differ from standard."
- **Files modified:** tests/dataset/test_fold_boundaries_wiring.py
- **Commit:** 2e0133b

## Commits

| Hash | Description |
|------|-------------|
| f4c0af0 | feat(28-01): wire fold_boundaries into ContinuousTrainer._do_retrain() |
| 2e0133b | fix(28-01): trim docstring to satisfy E501 line-length in wiring test |

## Self-Check: PASSED
