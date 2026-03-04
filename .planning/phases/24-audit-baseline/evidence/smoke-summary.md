# E2E Pipeline Smoke Test Summary

**Date:** 2026-03-04
**Config:** `/tmp/smoke_config.yaml` (freq=1h, horizon=4h, tau=0.005, sentiment/garch/macro/onchain disabled)
**Data source:** Real yfinance data (BTC-USD, start 2025-01-01, interval 1h)

## Results

| Stage | Command | Result | Output | Error (if any) |
|-------|---------|--------|--------|----------------|
| 1. Ingest prices | `bitbat prices pull --symbol BTC-USD --start 2025-01-01 --interval 1h` | PASS | 10,106 rows saved to `btcusd_yf_1h.parquet` | -- |
| 2. Build features | `bitbat features build` | PASS | 10,059 rows in feature matrix (`1h_4h/dataset.parquet`) | -- |
| 3. Train model | `bitbat model train` | PASS | XGBoost model saved to `models/1h_4h/xgb.json` | -- |
| 4. Batch prediction | `bitbat batch run` | PASS | Prediction stored at `predictions/1h_4h.parquet` + autonomous DB | -- |
| 5. Monitor run-once | `bitbat monitor run-once` | PARTIAL | Monitor completed full cycle; validation, drift detection, and prediction all ran | Retraining failed: `ValueError: Not enough samples for configured windows: 10059 < 17280` |

## Summary

**4/5 stages fully passed, 1 partial pass.**

### Stage 5 Analysis

The monitor ran its full monitoring cycle successfully:
- Validations: 6 completed
- Drift detection: True (drift detected, which is expected for a fresh short dataset)
- Prediction state: `duplicate_bar` (prediction for latest bar already existed from Stage 4)
- Realization state: realized (1 pending validation)

The only failure was the retraining sub-step, which required 17,280 samples (2 years of 1h bars) but the smoke test dataset only had 10,059 (~14 months). This is expected behavior: the walk-forward CV window requirements are larger than our test dataset. The monitor itself completed successfully.

### Observations

1. **Model output path does not respect `data_dir`:** Stage 3 saved the model to `models/1h_4h/xgb.json` relative to the project root, not to `/tmp/smoke_test/data/models/`. This means the `--config` data_dir only partially propagates. The model persistence path appears hardcoded or uses a separate config key.

2. **Autonomous DB path partially respects config:** Stage 4 stored predictions in both `predictions/1h_4h.parquet` (correctly under data_dir) and `autonomous.db` at `data/autonomous.db` (project root, not smoke test dir).

3. **No --tau option on model train:** Confirmed. `model train --help` shows `--freq`, `--horizon`, `--family` only. The CORR-01 bug (retrainer passing `--tau` to `model train`) would cause `UsageError: No such option: --tau` at retraining time.

### Known Gap: CORR-01

The `model train` command succeeds when called directly (as in this smoke test). The CORR-01 bug only manifests when the autonomous retrainer (`AutoRetrainer.retrain()`) calls `model train` via subprocess with the `--tau` argument. This cannot be tested via direct CLI invocation and is documented as a pre-validated CRITICAL finding from code inspection.

### Known Gap: CORR-02

The CV key mismatch bug (CORR-02) was not directly testable in this smoke test because the retraining sub-step failed before reaching the CV score comparison logic (due to insufficient data). This bug is pre-validated from code inspection and documented as CRITICAL.

---
*Generated: 2026-03-04*
*Plan: 24-03 (Audit Baseline - Coverage & Smoke Test)*
