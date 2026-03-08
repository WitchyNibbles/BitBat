# Root Cause Analysis: Live Accuracy Collapse

- **Date:** 2026-03-08
- **Phase:** 29 — Diagnosis
- **Note:** Committed before any fix code (Phase 30)

---

## Observed Symptom

The live monitoring system reported **38 correct / 266 realized predictions = 14.3% directional accuracy** across two active frequency/horizon pairs.

| Pair | Predictions | Correct |
|------|-------------|---------|
| 5m_30m | 188 | ~27 |
| 1h_4h | 78 | ~11 |

**Expected baseline:** 33% (3-class up/down/flat random baseline)

**Observed accuracy is less than half the random baseline**, indicating systematic failure rather than a model quality problem. The collapse was caused by three compounding bugs across the Model Training, Serving/Inference, and Validation/Realization stages.

---

## Pipeline Stage Trace

The pipeline stages are:

```
Stage 1: Ingestion → Stage 2: Features → Stage 3: Labels →
Stage 4: Model Training → Stage 5: Serving/Inference → Stage 6: Validation/Realization
```

Stages 1–3 are confirmed NOT at fault (see "Stages NOT at Fault" section). The bugs are in Stages 4–6.

---

### Stage 4: Model Training (PRIMARY)

**Bug:** `objective = "reg:squarederror"` — regression objective trained on `r_forward` float values, not classification labels.

**File:** `src/bitbat/model/train.py`, line 53

**Root cause detail:**
The model was trained with a regression objective on the raw forward return float (`r_forward`), instead of a multi-class classification objective on direction labels (`up` / `down` / `flat`). As a result, the model outputs a continuous return magnitude estimate rather than class probabilities. The output is systematically negative-biased because the training distribution of `r_forward` has a negative mean (mean = -0.00246 on the live window).

**Evidence:** `xgb.Booster.save_config()` on both model artifacts (`models/5m_30m/xgb.json`, `models/1h_4h/xgb.json`) returns `"reg:squarederror"` in the objective field. The correct objective for 3-class direction prediction is `multi:softprob`.

**Reproducible trace:**
```bash
poetry run pytest tests/diagnosis/test_pipeline_stage_trace.py::test_model_objective_is_regression -v
```

---

### Stage 5: Serving / Inference (SECONDARY)

**Bug:** Direction derived from `sign(predicted_return)` with no tau threshold; binary (up/down) only — no flat class output.

**File:** `src/bitbat/model/infer.py`, line 86

**Root cause detail:**
The inference code converts the regression output to a direction string using:
```python
predicted_direction = "up" if predicted_return > 0 else "down"
```
This mapping has two flaws:
1. No `tau` threshold — any return above 0 is "up", any at or below 0 is "down". Flat returns near zero are misclassified rather than labeled "flat".
2. The "flat" class is entirely absent from predictions, making it impossible to correctly match actual flat moves.

Because Stage 4 produces negative-biased regression outputs (mean = -0.00246), the majority of predictions are mapped to "down". This produces 203/268 = 76% "down" predictions with 0% "flat".

**Evidence:** Live predictions parquet: 203/268 rows have `predicted_direction = "down"`, 65/268 have `"up"`, 0 have `"flat"`. Mean `predicted_return` = -0.00246.

**Reproducible trace:**
```bash
poetry run pytest tests/diagnosis/test_pipeline_stage_trace.py::test_serving_direction_bias -v
```

---

### Stage 6: Validation / Realization (TERTIARY)

**Bug:** `self.tau = 0.0` hardcoded at `validator.py:46`, overriding the constructor parameter; also, price lookup returns `actual_return = 0.0` for 179/266 rows due to data gaps in the validation window.

**File:** `src/bitbat/autonomous/validator.py`, line 46

**Root cause detail:**
Two independent sub-bugs compound here:

1. **Tau override:** The validator class sets `self.tau = 0.0` unconditionally at line 46, ignoring any `tau` value passed to the constructor. This means `classify_direction(actual_return, tau=0.0)` is called for all rows. With `tau=0.0`, any non-zero return (no matter how small) is classified as "up" or "down", and only exact zero returns become "flat". However:

2. **Price lookup gaps:** The price lookup function returns `actual_return = 0.0` for rows where the forward horizon price is not available in the stored OHLCV data. This affects 179/266 = 67.3% of all realized predictions. These rows get `actual_direction = "flat"` because `classify_direction(0.0, tau=0.0) = "flat"`.

**Compounding with Bugs 1+2:** With 76% of predictions labeled "down" (Bug 2 output) and 67% of actuals labeled "flat" (Bug 3 output), the match rate collapses: `"down" vs "flat" = wrong` for the majority of evaluated rows.

**Evidence:** 179/266 realized predictions have `actual_return = 0.0` exactly in `data/autonomous.db`, confirmed by direct SQLite query.

**Reproducible trace:**
```bash
poetry run pytest tests/diagnosis/test_pipeline_stage_trace.py::test_validation_zero_return_corruption -v
```

---

## Compounding Effect

The three bugs do not act independently — they form a failure cascade:

1. **Bug 1 (Stage 4)** produces regression outputs with mean = -0.00246. The model was never asked to predict direction; it predicts return magnitude.

2. **Bug 2 (Stage 5)** converts those negative-biased floats to direction labels using a sign threshold. Because most regression outputs are slightly negative, 76% of predictions become "down". The "flat" class never appears.

3. **Bug 3 (Stage 6)** corrupts the ground truth. Price lookup gaps fill 67% of `actual_return` values with 0.0. With `tau=0.0`, those zeros become `actual_direction = "flat"`.

**Final state of live evaluation window:**
- Predicted: 76% "down", 24% "up", 0% "flat"
- Actual: 67% "flat", 33% "up"/"down"
- A "down" prediction matching a "flat" actual = wrong

The result is **38/266 = 14.3%** — less than half the 33% random baseline. Without these three bugs, the evaluation would use a correctly trained classifier, a tau-thresholded 3-class output, and accurate ground truth labels.

---

## Summary Table

| Stage | Bug | Severity | Fix Phase |
|-------|-----|----------|-----------|
| Stage 4: Model Training | `reg:squarederror` regression objective instead of `multi:softprob` classification | PRIMARY | Phase 30 |
| Stage 5: Serving / Inference | Direction from `sign(predicted_return)` with no tau; no "flat" class output | SECONDARY | Phase 30 |
| Stage 6: Validation / Realization | `self.tau = 0.0` hardcoded overrides constructor; price lookup returns 0.0 for 67% of rows | TERTIARY | Phase 30 |

---

## Stages NOT at Fault

**Stage 1 — Ingestion:** OHLCV data is present and complete for the feature window. No gaps were found in the raw price data used for feature engineering (separate from the validation window price gaps in Bug 3). Data confirmed via `tests/features/test_leakage.py`.

**Stage 2 — Features:** Feature values are numerically correct. No data leakage was introduced. The leakage guardrail test (`tests/features/test_leakage.py`) continues to pass. Feature distributions are normal and within expected ranges.

**Stage 3 — Labels:** The label column `direction` in the assembled dataset is correctly computed for the configured `tau` at dataset assembly time. Label distribution is binary (up/down only) because `tau=0.0` was used during dataset assembly — this is consistent with Bug 3's tau=0.0, but the labeling computation itself is correct for the tau it received. The labeling module is not at fault.

---

## Reproducible Accuracy Collapse Trace

To confirm the 14.3% accuracy figure directly from the live SQLite database:

```bash
poetry run pytest tests/diagnosis/test_pipeline_stage_trace.py::test_accuracy_below_random_baseline -v
```

This test queries `data/autonomous.db` and asserts `hit_rate < 0.33`.

---

*This document was committed on 2026-03-08 as the DIAG-02 deliverable. No fix code exists at this commit. Phase 30 will address all three bugs.*
