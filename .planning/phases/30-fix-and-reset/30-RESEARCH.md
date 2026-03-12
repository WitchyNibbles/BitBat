# Phase 30: Fix & Reset - Research

**Researched:** 2026-03-08
**Domain:** ML pipeline bug remediation — XGBoost objective fix, inference direction fix, validator tau fix, CLI reset command, test inversion
**Confidence:** HIGH (all findings from direct code inspection of live codebase)

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FIXR-01 | Root cause of live accuracy ~1% is fixed in code | Three bugs identified in ROOT_CAUSE.md. Fix locations: `train.py:53`, `infer.py:86`, `validator.py:46`. Each has a clear, minimal code change. Tests in `tests/diagnosis/` must be inverted after fixes. |
| FIXR-02 | A clean reset procedure (data/ + models/ + autonomous.db) is executable via CLI command(s) and documented | No reset command exists today. Must add `bitbat system reset` (or equivalent) as a Click command in `cli.py`. Paths are config-driven (`data_dir`) plus hardcoded `autonomous.db` default. |
| FIXR-03 | After reset + retrain, live directional accuracy on realized predictions exceeds random baseline (>33%) | Requires the three bugs above to be fixed before retrain is possible. The accuracy gate must be verified by an automated test querying `data/autonomous.db` after a retrain cycle completes. |
</phase_requirements>

---

## Summary

Phase 30 applies targeted fixes to the three bugs documented in `ROOT_CAUSE.md`, adds a CLI reset command, and inverts the Phase 29 diagnostic tests from "bug is present" to "bug is fixed". The fix scope is narrow: three specific lines of code in three separate files. No architectural changes are needed.

The three fixes are:

1. **`train.py` (FIXR-01, Bug 1):** Change `objective: reg:squarederror` to `objective: multi:softprob` with `num_class: 3`. Training target must switch from `r_forward` (float) to `label` (integer-encoded: `up=0, down=1, flat=2`). The dataset already has the `label` column with values `{"up", "down", "flat"}` enforced by `contracts.py`.

2. **`infer.py` (FIXR-01, Bug 2):** Replace `predicted_direction = "up" if predicted_return > 0 else "down"` with a 3-class argmax over the `(n_samples, 3)` probability matrix returned by `multi:softprob`. Map integer class index back to string: `{0: "up", 1: "down", 2: "flat"}`. The `predict_bar` function signature should continue to accept the same inputs but return correct 3-class direction strings.

3. **`validator.py` (FIXR-01, Bug 3):** Remove the hardcoded `self.tau = 0.0` at line 46. Wire `self.tau` from the constructor parameter. The constructor already accepts `tau: float | None = None`; the fix is to use `self.tau = tau if tau is not None else config_tau`.

**Primary recommendation:** Fix all three bugs in the same wave (they are interdependent for accuracy recovery), add the CLI reset command, invert the four diagnostic tests, and validate end-to-end with a unit-testable accuracy check.

---

## Standard Stack

All tooling is already present. No new dependencies required.

### Core (already present)
| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| `xgboost` | 2.1.4 | Classification training + inference | `multi:softprob` produces `(n, 3)` probability matrix in XGBoost 2.x |
| `click` | project | CLI command for reset | Add `@_cli.group` or subcommand under `system` group |
| `shutil` | stdlib | `rmtree` for directory deletion | Prefer over `os.remove` for recursive directory deletion |
| `pathlib.Path` | stdlib | Config-driven path resolution | `data_dir` from config, `autonomous.db` from `autonomous.database_url` |
| `pytest` | project | Test suite — unit + integration | Existing tests; invert 4 diagnostic assertions |
| `sqlite3` | stdlib | Direct DB query in tests | Already used in `tests/diagnosis/test_pipeline_stage_trace.py` |

### Supporting
| Library | Purpose | When to Use |
|---------|---------|-------------|
| `pandas` | Label encoding (`pd.Series.map`) | Convert `"up"/"down"/"flat"` → integer before `DMatrix` |
| `numpy` | `np.argmax` over probability matrix | Decode `(n, 3)` softprob output to class index |

**No new installation required.** All libraries are in the existing Poetry environment.

---

## Architecture Patterns

### Bug Fix Pattern: XGBoost Classification (Bug 1 — train.py)

XGBoost `multi:softprob` in version 2.1.4 requires:
- Integer-encoded labels in range `[0, num_class)` — string labels will error
- `num_class: 3` parameter
- `eval_metric: mlogloss` (appropriate for multiclass)

The dataset has `label` column values `{"up", "down", "flat"}`. A canonical encoding is:
```python
DIRECTION_CLASSES = {"up": 0, "down": 1, "flat": 2}
INT_TO_DIRECTION = {v: k for k, v in DIRECTION_CLASSES.items()}
```

The encoding must be consistent between train and infer. The planner should ensure both files use the same constant (e.g., defined in a shared location or duplicated explicitly).

**Exact XGBoost parameter change in `train.py`:**
```python
# BEFORE (Bug 1)
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    ...
}
# label=y_float (r_forward)

# AFTER (Fix 1)
params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    ...
}
# label=y_encoded (integer 0/1/2 from direction labels)
```

**Target column:** The training CLI at `cli.py:996` passes `y = dataset["r_forward"]`. This must be changed to use the integer-encoded `label` column. The `label` column exists in `dataset.parquet` (enforced by `contracts.py`).

### Bug Fix Pattern: Inference Direction Decoding (Bug 2 — infer.py)

With `multi:softprob`, `booster.predict(dmatrix)` returns a `(n_samples, 3)` float array in XGBoost 2.1.4. For a single-row prediction (`predict_bar`), this is shape `(1, 3)`.

```python
# BEFORE (Bug 2)
predicted_return = float(booster.predict(dmatrix)[0])
predicted_direction = "up" if predicted_return > 0 else "down"

# AFTER (Fix 2)
probs = booster.predict(dmatrix)          # shape (1, 3) for softprob
class_idx = int(np.argmax(probs[0]))
predicted_direction = INT_TO_DIRECTION[class_idx]  # "up", "down", or "flat"
p_up = float(probs[0][DIRECTION_CLASSES["up"]])
p_down = float(probs[0][DIRECTION_CLASSES["down"]])
p_flat = float(probs[0][DIRECTION_CLASSES["flat"]])
```

The existing `directional_confidence()` function in `infer.py` was written for a regression model (sigmoid of return magnitude) and is now obsolete. The fix replaces it with direct probability extraction from `probs[0]`. The return payload should add `p_flat` or keep `p_up`/`p_down` for backward API compatibility with the dashboard and API.

The `predicted_return` field in the output dict will no longer be meaningful (the model doesn't output a return scalar). Options: (a) set to `None`, (b) keep as `float(probs[0][0] - probs[0][1])` as a rough directionality score. Option (a) is simpler and more honest.

### Bug Fix Pattern: Validator Tau (Bug 3 — validator.py)

The fix is minimal:
```python
# BEFORE (Bug 3, validator.py:46)
self.tau = 0.0  # hardcoded, ignores constructor parameter

# AFTER (Fix 3)
cfg_tau = load_config().get("tau", 0.01)
self.tau = tau if tau is not None else cfg_tau
```

The config has `tau: 0.01` in `default.yaml`. The constructor signature already has `tau: float | None = None`. This fix makes the constructor parameter functional.

Note: Bug 3 also involves price gaps in the validation window (179/266 rows returning `actual_return = 0.0`). The tau fix alone corrects the classification logic. The price gap issue is a data availability problem — after a full reset + re-ingestion (FIXR-02 flow), the validation window will have fresh data without historical gaps. No additional code fix is needed for the price lookup gap beyond the reset procedure.

### Pattern: CLI Reset Command (FIXR-02)

Add a new CLI subcommand. Based on the existing CLI structure:

```python
@_cli.group(help="System utilities.")
def system() -> None:
    pass

@system.command("reset")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
def system_reset(yes: bool) -> None:
    """Delete data/, models/, and autonomous.db for a clean-slate restart."""
    ...
```

**Paths to delete:**
1. `data_dir` from config (default `"data"`) — use `shutil.rmtree(data_path, ignore_errors=True)`
2. `Path("models")` — hardcoded in `train.py:_default_model_path` and `persist.py:default_model_artifact_path` (both default to `"models"`)
3. `autonomous.db` — path parsed from `autonomous.database_url` in config (default `"sqlite:///data/autonomous.db"` → strip `sqlite:///` prefix)

Note: `models/` is NOT under `data_dir` — it is a separate hardcoded relative path. The reset command must delete both independently.

**Confirmation guard:** Always prompt `"This will delete all data, models, and the monitor database. Continue? [y/N]"` unless `--yes` flag is passed. This prevents accidental production wipes.

**CLI registration pattern (existing style):**
```python
# cli.py groups are registered at module level like:
@_cli.group(help="System utilities.")
def system() -> None:
    pass
```

### Pattern: Test Inversion (Phase 29 → Phase 30)

The four tests in `tests/diagnosis/test_pipeline_stage_trace.py` were written as "bug-present" assertions. After Phase 30 fixes, they must be inverted to "bug-fixed" assertions:

| Test | Bug-present assertion | Bug-fixed assertion |
|------|-----------------------|---------------------|
| `test_model_objective_is_regression` | `objective == "reg:squarederror"` | `objective == "multi:softprob"` |
| `test_serving_direction_bias` | `down_count > up_count * 2` | `abs(down_count - up_count) / total < 0.5` (balanced) |
| `test_validation_zero_return_corruption` | `zero_count >= 100` | `zero_count < 50` (after reset, most rows will have valid prices) |
| `test_accuracy_below_random_baseline` | `accuracy < 0.33` | `accuracy > 0.33` |

Tests that read from `data/autonomous.db` (tests 2, 3, 4) will `pytest.skip()` if the DB doesn't exist. After reset + retrain, these can only pass if the pipeline has been run. The unit tests for the code fixes (objective, direction decoding, tau) must pass on fresh code without needing live data.

### Anti-Patterns to Avoid

- **Testing against live DB state in fast unit tests:** Tests for Bug 1 fix (objective) and Bug 2 fix (direction decoding) must work in `tmp_path` without needing `data/autonomous.db`. Only accuracy-gate tests should require live DB.
- **Inverting tests without a NEW test for the fix:** Each inverted diagnostic test should be paired with a new deterministic unit test that verifies the fixed behavior with synthetic data. Do not rely solely on the inverted DB-dependent assertions.
- **Deleting `models/` using config `data_dir`:** `models/` is NOT inside `data_dir`. The reset command must explicitly delete the hardcoded `models/` path AND the config-driven `data_dir`. This is a real gotcha.
- **Forgetting to update `model_train` CLI:** The CLI at `cli.py:996` passes `y = dataset["r_forward"]`. After fixing `fit_xgb` to expect integer-encoded labels, the CLI call must also be updated to encode the labels.
- **Leaving `predicted_return` as float in `predict_bar` output:** After fixing Bug 2, the model no longer produces a meaningful scalar return. Leaving it as the raw regression float output would be misleading. Set to `None` or a probability difference score, and update callers that reference `predicted_return` in `autonomous/`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Directory deletion with error handling | Manual `os.listdir` + `os.remove` loops | `shutil.rmtree(path, ignore_errors=True)` | Handles nested dirs, permission edge cases, non-existent paths |
| Multiclass label encoding | Custom string→int mapping per file | Module-level constant `DIRECTION_CLASSES = {"up": 0, "down": 1, "flat": 2}` (defined once, imported) | Consistency across train+infer; prevents class index mismatch bugs |
| Probability extraction from `multi:softprob` | Custom sigmoid or re-mapping | `np.argmax(probs, axis=1)` + index lookup | XGBoost 2.x `multi:softprob` output is already a valid `(n, 3)` probability matrix |
| SQLite path extraction from URL | Custom URL parser | `url.replace("sqlite:///", "")` | The format is fixed; `autonomous.database_url` always uses `sqlite:///` prefix |

**Key insight:** The fixes are targeted 1-5 line changes per file. Avoid over-engineering (e.g., adding abstract label encoders, factory patterns for objectives). The simplest correct change is the right change.

---

## Common Pitfalls

### Pitfall 1: Label Encoding Inconsistency Between Train and Infer

**What goes wrong:** `train.py` encodes `"up"→0, "down"→1, "flat"→2` but `infer.py` decodes with a different mapping. Model predicts class index 0 as "down" when it was trained as "up".

**Why it happens:** The mapping is defined in two places (or not defined explicitly — hardcoded inline).

**How to avoid:** Define `DIRECTION_CLASSES = {"up": 0, "down": 1, "flat": 2}` once. Either in a shared constants file (e.g., `src/bitbat/model/constants.py`) or duplicated identically in both `train.py` and `infer.py` with a comment pointing to the twin. The planner should choose one location.

**Warning signs:** Accuracy barely above 33% (random) or directional distribution is still skewed after fix.

### Pitfall 2: XGBoost Predict Output Shape Mismatch

**What goes wrong:** With `multi:softprob`, `booster.predict(dmatrix)` returns `(n_samples, 3)` in XGBoost 2.x. Code written for regression (expecting a 1D `(n_samples,)` array) will break when indexing `preds[0]` — it gets a length-3 array, not a scalar.

**Why it happens:** Regression and softprob predictions have different shapes. This will cause a `TypeError` or silent wrong behavior if indexing is not updated.

**How to avoid:** After changing objective to `multi:softprob`, update ALL predict call sites: `infer.py:predict_bar` and any batch inference code in `cli.py` or `autonomous/`.

**Warning signs:** `TypeError: float() argument must be a string or a number, not 'numpy.ndarray'` at inference time.

### Pitfall 3: `model_train` CLI Still Passes `r_forward` as Target

**What goes wrong:** `cli.py:996` passes `y = dataset["r_forward"]` to `fit_xgb`. After changing `fit_xgb` to expect integer-encoded class labels, this will either error on `DMatrix` creation (floats are not valid class labels for multiclass) or silently train a broken model.

**Why it happens:** The training function signature changed but all call sites were not updated.

**How to avoid:** Search for all `fit_xgb(` and `fit_baseline(` calls. Update each to pass `dataset["label"].map(DIRECTION_CLASSES).astype(int)` as `y`.

**Warning signs:** XGBoost training raises `ValueError` about invalid label values.

### Pitfall 4: Reset Deletes Data But Not Models (or Vice Versa)

**What goes wrong:** The reset command deletes `data/` but not `models/`. After re-ingestion and feature build, the old (buggy) model is still loaded by the inference pipeline.

**Why it happens:** `models/` is a hardcoded relative path (`Path("models")` in `train.py` and `persist.py`), not under `data_dir`.

**How to avoid:** The reset command must explicitly delete both `data_dir` AND `Path("models")`. Document this explicitly in the CLI help text.

**Warning signs:** After reset + retrain, `monitor status` shows the old model artifact date.

### Pitfall 5: Diagnostic Tests Still Pass After Fix (Wrong Inversion)

**What goes wrong:** The four tests in `test_pipeline_stage_trace.py` were written to PASS when bugs are present. After Phase 30 fix, they should FAIL (confirming bugs are gone), and then be updated to have the inverted assertion. If they are not inverted, the test suite will report false failures.

**Why it happens:** The inversion step is easily skipped in the rush to fix the bugs.

**How to avoid:** Inversion of all four tests is a required deliverable of Phase 30. Add a new deterministic unit test for each fix (using synthetic data and `tmp_path`), separate from the DB-dependent diagnostic tests.

**Warning signs:** CI is red with the four diagnosis tests failing after Phase 30 merges.

---

## Code Examples

### Fix 1: XGBoost Classification Training

```python
# src/bitbat/model/train.py (verified against XGBoost 2.1.4 API)
DIRECTION_CLASSES = {"up": 0, "down": 1, "flat": 2}

def fit_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,  # Now expects string direction labels: "up"/"down"/"flat"
    *,
    seed: int = 42,
    persist: bool = True,
) -> tuple[xgb.Booster, dict[str, float]]:
    y_encoded = y_train.map(DIRECTION_CLASSES).astype(int)
    dtrain = xgb.DMatrix(X_train.astype(float), label=y_encoded.to_numpy())
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "seed": seed,
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    booster = xgb.train(params, dtrain, num_boost_round=100)
    ...
```

### Fix 2: Inference Direction Decoding

```python
# src/bitbat/model/infer.py (verified against XGBoost 2.1.4 API)
INT_TO_DIRECTION = {0: "up", 1: "down", 2: "flat"}
DIRECTION_CLASSES = {"up": 0, "down": 1, "flat": 2}

def predict_bar(model, features_row, timestamp=None, *, tau=0.01):
    booster = _ensure_model(model)
    ...
    dmatrix = xgb.DMatrix(features_row.to_frame().T, feature_names=feature_names)
    probs = booster.predict(dmatrix)  # shape (1, 3) for multi:softprob
    class_idx = int(np.argmax(probs[0]))
    predicted_direction = INT_TO_DIRECTION[class_idx]
    p_up = float(probs[0][DIRECTION_CLASSES["up"]])
    p_down = float(probs[0][DIRECTION_CLASSES["down"]])
    p_flat = float(probs[0][DIRECTION_CLASSES["flat"]])
    return {
        "timestamp": timestamp,
        "predicted_return": None,       # No longer a regression model
        "predicted_price": None,
        "predicted_direction": predicted_direction,
        "p_up": p_up,
        "p_down": p_down,
        "p_flat": p_flat,
    }
```

### Fix 3: Validator Tau

```python
# src/bitbat/autonomous/validator.py (line 46 fix)
def __init__(self, db, freq="5m", horizon="30m", tau=None):
    self.db = db
    self.freq = freq
    self.horizon = horizon
    # FIX: use constructor tau, fall back to config
    cfg_tau = (get_runtime_config() or load_config()).get("tau", 0.01)
    self.tau = tau if tau is not None else float(cfg_tau)
    ...
```

### CLI Reset Command

```python
# src/bitbat/cli.py — new command group
@_cli.group(help="System lifecycle commands.")
def system() -> None:
    pass

@system.command("reset")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
def system_reset(yes: bool) -> None:
    """Delete data/, models/, and autonomous.db for a clean-slate restart.

    After reset, run: bitbat ingest prices-once, features build, model train.
    """
    if not yes:
        click.confirm(
            "This will delete all data, models, and the monitor database. Continue?",
            abort=True,
        )
    cfg = _config()
    data_dir = Path(str(cfg.get("data_dir", "data"))).expanduser()
    models_dir = Path("models")
    db_url = str(cfg.get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db"))
    db_path = Path(db_url.replace("sqlite:///", ""))

    import shutil
    deleted = []
    for target in [data_dir, models_dir]:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            deleted.append(str(target))
    if db_path.exists() and db_path not in [data_dir / db_path.name]:
        db_path.unlink(missing_ok=True)
        deleted.append(str(db_path))

    if deleted:
        click.echo(f"Reset complete. Deleted: {', '.join(deleted)}")
    else:
        click.echo("Nothing to delete — already clean.")
```

Note: `db_path` (`data/autonomous.db`) is typically inside `data_dir`, so `shutil.rmtree(data_dir)` already deletes it. The explicit `db_path.unlink()` is for cases where `autonomous.database_url` points outside `data_dir`.

### Train CLI Update

```python
# cli.py model_train — target column change
# BEFORE
y = dataset["r_forward"]

# AFTER
from bitbat.model.train import DIRECTION_CLASSES
y = dataset["label"].map(DIRECTION_CLASSES).astype(int)
```

### Inverted Diagnostic Tests

```python
# tests/diagnosis/test_pipeline_stage_trace.py (inverted assertions)

def test_model_objective_is_classification():
    """Bug 1 FIXED: Model now trained with multi:softprob classification objective."""
    ...
    objective = cfg["learner"]["objective"]["name"]
    assert objective == "multi:softprob", f"Expected classification objective, got: {objective}"

def test_serving_direction_is_balanced():
    """Bug 2 FIXED: Direction distribution includes flat class; no 2x down bias."""
    ...
    flat_count = counts.get("flat", 0)
    assert flat_count > 0, "Flat class must appear in predictions after fix"
    assert not (down_count > up_count * 2), "Down bias should be eliminated after fix"

def test_validation_zero_return_eliminated():
    """Bug 3 FIXED: Price lookup gap rows below threshold after reset + fresh data."""
    ...
    assert zero_count < 50, f"Expected < 50 zero-return rows after fix, got: {zero_count}"

def test_accuracy_exceeds_random_baseline():
    """Accuracy recovery: hit rate > 33% on realized predictions."""
    ...
    assert accuracy > 0.33, f"Expected accuracy > 0.33, got: {accuracy:.3f}"
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `reg:squarederror` on return floats | `multi:softprob` with 3 integer-encoded classes | Model now predicts direction probabilities, not return magnitude |
| `sign(predicted_return)` → binary up/down | `argmax(probs)` → 3-class up/down/flat | Flat class appears in predictions; distribution balanced |
| Hardcoded `self.tau = 0.0` | Constructor `tau` wired to config `tau` | Validator correctly applies configured threshold for actual direction |
| No reset CLI command | `bitbat system reset --yes` | Operator can reach clean-slate without manual file manipulation |

---

## Open Questions

1. **`predicted_return` field backward compatibility**
   - What we know: The API (`api/routes/predictions.py`) and Streamlit dashboard may read `predicted_return` from the DB.
   - What's unclear: Whether setting `predicted_return = None` in the inference output will break API callers or dashboard rendering.
   - Recommendation: Search `api/` and `streamlit/` for `predicted_return` references before removing the field. If used, set to `p_up - p_down` as a signed confidence score instead of `None`.

2. **Batch inference paths in `cli.py` and `autonomous/`**
   - What we know: `batch_run` and `monitor_refresh` also call inference. These must also handle the new `(n, 3)` predict output shape.
   - What's unclear: Whether `batch_run` calls `predict_bar` or has its own DMatrix-level inference code.
   - Recommendation: Planner should check `cli.py:1201` (`batch_run`) and `autonomous/predictor.py` for additional predict call sites.

3. **Test accuracy gate: timing**
   - What we know: FIXR-03 requires accuracy > 33% on realized predictions. Realized predictions need to have their horizon pass (e.g., 30m or 4h) before `actual_return` is populated.
   - What's unclear: Whether the automated test for FIXR-03 can run in CI without a live pipeline.
   - Recommendation: The automated FIXR-03 test should be a DB-dependent `pytest.skip` if `data/autonomous.db` is absent (same pattern as Phase 29 tests). In CI without live data, it skips. The plan should document this explicitly.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `poetry run pytest tests/diagnosis/ tests/model/test_train.py tests/model/test_infer.py -x` |
| Full suite command | `poetry run pytest` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FIXR-01 (Bug 1) | `train.py` uses `multi:softprob` objective | unit | `poetry run pytest tests/model/test_train.py::test_fit_xgb_uses_classification_objective -x` | ❌ Wave 0 |
| FIXR-01 (Bug 1) | Training with direction labels produces valid class probabilities | unit | `poetry run pytest tests/model/test_train.py::test_fit_xgb_classification_output_shape -x` | ❌ Wave 0 |
| FIXR-01 (Bug 2) | `infer.py` returns 3-class direction including flat | unit | `poetry run pytest tests/model/test_infer.py::test_predict_bar_returns_three_classes -x` | ❌ Wave 0 |
| FIXR-01 (Bug 3) | `validator.py` uses constructor tau, not hardcoded 0.0 | unit | `poetry run pytest tests/autonomous/test_validator.py::test_validator_uses_constructor_tau -x` | ❌ Wave 0 |
| FIXR-01 (inversion) | Diagnosis tests confirm bugs are fixed | integration | `poetry run pytest tests/diagnosis/test_pipeline_stage_trace.py -x` | ✅ (needs inversion) |
| FIXR-02 | `bitbat system reset --yes` deletes data/ and models/ | unit | `poetry run pytest tests/test_cli.py::test_system_reset_command -x` | ❌ Wave 0 |
| FIXR-03 | Realized accuracy > 33% after reset + retrain | integration | `poetry run pytest tests/diagnosis/test_pipeline_stage_trace.py::test_accuracy_exceeds_random_baseline -x` | ❌ Wave 0 (requires inversion) |

### Sampling Rate
- **Per task commit:** `poetry run pytest tests/model/test_train.py tests/model/test_infer.py -x`
- **Per wave merge:** `poetry run pytest`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/model/test_train.py` — add `test_fit_xgb_uses_classification_objective` and `test_fit_xgb_classification_output_shape`
- [ ] `tests/model/test_infer.py` — add `test_predict_bar_returns_three_classes`
- [ ] `tests/autonomous/test_validator.py` — add `test_validator_uses_constructor_tau` (check if file exists)
- [ ] `tests/test_cli.py` — add `test_system_reset_command` (using `tmp_path`)
- [ ] `tests/diagnosis/test_pipeline_stage_trace.py` — invert all 4 existing assertions (rename test functions to describe fixed state)

---

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `src/bitbat/model/train.py` — confirmed `reg:squarederror` at line 53
- Direct code inspection: `src/bitbat/model/infer.py` — confirmed `sign()` at line 86
- Direct code inspection: `src/bitbat/autonomous/validator.py` — confirmed `self.tau = 0.0` at line 46
- Direct code inspection: `src/bitbat/cli.py` — confirmed no existing `reset` command; `model_train` passes `r_forward`
- Live verification: `poetry run python -c "import xgboost; print(xgboost.__version__)"` → 2.1.4
- XGBoost 2.1.4 API verification: `multi:softprob` returns `(n, 3)` float array; `multi:softmax` returns `(n,)` int array
- `ROOT_CAUSE.md` — three bugs with file/line citations and reproducible traces
- `src/bitbat/contracts.py` — label values `{"up", "down", "flat"}` enforced at feature contract boundary
- `src/bitbat/config/default.yaml` — `tau: 0.01`, `autonomous.database_url: "sqlite:///data/autonomous.db"`

### Secondary (MEDIUM confidence)
- XGBoost docs (training data from model): label encoding requirement for multiclass — integer labels `[0, num_class)` required
- `pyproject.toml` pytest config — testpaths, markers confirmed

### Tertiary (LOW confidence)
- None — all claims verified from direct code inspection or live Python verification

---

## Metadata

**Confidence breakdown:**
- Bug fix locations: HIGH — verified by direct code inspection and ROOT_CAUSE.md
- XGBoost API: HIGH — verified by live Python execution in the project environment
- CLI reset design: HIGH — based on existing click patterns in `cli.py`
- Test inversion pattern: HIGH — tests read from source, inversion is straightforward
- Backward compatibility of `predicted_return=None`: MEDIUM — requires checking API and dashboard callers

**Research date:** 2026-03-08
**Valid until:** 2026-04-07 (XGBoost 2.x API is stable; project codebase is the primary source)
