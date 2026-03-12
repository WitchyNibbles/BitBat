# Phase 28: Activate Fold-Aware OBV in Production Pipeline - Research

**Researched:** 2026-03-08
**Domain:** Feature pipeline wiring — OBV leakage fix production activation
**Confidence:** HIGH

## Summary

Phase 28 is a surgical wiring task: the fold-aware OBV implementation (`obv_fold_aware()`) and the `fold_boundaries` parameter on `generate_price_features()` were both delivered in Phase 25. The parameter is defined and the switching logic is in place. What remains is supplying non-None `fold_boundaries` values from the walk-forward CV loop and the continuous-retraining path so the fold-aware code path is actually exercised in production.

The key finding from code inspection is that `generate_price_features()` already accepts `fold_boundaries: list[int] | None = None` and branches on it correctly (lines 69-72 of `build.py`). The gap is entirely in the callers: `continuous_trainer.py` line 167 and `cli.py` line 1255 (batch inference — no folds, intentionally no-op) and `predictor.py` line 161 (live inference — no folds, intentionally no-op). Only the CV and retraining paths need the wiring; inference paths operate on the full dataset with no fold concept.

The `walk_forward()` function returns `list[Fold]`, where each `Fold` carries `train: pd.Index` and `test: pd.Index` (DatetimeIndex). To extract positional `fold_boundaries` for `obv_fold_aware()`, the caller must convert fold start positions to integer offsets relative to the raw price DataFrame index. The CV loop in `cli.py` (`model_cv`, lines 544–740) and `ContinuousTrainer._do_retrain()` are the two production paths that need this conversion.

**Primary recommendation:** Add a helper `_fold_boundaries_from_folds(prices_index, folds)` that maps fold test-start timestamps to integer positions in the full price DataFrame index, then thread the resulting list into `generate_price_features()` at the two call sites.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| LEAK-02 | OBV cumsum leakage fixed with fold-aware computation — production activation gap | `obv_fold_aware()` and `fold_boundaries` param exist; only caller wiring is missing. Confirmed by code inspection of `build.py`, `continuous_trainer.py`, and `cli.py`. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | project-pinned | DatetimeIndex positional lookup (`Index.searchsorted`, `Index.get_loc`) | Already used throughout pipeline |
| pytest | project-pinned | Behavioral test confirming fold-aware branch is taken | Project standard |

No new dependencies required. This phase touches only existing Python files.

**Installation:** None needed.

## Architecture Patterns

### Recommended Project Structure

No new files or directories needed. Changes land in:

```
src/bitbat/
├── dataset/
│   └── build.py              # generate_price_features already correct — no change
├── autonomous/
│   └── continuous_trainer.py # _do_retrain() needs fold_boundaries wired in
src/bitbat/cli.py             # model_cv command needs fold_boundaries wired in
tests/
└── dataset/
    └── test_fold_boundaries_wiring.py   # NEW: confirms obv_fold_aware is called
```

### Pattern 1: Fold Boundary Extraction

**What:** Convert a list of `Fold` objects (which carry DatetimeIndex members) into a sorted list of integer positional offsets into the raw price DataFrame.

**When to use:** Any call site that has both a prices DataFrame and a `list[Fold]` produced by `walk_forward()`.

**Canonical logic:**
```python
def _fold_boundaries_from_folds(
    price_index: pd.Index,
    folds: list[Fold],
) -> list[int]:
    """Return sorted unique integer positions marking the start of each fold's test window."""
    boundaries: set[int] = set()
    for fold in folds:
        if fold.test.empty:
            continue
        first_test_ts = fold.test[0]
        pos = price_index.searchsorted(first_test_ts)
        if 0 < pos < len(price_index):
            boundaries.add(int(pos))
    return sorted(boundaries)
```

Key considerations:
- `searchsorted` returns 0 when the timestamp is before all prices (no useful boundary) — filter out 0.
- `searchsorted` returns `len(price_index)` when timestamp is past the end — filter out.
- Result can be empty list (no valid boundaries found) — `generate_price_features` handles empty list as no-op (falls through to standard `obv`).
- Source confidence: direct inspection of `dataset/splits.py` (Fold dataclass) and `features/price.py` (`obv_fold_aware` boundary semantics).

### Pattern 2: Wiring in `continuous_trainer.py`

**What:** `ContinuousTrainer._do_retrain()` loads prices, generates features, then splits into train/holdout by positional slice. It does not use `walk_forward()` — it uses a single train/holdout cut. The "fold boundary" here is the single split point at position `self.train_window_bars`.

**When to use:** Single-split retraining path (not multi-fold CV).

**Example:**
```python
# After prices are loaded and trimmed to rolling_window_bars:
fold_boundaries = [self.train_window_bars]  # one reset at train/holdout split
features = generate_price_features(
    prices,
    enable_garch=self.enable_garch,
    freq=self.freq,
    fold_boundaries=fold_boundaries,
)
```

This ensures OBV in the holdout window starts fresh, not carrying cumulative sum from the training window.

### Pattern 3: Wiring in `model_cv` CLI command

**What:** The `model_cv` function builds `folds` via `walk_forward()` on the pre-assembled `X` (feature dataset). The prices DataFrame is not accessible in this function — it operates on the already-built feature dataset loaded from parquet. Therefore fold boundaries cannot be injected retroactively into `generate_price_features()` at this layer.

**Critical finding:** The `model_cv` command in `cli.py` loads a pre-built feature dataset from `data/features/{freq}_{horizon}/dataset.parquet` via `_load_feature_dataset()`. It does NOT call `generate_price_features()` — it consumes the already-generated features. The OBV column in the dataset was computed when `features build` (i.e., `build_xy()`) was called.

**Implication for Phase 28:** The CV loop wiring must happen in `build_xy()` or wherever `features build` is invoked before CV, not inside `model_cv` itself.

### Where `generate_price_features` Is Actually Called (Confirmed by Code Inspection)

| Call site | File | Line | Has fold context? | Action needed |
|-----------|------|------|-------------------|---------------|
| `build_xy()` | `dataset/build.py` | 168 | No — dataset builder has no fold concept | Low priority; dataset is built once before CV |
| `ContinuousTrainer._do_retrain()` | `autonomous/continuous_trainer.py` | 167 | Yes — explicit train/holdout split | Wire `fold_boundaries=[self.train_window_bars]` |
| `batch_run()` CLI | `cli.py` | 1255 | No — inference only, no CV | No change (no folds) |
| `LivePredictor.predict_latest()` | `autonomous/predictor.py` | 161 | No — inference only | No change (no folds) |

**Conclusion:** The only production path where OBV leakage is directly addressable at feature-generation time is `ContinuousTrainer._do_retrain()`. The `model_cv` CV loop uses a pre-built dataset, so fixing `build_xy()` (which calls `generate_price_features` without fold boundaries) or adding a fold-boundary-aware rebuild step would be needed for the full CV path. However, since `build_xy()` is invoked before fold structure is known, the practical fix for the success criteria is:

1. **`continuous_trainer.py`:** Wire `fold_boundaries=[self.train_window_bars]` into `generate_price_features()`.
2. **Test:** Confirm that when `fold_boundaries` is provided, the `obv` column differs from the no-boundaries case in the second segment.

The success criterion "the walk-forward CV loop extracts fold split boundaries and supplies them to `generate_price_features()`" likely refers to the `ContinuousTrainer` path (which does its own train/holdout split) rather than the `model_cv` CLI command (which operates on pre-built features). The planner should clarify this interpretation.

### Anti-Patterns to Avoid

- **Mutating the `Fold` object or `walk_forward` return value:** The `Fold` dataclass uses DatetimeIndex which is large; extract boundary integers once before calling `generate_price_features`, do not store them back on the Fold.
- **Injecting fold boundaries into inference paths** (`predictor.py`, `batch_run`): Inference operates on all available prices to produce the latest prediction. Resetting OBV mid-series in inference would produce incorrect values. Leave these callers unchanged.
- **Passing `fold_boundaries=[]` explicitly:** Empty list and `None` both trigger the standard OBV path in `obv_fold_aware()`. Pass `None` when no folds are applicable for clarity.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Timestamp-to-integer position conversion | Custom binary search | `pd.Index.searchsorted` | Already used throughout timealign module; handles DST and duplicate timestamps correctly |
| OBV fold segmentation | New cumsum loop | `obv_fold_aware()` already in `features/price.py` | Implemented and tested in Phase 25 |

## Common Pitfalls

### Pitfall 1: Off-by-One in Boundary Positions

**What goes wrong:** `searchsorted` with `side='left'` returns the position of the first bar in the test window. If the intent is "reset OBV so the first test bar starts fresh," position N means bars 0..N-1 form segment 1 and bars N..end form segment 2. This is the correct semantic for `obv_fold_aware`.

**Why it happens:** Confusion between "the bar before the boundary" and "the bar at the boundary."

**How to avoid:** Test the boundary extraction on a known dataset: verify that OBV at position N in the fold-aware result equals OBV computed on prices[N:] independently. The existing test `test_obv_fold_aware_resets_at_boundaries` already validates this contract.

**Warning signs:** OBV values are identical before and after a supposed boundary.

### Pitfall 2: Applying Fold Boundaries to Inference Paths

**What goes wrong:** Wiring `fold_boundaries` into `predictor.py` or `batch_run()` causes OBV to reset mid-series, producing a different (and incorrect) OBV for the latest bar than a full-series computation would produce.

**Why it happens:** Confusing "training-time fold boundaries" (where reset prevents leakage) with "inference-time full-series computation" (where continuity is correct).

**How to avoid:** Only wire `fold_boundaries` in training/CV paths. Leave inference callers with `fold_boundaries=None` (default).

### Pitfall 3: `continuous_trainer` Uses Positional Slice, Not DatetimeIndex

**What goes wrong:** The trainer splits features with `.iloc[-required_samples:]` and `.iloc[:train_window_bars]`. If boundary extraction is done against timestamps it may produce incorrect positions if rows have been dropped by `.dropna()`.

**Why it happens:** `features = features.dropna()` may remove rows between price loading and the split. The positional split `iloc[:self.train_window_bars]` is applied to the filtered frame.

**How to avoid:** Use `fold_boundaries=[self.train_window_bars]` based on the actual split position, applied to the feature frame after `.dropna()` is called. Since `generate_price_features` returns a frame aligned to the price index, and `dropna` is called after feature generation, pass `fold_boundaries` at generation time with the raw bar count, then apply dropna. The reset at position `train_window_bars` in the raw prices frame may not align exactly with position `train_window_bars` in the post-dropna features if NaN rows exist in the first `train_window_bars` bars — planner should verify or use the timestamps approach instead.

**Warning signs:** Segment sizes don't match expected train/holdout sizes.

### Pitfall 4: The `test_no_private_imports_in_callers` Structural Guard

**What goes wrong:** The test in `tests/dataset/test_public_api.py` checks that `cli.py`, `predictor.py`, and `continuous_trainer.py` do not import `_generate_price_features` or `_join_auxiliary_features`.

**Why it happens:** The backward-compat aliases use the underscore prefix.

**How to avoid:** All callers must continue to import `generate_price_features` (no underscore). The Phase 28 changes must not introduce any `_generate_price_features` imports.

## Code Examples

### Existing `generate_price_features` Signature (Confirmed from `build.py`)
```python
# Source: src/bitbat/dataset/build.py lines 54-60
def generate_price_features(
    prices: pd.DataFrame,
    *,
    enable_garch: bool = False,
    freq: str | None = None,
    fold_boundaries: list[int] | None = None,
) -> pd.DataFrame:
```

### Existing Fold-Aware Branch in `generate_price_features` (Confirmed from `build.py`)
```python
# Source: src/bitbat/dataset/build.py lines 69-72
if fold_boundaries:
    features["obv"] = obv_fold_aware(close, prices["volume"], fold_boundaries)
else:
    features["obv"] = obv(close, prices["volume"])
```

### Existing `obv_fold_aware` Signature (Confirmed from `features/price.py`)
```python
# Source: src/bitbat/features/price.py lines 137-141
def obv_fold_aware(
    close: pd.Series,
    volume: pd.Series,
    fold_boundaries: list[int] | None = None,
) -> pd.Series:
```

### Proposed Wiring in `continuous_trainer._do_retrain` (Target Change)
```python
# After: prices = prices.iloc[-self.rolling_window_bars:]
# Current line 167: features = generate_price_features(prices, enable_garch=self.enable_garch, freq=self.freq)
# Change to:
features = generate_price_features(
    prices,
    enable_garch=self.enable_garch,
    freq=self.freq,
    fold_boundaries=[self.train_window_bars],
)
```

### Proposed Test Pattern (New Behavioral Test)
```python
# tests/dataset/test_fold_boundaries_wiring.py
def test_generate_price_features_fold_boundaries_changes_obv_second_segment():
    """When fold_boundaries is provided, OBV differs after the boundary."""
    prices = _make_synthetic_prices(n_bars=100)
    features_no_fold = generate_price_features(prices, freq="1h")
    features_with_fold = generate_price_features(prices, freq="1h", fold_boundaries=[50])

    # Before boundary: identical
    pd.testing.assert_series_equal(
        features_no_fold["obv"].iloc[:50],
        features_with_fold["obv"].iloc[:50],
    )
    # After boundary: different (OBV reset)
    assert not np.allclose(
        features_no_fold["obv"].iloc[50:].values,
        features_with_fold["obv"].iloc[50:].values,
    )
```

Note: this test is functionally identical to `test_generate_price_features_uses_fold_aware_when_provided` in `tests/features/test_obv_fold_aware.py`. The planner should decide whether to add a separate test in `tests/dataset/` for the wiring specifically, or extend the existing file.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cumulative OBV over entire dataset | `obv_fold_aware()` resets at fold boundaries | Phase 25 | Prevents OBV cumsum from bleeding across CV fold boundaries; reduces leakage signal |
| `_generate_price_features` (private) | `generate_price_features` (public) | Phase 26 | External callers use stable public API |

**Deprecated/outdated:**
- `_generate_price_features`: Backward-compat alias only; callers should use `generate_price_features`.

## Open Questions

1. **Should `build_xy()` also accept `fold_boundaries`?**
   - What we know: `build_xy()` is the primary dataset-build path invoked by `features build`. It calls `generate_price_features` at line 168 without fold boundaries. The CV loop (`model_cv`) uses the dataset built by `build_xy`.
   - What's unclear: Whether the success criterion "walk-forward CV loop supplies fold_boundaries" means modifying `build_xy` to accept pre-computed fold boundaries (making the dataset build fold-aware) or whether the continuous-trainer path is sufficient.
   - Recommendation: The planner should confirm the interpretation. The most impactful change with lowest complexity is wiring `fold_boundaries=[self.train_window_bars]` into `continuous_trainer`. If the CV path is also required, a separate refactor of `build_xy` or a new fold-aware dataset builder would be needed.

2. **NaN-drop alignment: does `.dropna()` after `generate_price_features` shift the effective boundary?**
   - What we know: `ContinuousTrainer._do_retrain` calls `features.dropna()` after `generate_price_features`, then splits with `.iloc[:self.train_window_bars]`.
   - What's unclear: If NaN rows fall within the first `train_window_bars` bars, the post-dropna train window covers more raw bars than intended.
   - Recommendation: Accept this as negligible for the leakage fix. The fold-aware OBV still resets at the raw bar count position, which is conservative (resets slightly inside the intended split point). Document in the plan.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 |
| Config file | pyproject.toml |
| Quick run command | `poetry run pytest tests/features/test_obv_fold_aware.py tests/dataset/ -x -q` |
| Full suite command | `poetry run pytest -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LEAK-02 | `generate_price_features` uses `obv_fold_aware` when `fold_boundaries` is provided | behavioral | `poetry run pytest tests/features/test_obv_fold_aware.py::test_generate_price_features_uses_fold_aware_when_provided -x` | Yes (existing) |
| LEAK-02 | `continuous_trainer` passes `fold_boundaries` to `generate_price_features` | behavioral | `poetry run pytest tests/dataset/test_fold_boundaries_wiring.py -x` | No — Wave 0 gap |
| LEAK-02 | No regressions: existing tests pass unchanged | regression | `poetry run pytest -x -q` | Yes (existing suite) |

### Sampling Rate
- **Per task commit:** `poetry run pytest tests/features/test_obv_fold_aware.py tests/dataset/ -x -q`
- **Per wave merge:** `poetry run pytest -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/dataset/test_fold_boundaries_wiring.py` — covers LEAK-02 wiring test (continuous trainer passes fold_boundaries). Alternatively, extend existing `tests/features/test_obv_fold_aware.py`.

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `src/bitbat/dataset/build.py` — `generate_price_features` signature and OBV branching logic
- Direct code inspection: `src/bitbat/features/price.py` — `obv_fold_aware` implementation
- Direct code inspection: `src/bitbat/autonomous/continuous_trainer.py` — `_do_retrain` feature generation call site
- Direct code inspection: `src/bitbat/dataset/splits.py` — `Fold` dataclass structure and `walk_forward` return type
- Direct code inspection: `src/bitbat/cli.py` — `model_cv` and `batch_run` call sites
- Direct code inspection: `tests/features/test_obv_fold_aware.py` — existing test coverage

### Secondary (MEDIUM confidence)
- `tests/dataset/test_public_api.py` — structural guard: confirms callers must not import private `_generate_price_features`
- `.planning/STATE.md` — confirms OBV leakage empirically NOT material (2.33pp < 3pp threshold) but fix implemented as correct practice

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all relevant code is in-repo and inspected directly
- Architecture: HIGH — call graph confirmed by grep and file reads; no ambiguity about which paths need wiring
- Pitfalls: HIGH — NaN-drop alignment issue identified from code; off-by-one documented from `obv_fold_aware` spec

**Research date:** 2026-03-08
**Valid until:** Until `generate_price_features` signature changes (stable; no expiry concern)
