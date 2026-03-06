"""Empirical assessment of OBV cumsum fold-boundary leakage.

This module measures the impact of OBV cumsum leakage on model performance
by comparing walk-forward accuracy with and without OBV features.  It also
proves mechanically that cumsum carries state across fold boundaries.

The assessment (test 1) always passes -- it is a measurement, not a gate.
The mechanical proof (test 2) always passes -- it shows the leakage exists.
Test 3 validates the fold-aware OBV fix from ``obv_fold_aware()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.dataset.build import _generate_price_features
from bitbat.features.price import obv, obv_fold_aware

pytestmark = pytest.mark.behavioral


def _make_synthetic_prices(n_bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic BTC-like OHLCV DataFrame for leakage assessment."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.0, scale=0.005, size=n_bars)
    close = 40000.0 * np.exp(np.cumsum(log_returns))

    high = close * (1.0 + rng.uniform(0.001, 0.01, size=n_bars))
    low = close * (1.0 - rng.uniform(0.001, 0.01, size=n_bars))
    open_ = close * (1.0 + rng.normal(0, 0.003, size=n_bars))
    volume = rng.uniform(100, 10000, size=n_bars)

    index = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_obv_leakage_impact_assessment() -> None:
    """Empirically measure directional accuracy impact of OBV cumsum leakage.

    Approach: run walk-forward evaluation twice (with OBV, without OBV) and
    compare average directional accuracy across three time-series folds.

    Assessment criterion: if absolute difference > 3 percentage points, the
    leakage is considered material.  The test records the result but always
    passes.
    """
    xgb = pytest.importorskip("xgboost")

    prices = _make_synthetic_prices(n_bars=2000, seed=42)

    # Generate features with and without OBV
    features_with_obv = _generate_price_features(prices, enable_garch=False, freq="1h")
    features_without_obv = features_with_obv.drop(columns=["obv"])

    # Create forward return labels: 4-bar horizon, shifted to align
    forward_returns = prices["close"].pct_change(4).shift(-4)
    labels = (forward_returns > 0).astype(int)

    # Drop NaN rows from features and align labels
    valid_with = features_with_obv.dropna().index.intersection(labels.dropna().index)
    valid_without = features_without_obv.dropna().index.intersection(labels.dropna().index)

    # Use common valid rows for fair comparison
    common_valid = valid_with.intersection(valid_without)
    X_with = features_with_obv.loc[common_valid]
    X_without = features_without_obv.loc[common_valid]
    y = labels.loc[common_valid]

    # 3-fold time-series split: train on first N, test on next M
    folds = [
        (slice(0, 600), slice(600, 800)),
        (slice(0, 800), slice(800, 1000)),
        (slice(0, 1000), slice(1000, 1200)),
    ]

    acc_with_obv: list[float] = []
    acc_without_obv: list[float] = []

    for train_sl, test_sl in folds:
        X_train_w = X_with.iloc[train_sl]
        X_test_w = X_with.iloc[test_sl]
        X_train_wo = X_without.iloc[train_sl]
        X_test_wo = X_without.iloc[test_sl]
        y_train = y.iloc[train_sl]
        y_test = y.iloc[test_sl]

        if len(y_test) == 0:
            continue

        # With OBV
        model_w = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        model_w.fit(X_train_w, y_train)
        preds_w = model_w.predict(X_test_w)
        acc_w = (preds_w == y_test.values).mean()
        acc_with_obv.append(acc_w)

        # Without OBV
        model_wo = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        model_wo.fit(X_train_wo, y_train)
        preds_wo = model_wo.predict(X_test_wo)
        acc_wo = (preds_wo == y_test.values).mean()
        acc_without_obv.append(acc_wo)

    mean_with = np.mean(acc_with_obv)
    mean_without = np.mean(acc_without_obv)
    diff = abs(mean_with - mean_without)

    is_material = diff > 0.03

    # Record results via print for test output inspection
    print(f"\n--- OBV Leakage Impact Assessment ---")
    print(f"Mean accuracy WITH OBV:    {mean_with:.4f}")
    print(f"Mean accuracy WITHOUT OBV: {mean_without:.4f}")
    print(f"Absolute difference:       {diff:.4f}")
    print(f"Material (>3pp)?           {is_material}")
    print(f"Per-fold WITH OBV:         {[f'{a:.4f}' for a in acc_with_obv]}")
    print(f"Per-fold WITHOUT OBV:      {[f'{a:.4f}' for a in acc_without_obv]}")

    if not is_material:
        print("RESULT: OBV fold-boundary leakage is NOT material (<3pp).")
        print("The fold-aware fix is still implemented as correct practice.")
    else:
        print("RESULT: OBV fold-boundary leakage IS material (>3pp).")
        print("The fold-aware fix is critical for model integrity.")


def test_obv_cumsum_leaks_across_fold_boundary() -> None:
    """Prove that the current OBV implementation leaks information across folds.

    Single cumsum over 100 bars produces different values for bars 50-99
    compared to computing OBV on bars 50-99 independently, because the
    single cumsum carries forward the accumulated state from bars 0-49.
    """
    rng = np.random.default_rng(42)
    n_bars = 100
    index = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")

    log_returns = rng.normal(loc=0.0, scale=0.005, size=n_bars)
    close = pd.Series(40000.0 * np.exp(np.cumsum(log_returns)), index=index)
    volume = pd.Series(rng.uniform(100, 10000, size=n_bars), index=index)

    # Single cumsum over all 100 bars
    obv_full = obv(close, volume)

    # Independent cumsum for bars 50-99 only
    obv_second_half_independent = obv(close.iloc[50:], volume.iloc[50:])

    # The values MUST differ -- that's the leakage
    full_second_half = obv_full.iloc[50:].values
    independent_second_half = obv_second_half_independent.values

    # At least some values should differ because of carried-forward state
    assert not np.allclose(full_second_half, independent_second_half), (
        "Expected OBV values to differ between full-series and independent "
        "computation for bars 50-99, but they were identical. This means "
        "OBV cumsum does NOT carry state across boundaries -- no leakage."
    )


def test_obv_fold_aware_resets_cumsum() -> None:
    """Verify that obv_fold_aware resets cumsum at fold boundaries.

    Bars 50-99 from obv_fold_aware with boundary at 50 must equal
    computing OBV on bars 50-99 in isolation.
    """
    rng = np.random.default_rng(42)
    n_bars = 100
    index = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")

    log_returns = rng.normal(loc=0.0, scale=0.005, size=n_bars)
    close = pd.Series(40000.0 * np.exp(np.cumsum(log_returns)), index=index)
    volume = pd.Series(rng.uniform(100, 10000, size=n_bars), index=index)

    # Fold-aware OBV with boundary at bar 50
    obv_fa = obv_fold_aware(close, volume, fold_boundaries=[0, 50, 100])

    # Independent OBV for bars 50-99
    obv_independent = obv(close.iloc[50:], volume.iloc[50:])

    pd_testing.assert_series_equal(
        obv_fa.iloc[50:].reset_index(drop=True),
        obv_independent.reset_index(drop=True),
        check_names=False,
        obj="OBV fold-aware second segment vs independent",
    )
