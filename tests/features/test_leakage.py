"""PR-AUC guardrail and no-lookahead tests for feature engineering.

These tests detect train/test information leakage by verifying that:
1. A model trained on random labels cannot achieve suspiciously high PR-AUC.
2. Feature values at row N are computable from data at or before row N.
3. OBV specifically does not use future data.

Referenced by CLAUDE.md under "tests/features/test_leakage.py -- no future data in features".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.dataset.build import _generate_price_features
from bitbat.features.price import obv

pytestmark = pytest.mark.behavioral


def _make_synthetic_prices(n_bars: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic BTC-like OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    # Random walk for close prices starting at ~40000
    log_returns = rng.normal(loc=0.0, scale=0.005, size=n_bars)
    close = 40000.0 * np.exp(np.cumsum(log_returns))

    # Derive OHLCV columns from close
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


def test_no_feature_leakage_pr_auc_guardrail() -> None:
    """A model predicting random labels from features should get PR-AUC near 0.5.

    If PR-AUC exceeds 0.7 it indicates the features contain future information,
    since random labels should be unpredictable from any legitimate features.
    """
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    xgb = pytest.importorskip("xgboost")
    from sklearn.metrics import average_precision_score

    prices = _make_synthetic_prices(n_bars=250, seed=42)
    features = _generate_price_features(prices, enable_garch=False, freq="1h")
    features = features.dropna()

    # Random binary labels -- intentionally unpredictable
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=len(features))

    # Time-based split: first 70% train, last 30% test
    split_idx = int(len(features) * 0.7)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    assert pr_auc < 0.7, (
        f"PR-AUC {pr_auc:.3f} exceeds 0.7 threshold -- features may leak future data. "
        "A model predicting random labels should achieve ~0.5 PR-AUC."
    )


def test_features_no_future_timestamps() -> None:
    """Feature values at row N must be identical whether computed on full or truncated data.

    For each sampled row N, we compute features on the first N+1 rows only and
    verify the result matches the full-dataset computation at row N.
    """
    prices = _make_synthetic_prices(n_bars=100, seed=42)
    features_full = _generate_price_features(prices, enable_garch=False, freq="1h")

    # Sample a few rows past the warm-up period to check
    check_indices = [40, 60, 80, 95]
    for idx in check_indices:
        truncated_prices = prices.iloc[: idx + 1].copy()
        features_trunc = _generate_price_features(truncated_prices, enable_garch=False, freq="1h")

        # Both should have this index present
        ts = prices.index[idx]
        if ts in features_full.index and ts in features_trunc.index:
            full_row = features_full.loc[ts]
            trunc_row = features_trunc.loc[ts]
            # Compare only columns present in both (truncated may have fewer due to NaN)
            common_cols = full_row.index.intersection(trunc_row.index)
            pd_testing.assert_series_equal(
                full_row[common_cols],
                trunc_row[common_cols],
                check_names=False,
                rtol=1e-10,
                obj=f"Feature row at index {idx}",
            )


def test_obv_no_lookahead() -> None:
    """OBV values for bars 0-49 must be identical whether computed on 50 or 100 bars."""
    rng = np.random.default_rng(42)
    n_bars = 100
    index = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")

    # Realistic price movements
    log_returns = rng.normal(loc=0.0, scale=0.005, size=n_bars)
    close = pd.Series(40000.0 * np.exp(np.cumsum(log_returns)), index=index)
    volume = pd.Series(rng.uniform(100, 10000, size=n_bars), index=index)

    obv_full = obv(close, volume)
    obv_half = obv(close.iloc[:50], volume.iloc[:50])

    pd_testing.assert_series_equal(
        obv_full.iloc[:50],
        obv_half,
        check_names=True,
        obj="OBV first 50 bars",
    )
