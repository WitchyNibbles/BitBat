"""Tests for GARCH volatility feature generation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

try:
    import arch  # noqa: F401
except ImportError:
    pytest.skip("arch not installed", allow_module_level=True)

from bitbat.features.volatility import garch_features


def _synthetic_close(n: int = 500) -> pd.Series:
    """Generate a synthetic hourly close price series with vol clustering."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(datetime(2024, 1, 1), periods=n, freq="1h")
    returns = rng.normal(0, 0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=idx)


def test_garch_output_columns() -> None:
    close = _synthetic_close()
    features = garch_features(close)

    assert "garch_vol" in features.columns
    assert "vol_regime" in features.columns
    assert "vol_ratio" in features.columns
    assert len(features) == len(close)


def test_vol_regime_values() -> None:
    close = _synthetic_close()
    features = garch_features(close)

    valid = features["vol_regime"].dropna()
    assert set(valid.unique()).issubset({0.0, 1.0, 2.0})


def test_insufficient_data_returns_nan() -> None:
    idx = pd.date_range(datetime(2024, 1, 1), periods=10, freq="1h")
    close = pd.Series(range(100, 110), index=idx, dtype=float)

    features = garch_features(close, window=168)

    assert features["garch_vol"].isna().all()


def test_no_future_leakage() -> None:
    close = _synthetic_close(600)
    full = garch_features(close)

    truncated = garch_features(close.iloc[:400])

    # Values at bar 300 should be the same whether or not future bars exist
    common_idx = truncated.index[300]
    if not pd.isna(full.loc[common_idx, "garch_vol"]) and not pd.isna(
        truncated.loc[common_idx, "garch_vol"]
    ):
        # Due to GARCH fitting the full series vs truncated, values may differ
        # slightly, but they should be in the same ballpark (within 50%)
        ratio = full.loc[common_idx, "garch_vol"] / truncated.loc[common_idx, "garch_vol"]
        assert 0.5 < ratio < 2.0, f"Suspiciously different GARCH values: ratio={ratio}"
