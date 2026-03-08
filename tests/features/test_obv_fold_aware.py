"""Fold-aware OBV correctness tests.

Validates that ``obv_fold_aware()`` resets the cumulative sum at fold
boundaries and that ``_generate_price_features()`` uses the fold-aware
variant when ``fold_boundaries`` is provided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.dataset.build import _generate_price_features
from bitbat.features.price import obv, obv_fold_aware

pytestmark = pytest.mark.behavioral


def _make_price_data(n_bars: int = 100, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Return (close, volume) series with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    log_returns = rng.normal(loc=0.0, scale=0.005, size=n_bars)
    close = pd.Series(40000.0 * np.exp(np.cumsum(log_returns)), index=index)
    volume = pd.Series(rng.uniform(100, 10000, size=n_bars), index=index)
    return close, volume


def _make_synthetic_prices(n_bars: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic BTC-like OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    log_returns = rng.normal(loc=0.0, scale=0.005, size=n_bars)
    close = 40000.0 * np.exp(np.cumsum(log_returns))
    high = close * (1.0 + rng.uniform(0.001, 0.01, size=n_bars))
    low = close * (1.0 - rng.uniform(0.001, 0.01, size=n_bars))
    open_ = close * (1.0 + rng.normal(0, 0.003, size=n_bars))
    volume = rng.uniform(100, 10000, size=n_bars)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def test_obv_fold_aware_no_boundaries_matches_standard() -> None:
    """obv_fold_aware with no boundaries must produce identical output to obv."""
    close, volume = _make_price_data(100)

    obv_standard = obv(close, volume)
    obv_fa = obv_fold_aware(close, volume)

    pd_testing.assert_series_equal(obv_standard, obv_fa, check_names=True)


def test_obv_fold_aware_resets_at_boundaries() -> None:
    """Bars 50-99 from fold-aware OBV must match independent OBV on bars 50-99."""
    close, volume = _make_price_data(100)

    obv_fa = obv_fold_aware(close, volume, fold_boundaries=[0, 50, 100])

    # Independent OBV for bars 50-99 only
    obv_independent = obv(close.iloc[50:], volume.iloc[50:])

    pd_testing.assert_series_equal(
        obv_fa.iloc[50:].reset_index(drop=True),
        obv_independent.reset_index(drop=True),
        check_names=False,
        obj="Second segment matches independent computation",
    )


def test_obv_fold_aware_single_boundary_at_zero() -> None:
    """A single boundary at 0 should behave identically to no boundaries."""
    close, volume = _make_price_data(100)

    obv_standard = obv(close, volume)
    obv_fa = obv_fold_aware(close, volume, fold_boundaries=[0])

    pd_testing.assert_series_equal(obv_standard, obv_fa, check_names=True)


def test_generate_price_features_uses_fold_aware_when_provided() -> None:
    """_generate_price_features with fold_boundaries differs from without."""
    prices = _make_synthetic_prices(n_bars=100)

    features_default = _generate_price_features(prices, freq="1h")
    features_fold = _generate_price_features(prices, freq="1h", fold_boundaries=[0, 50])

    # The OBV column should differ for bars after the boundary
    obv_default = features_default["obv"]
    obv_fold = features_fold["obv"]

    # First 50 bars should be identical (same segment, no reset)
    pd_testing.assert_series_equal(
        obv_default.iloc[:50],
        obv_fold.iloc[:50],
        check_names=True,
        obj="First segment identical",
    )

    # Bars after boundary 50 should differ (reset vs carry-forward)
    second_half_default = obv_default.iloc[50:].values
    second_half_fold = obv_fold.iloc[50:].values
    assert not np.allclose(
        second_half_default, second_half_fold
    ), "Expected OBV to differ after fold boundary when fold_boundaries is provided"
