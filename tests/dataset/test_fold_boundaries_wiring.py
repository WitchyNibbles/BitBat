"""Behavioral test confirming fold_boundaries wiring in the feature pipeline.

Verifies that generate_price_features() uses obv_fold_aware() when
fold_boundaries is provided, producing different OBV values after the split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.dataset.build import generate_price_features

pytestmark = pytest.mark.behavioral


def _make_synthetic_prices(n_bars: int = 100, seed: int = 42) -> pd.DataFrame:
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


def test_generate_price_features_fold_boundaries_changes_obv_second_segment() -> None:
    """When fold_boundaries=[50] is provided, OBV after position 50 must differ from standard."""
    prices = _make_synthetic_prices(n_bars=100)
    features_no_fold = generate_price_features(prices, freq="1h")
    features_with_fold = generate_price_features(prices, freq="1h", fold_boundaries=[50])

    # First 50 bars: identical (both segments start from the same data, no reset yet)
    pd_testing.assert_series_equal(
        features_no_fold["obv"].iloc[:50],
        features_with_fold["obv"].iloc[:50],
        check_names=True,
        obj="OBV before boundary must be identical",
    )

    # After boundary: must differ (fold-aware resets; standard carries forward)
    assert not np.allclose(
        features_no_fold["obv"].iloc[50:].values,
        features_with_fold["obv"].iloc[50:].values,
    ), "Expected OBV to differ after fold boundary when fold_boundaries=[50] is provided"
