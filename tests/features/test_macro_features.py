"""Tests for macroeconomic feature generation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from bitbat.features.macro import generate_macro_features


def _synthetic_macro(n_days: int = 100) -> pd.DataFrame:
    """Build a synthetic daily macro DataFrame."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(datetime(2024, 1, 1), periods=n_days, freq="D")
    return pd.DataFrame({
        "date": idx,
        "fed_funds_rate": 5.25 + rng.normal(0, 0.01, n_days).cumsum(),
        "treasury_10y": 4.50 + rng.normal(0, 0.02, n_days).cumsum(),
        "usd_index": 104.0 + rng.normal(0, 0.1, n_days).cumsum(),
        "vix": 15.0 + rng.normal(0, 0.3, n_days).cumsum().clip(min=10),
        "inflation_5y_breakeven": 2.3 + rng.normal(0, 0.005, n_days).cumsum(),
    })


def test_output_columns_have_macro_prefix() -> None:
    macro = _synthetic_macro()
    features = generate_macro_features(macro, freq="1h")

    for col in features.columns:
        assert col.startswith("macro_"), f"Column {col} missing macro_ prefix"


def test_hourly_resampling() -> None:
    macro = _synthetic_macro(10)
    features = generate_macro_features(macro, freq="1h")

    # 10 days → ~240 hourly bars (10*24)
    assert len(features) >= 200


def test_yield_spread_computed() -> None:
    macro = _synthetic_macro()
    features = generate_macro_features(macro, freq="1h")

    assert "macro_yield_spread" in features.columns
    # Yield spread = treasury_10y - fed_funds_rate
    valid = features["macro_yield_spread"].dropna()
    assert len(valid) > 0


def test_vix_level_passthrough() -> None:
    macro = _synthetic_macro()
    features = generate_macro_features(macro, freq="1h")

    assert "macro_vix_level" in features.columns
    assert features["macro_vix_level"].dropna().gt(0).all()


def test_z_scores_exist_for_each_series() -> None:
    macro = _synthetic_macro()
    features = generate_macro_features(macro, freq="1h")

    for col in ["fed_funds_rate", "treasury_10y", "usd_index", "vix", "inflation_5y_breakeven"]:
        z_col = f"macro_{col}_z"
        assert z_col in features.columns, f"Missing z-score column: {z_col}"
