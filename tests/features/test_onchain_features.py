"""Tests for on-chain feature generation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from bitbat.features.onchain import generate_onchain_features


def _synthetic_onchain(n_days: int = 100) -> pd.DataFrame:
    """Build a synthetic daily on-chain DataFrame."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(datetime(2024, 1, 1), periods=n_days, freq="D")
    return pd.DataFrame({
        "date": idx,
        "hashrate": 500e6 + rng.normal(0, 1e6, n_days).cumsum(),
        "tx_count": 300000 + rng.normal(0, 5000, n_days).cumsum(),
        "mempool_size": 50000 + rng.normal(0, 2000, n_days).cumsum().clip(min=1000),
        "avg_block_size": 1.5 + rng.normal(0, 0.01, n_days).cumsum(),
    })


def test_output_columns_have_onchain_prefix() -> None:
    onchain = _synthetic_onchain()
    features = generate_onchain_features(onchain, freq="1h")

    for col in features.columns:
        assert col.startswith("onchain_"), f"Column {col} missing onchain_ prefix"


def test_hourly_resampling() -> None:
    onchain = _synthetic_onchain(10)
    features = generate_onchain_features(onchain, freq="1h")

    assert len(features) >= 200  # 10 days * 24 hours


def test_z_scores_present() -> None:
    onchain = _synthetic_onchain()
    features = generate_onchain_features(onchain, freq="1h")

    assert "onchain_hashrate_z" in features.columns
    assert "onchain_tx_count_z" in features.columns
    assert "onchain_mempool_z" in features.columns


def test_change_features_present() -> None:
    onchain = _synthetic_onchain()
    features = generate_onchain_features(onchain, freq="1h")

    assert "onchain_hashrate_change" in features.columns
    assert "onchain_block_size_change" in features.columns
