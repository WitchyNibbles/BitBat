from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bitbat.dataset.build import build_xy

pytestmark = pytest.mark.integration

def _make_prices(start: datetime, periods: int = 60) -> pd.DataFrame:
    timestamps = [start + timedelta(hours=i) for i in range(periods)]
    close = (
        np.linspace(100, 120, periods)
        + np.random.default_rng(0).normal(0, 1, periods)
    )
    high = close + np.random.default_rng(1).normal(1, 0.5, periods)
    low = close - np.random.default_rng(2).normal(1, 0.5, periods)
    volume = np.random.default_rng(3).integers(100, 200, periods)
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(timestamps),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _make_news(start: datetime, periods: int = 60) -> pd.DataFrame:
    timestamps = [start + timedelta(hours=i) for i in range(periods)]
    titles = [f"Headline {i}" for i in range(periods)]
    scores = np.sin(np.linspace(0, 4 * np.pi, periods))
    return pd.DataFrame({
        "published_utc": pd.to_datetime(timestamps),
        "title": titles,
        "url": [f"http://example.com/{i}" for i in range(periods)],
        "source": "UnitTest",
        "lang": "en",
        "sentiment_score": scores,
    })


def test_build_xy_shapes_and_outputs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    start = datetime(2024, 1, 1)
    prices = _make_prices(start)
    news = _make_news(start)

    prices_path = tmp_path / "prices.parquet"
    news_path = tmp_path / "news.parquet"
    prices.to_parquet(prices_path)
    news.to_parquet(news_path)

    output_dir = tmp_path / "data"
    monkeypatch.chdir(tmp_path)

    X, y, meta = build_xy(
        prices_parquet=prices_path,
        news_parquet=news_path,
        freq="1h",
        horizon="2h",
        start="2024-01-01 05:00:00",
        end="2024-01-03 00:00:00",
    )

    assert not X.empty
    assert len(X) == len(y)
    assert X.isna().sum().max() == 0
    assert all(column.startswith("feat_") for column in X.columns)
    # y should be float64 forward returns (regression)
    assert y.dtype == np.float64

    dataset_path = output_dir / "features" / "1h_2h" / "dataset.parquet"
    assert dataset_path.exists()
    dataset = pd.read_parquet(dataset_path)
    assert "label" in dataset.columns
    assert "r_forward" in dataset.columns
    assert dataset["label"].isna().sum() == 0
    expected_labels = np.where(
        dataset["r_forward"] > 0.0,
        "up",
        np.where(dataset["r_forward"] < 0.0, "down", "flat"),
    )
    assert (dataset["label"].to_numpy() == expected_labels).all()
    assert dataset["timestamp_utc"].is_monotonic_increasing
    assert dataset["timestamp_utc"].is_unique
    feature_columns = [
        column for column in dataset.columns if column.startswith("feat_")
    ]
    assert feature_columns
    assert all(column.startswith("feat_") for column in feature_columns)

    meta_path = output_dir / "features" / "1h_2h" / "meta.json"
    assert meta_path.exists()
    saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert saved_meta["freq"] == "1h"
    assert saved_meta["horizon"] == "2h"
    assert saved_meta["seed"] is None
    assert saved_meta["version"] == "unknown"
    # New regression meta fields
    assert "target_mean" in saved_meta
    assert "target_std" in saved_meta


def test_build_xy_triple_barrier_label_mode_compatibility(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    start = datetime(2024, 1, 1)
    prices = _make_prices(start)
    news = _make_news(start)

    prices_path = tmp_path / "prices.parquet"
    news_path = tmp_path / "news.parquet"
    prices.to_parquet(prices_path)
    news.to_parquet(news_path)

    monkeypatch.chdir(tmp_path)

    output_default = tmp_path / "data_default"
    output_barrier = tmp_path / "data_barrier"

    X_default, y_default, _ = build_xy(
        prices_parquet=prices_path,
        news_parquet=news_path,
        freq="1h",
        horizon="2h",
        start="2024-01-01 05:00:00",
        end="2024-01-03 00:00:00",
        output_root=output_default,
        label_mode="return_direction",
    )

    X_barrier, y_barrier, _ = build_xy(
        prices_parquet=prices_path,
        news_parquet=news_path,
        freq="1h",
        horizon="2h",
        start="2024-01-01 05:00:00",
        end="2024-01-03 00:00:00",
        output_root=output_barrier,
        label_mode="triple_barrier",
        barrier_take_profit=0.01,
        barrier_stop_loss=0.01,
    )

    assert not X_default.empty
    assert not X_barrier.empty
    assert list(X_default.columns) == list(X_barrier.columns)
    assert y_default.dtype == np.float64
    assert y_barrier.dtype == np.float64

    default_dataset = pd.read_parquet(
        output_default / "features" / "1h_2h" / "dataset.parquet"
    )
    barrier_dataset = pd.read_parquet(
        output_barrier / "features" / "1h_2h" / "dataset.parquet"
    )

    assert set(default_dataset["label"].unique()).issubset({"up", "down", "flat"})
    assert set(barrier_dataset["label"].unique()).issubset(
        {"take_profit", "stop_loss", "timeout"}
    )
