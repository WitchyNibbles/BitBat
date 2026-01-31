from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bitbat.dataset.build import build_xy


def _make_prices(start: datetime, periods: int = 60) -> pd.DataFrame:
    timestamps = [start + timedelta(hours=i) for i in range(periods)]
    close = np.linspace(100, 120, periods) + np.random.default_rng(0).normal(0, 1, periods)
    high = close + np.random.default_rng(1).normal(1, 0.5, periods)
    low = close - np.random.default_rng(2).normal(1, 0.5, periods)
    volume = np.random.default_rng(3).integers(100, 200, periods)
    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(timestamps),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_news(start: datetime, periods: int = 60) -> pd.DataFrame:
    timestamps = [start + timedelta(hours=i) for i in range(periods)]
    titles = [f"Headline {i}" for i in range(periods)]
    scores = np.sin(np.linspace(0, 4 * np.pi, periods))
    return pd.DataFrame(
        {
            "published_utc": pd.to_datetime(timestamps),
            "title": titles,
            "url": [f"http://example.com/{i}" for i in range(periods)],
            "source": "UnitTest",
            "lang": "en",
            "sentiment_score": scores,
        }
    )


def test_build_xy_shapes_and_outputs(tmp_path: Path, monkeypatch: Any) -> None:
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
        tau=0.01,
        start="2024-01-01 05:00:00",
        end="2024-01-03 00:00:00",
    )

    assert not X.empty
    assert len(X) == len(y)
    assert X.isna().sum().max() == 0
    assert all(column.startswith("feat_") for column in X.columns)
    assert set(y.unique()).issubset({"up", "down", "flat"})

    dataset_path = output_dir / "features" / "1h_2h" / "dataset.parquet"
    assert dataset_path.exists()
    dataset = pd.read_parquet(dataset_path)
    assert {"timestamp_utc", "label", "r_forward"}.issubset(dataset.columns)
    feature_columns = [column for column in dataset.columns if column.startswith("feat_")]
    assert feature_columns
    assert all(column.startswith("feat_") for column in feature_columns)

    meta_path = output_dir / "features" / "1h_2h" / "meta.json"
    assert meta_path.exists()
    saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert saved_meta["freq"] == "1h"
    assert saved_meta["horizon"] == "2h"
    assert saved_meta["tau"] == 0.01
    assert saved_meta["seed"] is None
    assert saved_meta["version"] == "unknown"
