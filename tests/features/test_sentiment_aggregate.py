from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from alpha.features.sentiment import aggregate


def _make_news(start: datetime) -> pd.DataFrame:
    timestamps = [start + timedelta(minutes=30 * i) for i in range(6)]
    scores = np.linspace(-1, 1, len(timestamps))
    return pd.DataFrame(
        {
            "published_utc": pd.to_datetime(timestamps, utc=True),
            "sentiment_score": scores,
        }
    )


def test_aggregate_excludes_future_news() -> None:
    start = datetime(2024, 1, 1, 0, 0, 0)
    news = _make_news(start)

    # Inject future news beyond bar end
    future_row = pd.DataFrame(
        {
            "published_utc": [pd.Timestamp("2024-01-01T05:30:00Z")],
            "sentiment_score": [0.9],
        }
    )
    news = pd.concat([news, future_row], ignore_index=True)

    bars = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [start + timedelta(hours=i) for i in range(6)], utc=True
            ),
        }
    )

    features = aggregate(news, bars, freq="1h", windows=["1h"])

    assert "sent_1h_mean" in features.columns

    assert pd.isna(features.loc[features.index[0], "sent_1h_mean"]) or np.isclose(
        features.loc[features.index[0], "sent_1h_mean"], news.iloc[0]["sentiment_score"]
    )

    future_ts = pd.Timestamp("2024-01-01 05:00:00")
    assert future_ts in features.index
    assert not np.isclose(
        features.loc[future_ts, "sent_1h_mean"],
        0.9,
    )


def test_aggregate_outputs_expected_columns() -> None:
    start = datetime(2024, 1, 1)
    news = _make_news(start)
    bars = pd.DataFrame(
        {"timestamp_utc": pd.to_datetime([start + timedelta(hours=i) for i in range(4)], utc=True)}
    )

    features = aggregate(news, bars, freq="1h", windows=["1h", "4h"])

    expected_cols = {
        "sent_1h_mean",
        "sent_1h_median",
        "sent_1h_pos",
        "sent_1h_neg",
        "sent_1h_neu",
        "sent_1h_count",
        "sent_1h_decay",
        "sent_4h_mean",
        "sent_4h_median",
        "sent_4h_pos",
        "sent_4h_neg",
        "sent_4h_neu",
        "sent_4h_count",
        "sent_4h_decay",
    }
    assert expected_cols.issubset(features.columns)
