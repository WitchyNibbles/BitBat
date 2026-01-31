"""Sentiment feature generation."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@lru_cache(maxsize=1)
def _get_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def score_vader(texts: pd.Series) -> pd.Series:
    """Score sentiment for texts using VADER compound metric."""
    analyzer = _get_analyzer()

    def _score(text: Any) -> float | None:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return np.nan
        stripped = str(text).strip()
        if not stripped:
            return np.nan
        return float(analyzer.polarity_scores(stripped)["compound"])

    scores = texts.apply(_score)
    scores.name = "sentiment_score"
    return scores


def aggregate(
    news_df: pd.DataFrame,
    bar_df: pd.DataFrame,
    freq: str,
    windows: Iterable[str] = ("1h", "4h", "24h"),
) -> pd.DataFrame:
    """Aggregate sentiment features per bar, respecting look-ahead purging."""
    required_news_cols = {"published_utc", "sentiment_score"}
    missing_news = required_news_cols - set(news_df.columns)
    if missing_news:
        raise KeyError(f"Missing columns in news_df: {missing_news}")

    required_bar_cols = {"timestamp_utc"}
    missing_bar = required_bar_cols - set(bar_df.columns)
    if missing_bar:
        raise KeyError(f"Missing columns in bar_df: {missing_bar}")

    news = news_df.copy()
    news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True)
    bars = bar_df.copy()
    bars["timestamp_utc"] = pd.to_datetime(bars["timestamp_utc"], utc=True)

    features = []
    news = news.sort_values("published_utc")
    for _, bar in bars.iterrows():
        end = bar["timestamp_utc"]
        window_features = {"timestamp_utc": end}
        for window in windows:
            window_delta = pd.to_timedelta(window)
            start = end - window_delta
            mask = (news["published_utc"] > start) & (news["published_utc"] <= end)
            window_slice = news.loc[mask]
            if not window_slice.empty and window_slice["published_utc"].max() > end:
                raise ValueError("Sentiment aggregation included future news.")
            prefix = f"sent_{window}".replace("h", "h_")
            if window_slice.empty:
                window_features[f"{prefix}mean"] = np.nan
                window_features[f"{prefix}median"] = np.nan
                window_features[f"{prefix}pos"] = 0
                window_features[f"{prefix}neg"] = 0
                window_features[f"{prefix}neu"] = 0
                window_features[f"{prefix}count"] = 0
                window_features[f"{prefix}decay"] = 0.0
                continue

            window_features[f"{prefix}mean"] = window_slice["sentiment_score"].mean()
            window_features[f"{prefix}median"] = window_slice["sentiment_score"].median()
            window_features[f"{prefix}pos"] = (window_slice["sentiment_score"] > 0.05).sum()
            window_features[f"{prefix}neg"] = (window_slice["sentiment_score"] < -0.05).sum()
            window_features[f"{prefix}neu"] = (
                (window_slice["sentiment_score"] >= -0.05)
                & (window_slice["sentiment_score"] <= 0.05)
            ).sum()
            window_features[f"{prefix}count"] = len(window_slice)

            weights = np.exp(
                -((end - window_slice["published_utc"]).dt.total_seconds())
                / window_delta.total_seconds()
            )
            window_features[f"{prefix}decay"] = float(
                (
                    ((window_slice["sentiment_score"] > 0.05).astype(float))
                    - ((window_slice["sentiment_score"] < -0.05).astype(float))
                ).to_numpy()
                @ weights
            )

        features.append(window_features)

    feature_df = pd.DataFrame(features)
    feature_df = feature_df.sort_values("timestamp_utc").set_index("timestamp_utc")
    feature_df.index = feature_df.index.tz_localize(None)

    return feature_df


def build_sentiment_features(data: Any) -> Any:
    """Generate sentiment-based features."""
    raise NotImplementedError("build_sentiment_features is not implemented yet.")
