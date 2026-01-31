"""Purging utilities to prevent label leakage."""

from __future__ import annotations

import pandas as pd


def mask_future(news_ts: pd.Series, bar_end_ts: pd.Series) -> pd.Series:
    """Return a mask that excludes rows with timestamps after their bar end.

    Leakage guarantee: only rows with news timestamps at or before the bar end
    are retained, preventing future information from leaking into the bar.
    """
    if not isinstance(news_ts, pd.Series) or not isinstance(bar_end_ts, pd.Series):
        raise TypeError("Both `news_ts` and `bar_end_ts` must be pandas Series.")

    if len(news_ts) != len(bar_end_ts):
        raise ValueError("`news_ts` and `bar_end_ts` must have matching length.")

    news_times = pd.to_datetime(news_ts, utc=True, errors="raise")
    bar_times = pd.to_datetime(bar_end_ts, utc=True, errors="raise")

    mask = news_times <= bar_times
    return mask.reset_index(drop=True)
