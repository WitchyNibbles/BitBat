"""Purging utilities to prevent label leakage."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import pandas as pd


class LabeledRecord(Protocol):
    """Protocol describing the minimum interface for purging."""

    start: int
    end: int


def purge_overlaps(records: Iterable[LabeledRecord]) -> list[LabeledRecord]:
    """Remove overlapping records to prevent leakage."""
    raise NotImplementedError("purge_overlaps is not implemented yet.")


def mask_future(news_ts: pd.Series, bar_end_ts: pd.Series) -> pd.Series:
    """Return a boolean mask retaining rows with timestamps <= corresponding bar end."""
    if not isinstance(news_ts, pd.Series) or not isinstance(bar_end_ts, pd.Series):
        raise TypeError("Both `news_ts` and `bar_end_ts` must be pandas Series.")

    if len(news_ts) != len(bar_end_ts):
        raise ValueError("`news_ts` and `bar_end_ts` must have matching length.")

    news_times = pd.to_datetime(news_ts, utc=True, errors="raise")
    bar_times = pd.to_datetime(bar_end_ts, utc=True, errors="raise")

    mask = news_times <= bar_times
    return mask.reset_index(drop=True)
