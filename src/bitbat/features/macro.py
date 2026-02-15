"""Macroeconomic feature generation from FRED data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_macro_features(
    macro_df: pd.DataFrame,
    freq: str = "1h",
) -> pd.DataFrame:
    """Generate macro features from daily FRED data, resampled to bar frequency.

    The input DataFrame should have a ``date`` column (or DatetimeIndex) with
    daily observations for columns like ``fed_funds_rate``, ``treasury_10y``,
    ``usd_index``, ``vix``, ``inflation_5y_breakeven``.

    All output columns are prefixed with ``macro_`` (the ``feat_`` prefix is
    added later by the dataset builder).

    Parameters
    ----------
    macro_df:
        Daily macro DataFrame.  Must have a datetime index or ``date`` column.
    freq:
        Target bar frequency for resampling (e.g. ``"1h"``).
    """
    df = macro_df.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Resample to target frequency with forward-fill (no look-ahead)
    df = df.resample(freq).ffill()

    features = pd.DataFrame(index=df.index)

    # Per-series features: daily change and 30-day rolling z-score
    series_cols = [c for c in df.columns if c not in {"date"}]
    for col in series_cols:
        series = df[col].astype("float64")
        # Daily percentage change (forward-filled, so change appears once/day)
        features[f"macro_{col}_change"] = series.pct_change()
        # Rolling z-score (30 days ≈ 720 hourly bars)
        roll_window = 720 if freq == "1h" else 30
        roll_mean = series.rolling(window=roll_window, min_periods=roll_window).mean()
        roll_std = series.rolling(window=roll_window, min_periods=roll_window).std()
        features[f"macro_{col}_z"] = (series - roll_mean) / roll_std.replace(0, np.nan)

    # Derived features
    if "treasury_10y" in df.columns and "fed_funds_rate" in df.columns:
        features["macro_yield_spread"] = df["treasury_10y"].astype("float64") - df[
            "fed_funds_rate"
        ].astype("float64")
    if "vix" in df.columns:
        features["macro_vix_level"] = df["vix"].astype("float64")

    return features
