"""On-chain feature generation from blockchain metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_onchain_features(
    onchain_df: pd.DataFrame,
    freq: str = "1h",
) -> pd.DataFrame:
    """Generate on-chain features from daily blockchain metrics.

    Input DataFrame should have a ``date`` column (or DatetimeIndex) with
    daily observations for columns like ``hashrate``, ``tx_count``,
    ``mempool_size``, ``avg_block_size``.

    All output columns are prefixed with ``onchain_`` (the ``feat_`` prefix
    is added later by the dataset builder).

    Parameters
    ----------
    onchain_df:
        Daily on-chain DataFrame.
    freq:
        Target bar frequency for resampling (e.g. ``"1h"``).
    """
    df = onchain_df.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Resample to target frequency with forward-fill
    df = df.resample(freq).ffill()

    features = pd.DataFrame(index=df.index)

    # Hash rate features
    if "hashrate" in df.columns:
        hr = df["hashrate"].astype("float64")
        features["onchain_hashrate_change"] = hr.pct_change()
        window_14d = 336 if freq == "1h" else 14  # 14 days
        roll_mean = hr.rolling(window=window_14d, min_periods=window_14d).mean()
        roll_std = hr.rolling(window=window_14d, min_periods=window_14d).std()
        features["onchain_hashrate_z"] = (hr - roll_mean) / roll_std.replace(0, np.nan)

    # Transaction count features
    if "tx_count" in df.columns:
        tx = df["tx_count"].astype("float64")
        window_14d = 336 if freq == "1h" else 14
        roll_mean = tx.rolling(window=window_14d, min_periods=window_14d).mean()
        roll_std = tx.rolling(window=window_14d, min_periods=window_14d).std()
        features["onchain_tx_count_z"] = (tx - roll_mean) / roll_std.replace(0, np.nan)

    # Mempool size features (shorter window — more volatile)
    if "mempool_size" in df.columns:
        mp = df["mempool_size"].astype("float64")
        window_7d = 168 if freq == "1h" else 7  # 7 days
        roll_mean = mp.rolling(window=window_7d, min_periods=window_7d).mean()
        roll_std = mp.rolling(window=window_7d, min_periods=window_7d).std()
        features["onchain_mempool_z"] = (mp - roll_mean) / roll_std.replace(0, np.nan)

    # Block size features
    if "avg_block_size" in df.columns:
        bs = df["avg_block_size"].astype("float64")
        features["onchain_block_size_change"] = bs.pct_change()

    return features
