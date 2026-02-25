"""Return calculation helpers used for labeling."""

from __future__ import annotations

import pandas as pd


def parse_horizon(horizon: str) -> pd.Timedelta:
    """Parse and validate a positive labeling horizon."""
    try:
        delta = pd.to_timedelta(horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid horizon '{horizon}'.") from exc

    if delta <= pd.Timedelta(0):
        raise ValueError("Horizon must be positive.")
    return delta


def forward_return_from_close(close: pd.Series, horizon: str) -> pd.Series:
    """Compute forward simple returns for a close-price series."""
    delta = parse_horizon(horizon)
    index = pd.to_datetime(close.index, utc=True, errors="raise")
    if not index.is_monotonic_increasing:
        raise ValueError("Price index must be sorted ascending.")
    if index.has_duplicates:
        raise ValueError("Price index must be unique.")

    close_numeric = pd.to_numeric(close, errors="raise").astype("float64")
    aligned_close = pd.Series(close_numeric.to_numpy(), index=index, name="close")
    future = pd.Series(aligned_close.to_numpy(), index=index - delta)
    aligned_future = future.reindex(index)

    returns = (aligned_future - aligned_close) / aligned_close
    returns = returns.astype("float64")
    returns = returns.where(~aligned_future.isna())
    returns.name = f"fwd_return_{horizon}"
    returns.index = close.index
    return returns


def forward_return(prices_df: pd.DataFrame, horizon: str) -> pd.Series:
    """Primary labeling method: forward simple returns over a horizon.

    The returned series aligns each timestamp with the close price at
    `timestamp + horizon` and computes (future - current) / current.
    """
    if "close" not in prices_df.columns:
        raise KeyError("`prices_df` must contain a 'close' column.")
    return forward_return_from_close(prices_df["close"], horizon)
