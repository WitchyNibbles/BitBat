"""Return calculation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd


def forward_return(prices_df: pd.DataFrame, horizon: str) -> pd.Series:
    """Compute forward simple returns at horizon H using close prices."""
    if "close" not in prices_df.columns:
        raise KeyError("`prices_df` must contain a 'close' column.")

    try:
        delta = pd.to_timedelta(horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid horizon '{horizon}'.") from exc

    if delta <= pd.Timedelta(0):
        raise ValueError("Horizon must be positive.")

    index = pd.to_datetime(prices_df.index, utc=True, errors="raise")
    close = pd.Series(
        prices_df["close"].to_numpy(dtype="float64"),
        index=index,
        name="close",
    )

    future = pd.Series(close.to_numpy(), index=index - delta)
    aligned = future.reindex(index)

    returns = (aligned - close) / close
    returns = returns.astype("float64")
    returns = returns.where(~aligned.isna())
    returns.name = f"fwd_return_{horizon}"

    # Reindex back to original index type if necessary.
    returns.index = prices_df.index
    return returns


class PricePoint(Protocol):
    """Minimal protocol describing price access."""

    close: float


def compute_log_returns(points: Sequence[PricePoint]) -> list[float]:
    """Compute log returns for a sequence of price points."""
    raise NotImplementedError("compute_log_returns is not implemented yet.")
