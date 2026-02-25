"""Triple-barrier labeling helpers."""

from __future__ import annotations

import pandas as pd

from bitbat.labeling.returns import parse_horizon


def _validate_barrier(value: float, *, name: str) -> float:
    threshold = float(value)
    if threshold <= 0:
        raise ValueError(f"{name} must be positive.")
    return threshold


def triple_barrier_from_close(
    close: pd.Series,
    *,
    horizon: str,
    take_profit: float,
    stop_loss: float,
    return_name: str = "r_forward",
    label_name: str = "label",
) -> pd.DataFrame:
    """Label each timestamp by first barrier hit or timeout within the horizon."""
    horizon_delta = parse_horizon(horizon)
    tp = _validate_barrier(take_profit, name="take_profit")
    sl = _validate_barrier(stop_loss, name="stop_loss")

    index_utc = pd.to_datetime(close.index, utc=True, errors="raise")
    if not index_utc.is_monotonic_increasing:
        raise ValueError("Price index must be sorted ascending.")
    if index_utc.has_duplicates:
        raise ValueError("Price index must be unique.")

    close_values = pd.to_numeric(close, errors="raise").astype("float64").to_numpy()
    labels = pd.Series(pd.NA, index=close.index, dtype="string", name=label_name)
    returns = pd.Series(float("nan"), index=close.index, dtype="float64", name=return_name)

    for start_idx, start_ts in enumerate(index_utc):
        end_ts = start_ts + horizon_delta
        end_idx = int(index_utc.searchsorted(end_ts, side="right"))
        if end_idx <= start_idx + 1:
            continue

        base_price = close_values[start_idx]
        event_label = "timeout"
        event_return = (close_values[end_idx - 1] - base_price) / base_price

        for path_idx in range(start_idx + 1, end_idx):
            path_return = (close_values[path_idx] - base_price) / base_price
            if path_return >= tp:
                event_label = "take_profit"
                event_return = path_return
                break
            if path_return <= -sl:
                event_label = "stop_loss"
                event_return = path_return
                break

        labels.iat[start_idx] = event_label
        returns.iat[start_idx] = float(event_return)

    return pd.DataFrame(
        {
            return_name: returns,
            label_name: labels,
        },
        index=close.index,
    )


def triple_barrier(
    prices_df: pd.DataFrame,
    *,
    horizon: str,
    take_profit: float,
    stop_loss: float,
    return_name: str = "r_forward",
    label_name: str = "label",
) -> pd.DataFrame:
    """Compute triple-barrier labels from a price frame with a `close` column."""
    if "close" not in prices_df.columns:
        raise KeyError("`prices_df` must contain a 'close' column.")

    return triple_barrier_from_close(
        prices_df["close"],
        horizon=horizon,
        take_profit=take_profit,
        stop_loss=stop_loss,
        return_name=return_name,
        label_name=label_name,
    )
