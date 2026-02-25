"""Target labeling strategies."""

from __future__ import annotations

import pandas as pd

from bitbat.labeling.returns import forward_return


def _validate_tau(tau: float) -> float:
    tau_value = float(tau)
    if tau_value < 0:
        raise ValueError("Threshold `tau` must be non-negative.")
    return tau_value


def classify(
    r: pd.Series,
    tau: float,
    *,
    name: str = "target",
) -> pd.Series:
    """Primary labeling method: classify returns into up/down/flat labels.

    Labels are assigned as:
    - "up" when r > tau
    - "down" when r < -tau
    - "flat" when |r| <= tau
    """
    tau_value = _validate_tau(tau)

    labels = pd.Series(pd.NA, index=r.index, dtype="string")
    labels[r > tau_value] = "up"
    labels[r < -tau_value] = "down"
    labels[(r.abs() <= tau_value)] = "flat"
    labels.name = name
    return labels


def direction_from_returns(
    returns: pd.Series,
    *,
    tau: float = 0.0,
    name: str = "label",
) -> pd.Series:
    """Derive direction labels from canonical forward returns."""
    return classify(returns, tau=tau, name=name)


def direction_from_prices(
    prices_df: pd.DataFrame,
    *,
    horizon: str,
    tau: float = 0.0,
    return_name: str = "r_forward",
    label_name: str = "label",
) -> pd.DataFrame:
    """Generate forward returns and direction labels from one horizon path."""
    returns = forward_return(prices_df, horizon).rename(return_name)
    direction = direction_from_returns(returns, tau=tau, name=label_name)
    return pd.DataFrame(
        {
            return_name: returns,
            label_name: direction,
        },
        index=returns.index,
    )
