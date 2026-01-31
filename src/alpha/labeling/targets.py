"""Target labeling strategies."""

from __future__ import annotations

import pandas as pd


def classify(r: pd.Series, tau: float) -> pd.Series:
    """Primary labeling method: classify returns into up/down/flat labels.

    Labels are assigned as:
    - "up" when r > tau
    - "down" when r < -tau
    - "flat" when |r| <= tau
    """
    if tau < 0:
        raise ValueError("Threshold `tau` must be non-negative.")

    labels = pd.Series(index=r.index, dtype="object")
    labels[r > tau] = "up"
    labels[r < -tau] = "down"
    labels[(r.abs() <= tau)] = "flat"
    labels.name = "target"
    return labels
