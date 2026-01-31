"""Target labeling strategies."""

from __future__ import annotations

from typing import Any

import pandas as pd


def classify(r: pd.Series, tau: float) -> pd.Series:
    """Classify returns into up/down/flat labels based on threshold tau."""
    if tau < 0:
        raise ValueError("Threshold `tau` must be non-negative.")

    labels = pd.Series(index=r.index, dtype="object")
    labels[r > tau] = "up"
    labels[r < -tau] = "down"
    labels[(r.abs() <= tau)] = "flat"
    labels.name = "target"
    return labels


def triple_barrier_method(*, horizons: int, upper: float, lower: float) -> Any:
    """Compute classification targets using the triple-barrier method."""
    raise NotImplementedError("triple_barrier_method is not implemented yet.")
