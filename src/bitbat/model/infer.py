"""Model inference helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

REQUIRED_FEATURES: list[str] | None = None
DIRECTION_CLASSES: dict[str, int] = {"up": 0, "down": 1, "flat": 2}
INT_TO_DIRECTION: dict[int, str] = {0: "up", 1: "down", 2: "flat"}


def _ensure_model(model: xgb.Booster | str | Path) -> xgb.Booster:
    if isinstance(model, (str | Path)):
        booster = xgb.Booster()
        booster.load_model(str(model))
        return booster
    return model


def directional_confidence(
    predicted_return: float,
    tau: float = 0.01,
    scale: float = 2.0,
) -> tuple[float, float]:
    """Derive directional probabilities (p_up, p_down) from a predicted return.

    Uses a sigmoid mapping centred at zero so that larger absolute predicted
    returns yield higher confidence in the predicted direction.  The *tau*
    parameter controls the return magnitude at which confidence reaches ~73%
    (i.e. ``1 / (1 + exp(-1))``).  *scale* stretches the sigmoid -- higher
    values make the transition sharper.

    Returns:
        ``(p_up, p_down)`` where both are in ``[0, 1]`` and sum to 1.
    """
    if tau <= 0:
        tau = 0.01
    normalised = predicted_return / tau
    p_up = 1.0 / (1.0 + math.exp(-scale * normalised))
    p_down = 1.0 - p_up
    return round(p_up, 6), round(p_down, 6)


def prediction_confidence(
    predicted_direction: str,
    *,
    p_up: float | None = None,
    p_down: float | None = None,
    p_flat: float | None = None,
) -> float | None:
    """Return the probability assigned to the winning predicted direction."""
    direction = str(predicted_direction).lower().strip()
    mapping = {
        "up": p_up,
        "down": p_down,
        "flat": p_flat,
    }
    direct = mapping.get(direction)
    if direct is not None:
        return round(float(direct), 6)

    candidates = [value for value in (p_up, p_down, p_flat) if value is not None]
    if not candidates:
        return None
    return round(float(max(candidates)), 6)


def predict_bar(
    model: xgb.Booster | str | Path,
    features_row: pd.Series,
    timestamp: Any | None = None,
    *,
    current_price: float | None = None,
    tau: float = 0.01,
) -> dict[str, Any]:
    """Predict return magnitude for a single bar.

    Args:
        model: Trained XGBoost booster or a path to a saved booster.
        features_row: Feature values for a single time step.
        timestamp: Optional timestamp to attach to the output payload.
        current_price: Current close price used to compute predicted_price.
        tau: Label threshold used to scale directional confidence.

    Returns:
        Dictionary with timestamp, predicted_return, predicted_price,
        predicted_direction, p_up, and p_down.
    """
    booster = _ensure_model(model)

    required = REQUIRED_FEATURES or booster.feature_names
    if required is None:
        required = list(features_row.index)

    missing = set(required) - set(features_row.index)
    if missing:
        raise KeyError(f"Missing required features: {sorted(missing)}")

    feature_names = list(features_row.index)
    dmatrix = xgb.DMatrix(features_row.to_frame().T, feature_names=feature_names)

    # multi:softprob returns shape (1, 3); argmax over class axis gives direction index
    probs = booster.predict(dmatrix)  # shape (1, 3)
    class_idx = int(np.argmax(probs[0]))
    predicted_direction = INT_TO_DIRECTION[class_idx]
    p_up = float(probs[0][DIRECTION_CLASSES["up"]])
    p_down = float(probs[0][DIRECTION_CLASSES["down"]])
    p_flat = float(probs[0][DIRECTION_CLASSES["flat"]])
    confidence = prediction_confidence(
        predicted_direction,
        p_up=p_up,
        p_down=p_down,
        p_flat=p_flat,
    )

    return {
        "timestamp": timestamp,
        "predicted_return": None,  # No longer a regression model
        "predicted_price": None,  # Cannot compute without predicted return
        "predicted_direction": predicted_direction,
        "p_up": p_up,
        "p_down": p_down,
        "p_flat": p_flat,
        "confidence": confidence,
    }
