"""Model inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

REQUIRED_FEATURES: list[str] | None = None


def _ensure_model(model: xgb.Booster | str | Path) -> xgb.Booster:
    if isinstance(model, (str | Path)):
        booster = xgb.Booster()
        booster.load_model(str(model))
        return booster
    return model


def predict_bar(
    model: xgb.Booster | str | Path,
    features_row: pd.Series,
    timestamp: Any | None = None,
    *,
    current_price: float | None = None,
) -> dict[str, Any]:
    """Predict return magnitude for a single bar.

    Args:
        model: Trained XGBoost booster or a path to a saved booster.
        features_row: Feature values for a single time step.
        timestamp: Optional timestamp to attach to the output payload.
        current_price: Current close price used to compute predicted_price.

    Returns:
        Dictionary with timestamp, predicted_return, predicted_price,
        and predicted_direction.
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
    predicted_return = float(booster.predict(dmatrix)[0])

    predicted_price = None
    if current_price is not None:
        predicted_price = current_price * (1 + predicted_return)

    predicted_direction = "up" if predicted_return > 0 else "down"

    return {
        "timestamp": timestamp,
        "predicted_return": predicted_return,
        "predicted_price": predicted_price,
        "predicted_direction": predicted_direction,
    }
