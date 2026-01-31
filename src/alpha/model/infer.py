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
    model: xgb.Booster,
    features_row: pd.Series,
    timestamp: Any | None = None,
    *,
    class_order: list[str] | None = None,
) -> dict[str, Any]:
    """Predict probability of up/down move for a single bar."""
    booster = _ensure_model(model)

    required = REQUIRED_FEATURES or booster.feature_names
    if required is None:
        required = list(features_row.index)

    missing = set(required) - set(features_row.index)
    if missing:
        raise KeyError(f"Missing required features: {sorted(missing)}")

    feature_names = list(features_row.index)
    dmatrix = xgb.DMatrix(features_row.to_frame().T, feature_names=feature_names)
    proba = booster.predict(dmatrix)[0]

    classes = class_order or ["down", "flat", "up"]
    if len(proba) != len(classes):
        raise ValueError("Mismatch between predicted probabilities and class labels.")

    mapping = dict(zip(classes, proba, strict=False))
    return {
        "timestamp": timestamp,
        "p_up": float(mapping.get("up", 0.0)),
        "p_down": float(mapping.get("down", 0.0)),
    }


def run_inference(model: Any, features: Any) -> Any:  # pragma: no cover - stub
    """Run inference using the provided model and features."""
    raise NotImplementedError("run_inference is not implemented yet.")
