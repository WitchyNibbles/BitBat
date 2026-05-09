"""Model inference helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from bitbat.model.persist import load_metadata, normalize_label_mode

REQUIRED_FEATURES: list[str] | None = None
DIRECTION_CLASSES: dict[str, int] = {"up": 0, "down": 1, "flat": 2}
INT_TO_DIRECTION: dict[int, str] = {0: "up", 1: "down", 2: "flat"}


def _ensure_model(model: xgb.Booster | str | Path) -> xgb.Booster:
    if isinstance(model, str | Path):
        booster = xgb.Booster()
        booster.load_model(str(model))
        return booster
    return model


def _artifact_contract(
    model: xgb.Booster | str | Path,
    booster: xgb.Booster,
) -> tuple[str, list[str]]:
    attrs = booster.attributes()
    label_mode = attrs.get("label_mode")
    raw_class_labels = attrs.get("class_labels_json")

    metadata: dict[str, Any] = {}
    if isinstance(model, str | Path):
        metadata = load_metadata(model)
        if label_mode in (None, ""):
            label_mode = metadata.get("label_mode")
        if raw_class_labels in (None, ""):
            class_labels = metadata.get("class_labels")
            if isinstance(class_labels, list):
                raw_class_labels = json.dumps(class_labels)

    resolved_label_mode = normalize_label_mode(str(label_mode or "direction"))
    if raw_class_labels not in (None, ""):
        decoded = json.loads(str(raw_class_labels))
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise ValueError("Artifact class_labels metadata must be a JSON string list.")
        class_labels = [str(item) for item in decoded]
    elif resolved_label_mode == "direction":
        class_labels = ["up", "down", "flat"]
    else:
        class_labels = ["take_profit", "stop_loss", "timeout"]

    return resolved_label_mode, class_labels


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
    label_mode, class_labels = _artifact_contract(model, booster)
    if label_mode != "direction":
        raise ValueError(
            f"predict_bar only supports direction artifacts; got label_mode='{label_mode}'."
        )
    if class_labels != ["up", "down", "flat"]:
        raise ValueError(
            "Direction artifacts must declare class_labels ['up', 'down', 'flat'] to infer safely."
        )

    required = REQUIRED_FEATURES or booster.feature_names
    if required is None:
        required = list(features_row.index)

    missing = set(required) - set(features_row.index)
    if missing:
        raise KeyError(f"Missing required features: {sorted(missing)}")

    classification = predict_classification(model, features_row, timestamp=timestamp)
    predicted_direction = str(classification["predicted_label"])
    probability_by_label = classification["probabilities"]
    p_up = probability_by_label["up"]
    p_down = probability_by_label["down"]
    p_flat = probability_by_label["flat"]
    confidence = prediction_confidence(
        predicted_direction,
        p_up=p_up,
        p_down=p_down,
        p_flat=p_flat,
    )
    predicted_price: float | None = None
    if current_price is not None:
        if predicted_direction == "up":
            predicted_price = float(current_price) * (1.0 + tau)
        elif predicted_direction == "down":
            predicted_price = float(current_price) * (1.0 - tau)
        else:
            predicted_price = float(current_price)

    return {
        "timestamp": timestamp,
        "predicted_return": None,  # No longer a regression model
        "predicted_price": predicted_price,
        "predicted_direction": predicted_direction,
        "p_up": p_up,
        "p_down": p_down,
        "p_flat": p_flat,
        "confidence": confidence,
    }


def predict_classification(
    model: xgb.Booster | str | Path,
    features_row: pd.Series,
    timestamp: Any | None = None,
) -> dict[str, Any]:
    """Predict a generic classification artifact with metadata-backed class labels."""
    booster = _ensure_model(model)
    label_mode, class_labels = _artifact_contract(model, booster)

    required = REQUIRED_FEATURES or booster.feature_names
    if required is None:
        required = list(features_row.index)

    missing = set(required) - set(features_row.index)
    if missing:
        raise KeyError(f"Missing required features: {sorted(missing)}")

    feature_names = list(features_row.index)
    dmatrix = xgb.DMatrix(features_row.to_frame().T, feature_names=feature_names)
    probs = np.asarray(booster.predict(dmatrix), dtype="float64")
    if probs.ndim != 2 or probs.shape[0] != 1 or probs.shape[1] != len(class_labels):
        raise ValueError("Classification artifact produced an unexpected probability shape.")

    predicted_idx = int(np.argmax(probs[0]))
    predicted_label = class_labels[predicted_idx]
    probability_by_label = {
        label: float(probs[0][index]) for index, label in enumerate(class_labels)
    }
    confidence = prediction_confidence(
        predicted_label,
        p_up=probability_by_label.get("up"),
        p_down=probability_by_label.get("down"),
        p_flat=probability_by_label.get("flat"),
    )
    if confidence is None:
        confidence = round(float(max(probability_by_label.values(), default=0.0)), 6)

    return {
        "timestamp": timestamp,
        "label_mode": label_mode,
        "class_labels": class_labels,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probability_by_label,
    }
