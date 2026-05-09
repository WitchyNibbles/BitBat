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


def _default_class_labels(label_mode: str) -> list[str]:
    resolved_label_mode = normalize_label_mode(label_mode)
    if resolved_label_mode == "direction":
        return ["up", "down", "flat"]
    if resolved_label_mode == "triple_barrier":
        return ["take_profit", "stop_loss", "timeout"]
    return ["act", "pass"]


def _resolve_xgb_metadata(
    model: xgb.Booster | str | Path,
    booster: xgb.Booster,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(metadata or {})
    if isinstance(model, str | Path):
        for key, value in load_metadata(model).items():
            payload.setdefault(key, value)

    attributes = booster.attributes()
    if payload.get("label_mode") in (None, ""):
        payload["label_mode"] = attributes.get("label_mode", "direction")
    raw_class_labels = attributes.get("class_labels_json")
    if raw_class_labels not in (None, "") and not payload.get("class_labels"):
        decoded = json.loads(str(raw_class_labels))
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise ValueError("Artifact class_labels metadata must be a JSON string list.")
        payload["class_labels"] = [str(item).strip().lower() for item in decoded]

    payload["family"] = str(payload.get("family", "xgb")).strip().lower()
    payload["label_mode"] = normalize_label_mode(str(payload.get("label_mode", "direction")))
    class_labels = payload.get("class_labels")
    if isinstance(class_labels, list | tuple):
        payload["class_labels"] = [str(label).strip().lower() for label in class_labels]
    else:
        payload["class_labels"] = _default_class_labels(str(payload["label_mode"]))
    return payload


def _regression_direction(
    predicted_return: float,
    *,
    tau: float,
) -> tuple[str, float, float, float]:
    resolved_tau = tau if tau > 0 else 0.01
    magnitude = min(abs(float(predicted_return)) / resolved_tau, 1.0)
    p_flat = round(max(0.0, 1.0 - magnitude), 6)
    directional_mass = round(1.0 - p_flat, 6)
    if predicted_return >= resolved_tau:
        return "up", directional_mass, 0.0, p_flat
    if predicted_return <= -resolved_tau:
        return "down", 0.0, directional_mass, p_flat
    if predicted_return >= 0.0:
        return "flat", round(directional_mass / 2.0, 6), 0.0, p_flat
    return "flat", 0.0, round(directional_mass / 2.0, 6), p_flat


def _required_feature_names(model: Any, features_row: pd.Series) -> list[str]:
    feature_names = getattr(model, "feature_names", None)
    if feature_names:
        return list(feature_names)
    feature_names_in = getattr(model, "feature_names_in_", None)
    if feature_names_in is not None:
        return [str(name) for name in feature_names_in]
    return list(features_row.index)


def _predict_xgb_classification(
    model: xgb.Booster | str | Path,
    booster: xgb.Booster,
    features_row: pd.Series,
    *,
    metadata: dict[str, Any],
    timestamp: Any | None,
    current_price: float | None,
    tau: float,
) -> dict[str, Any]:
    required = _required_feature_names(booster, features_row)
    missing = set(required) - set(features_row.index)
    if missing:
        raise KeyError(f"Missing required features: {sorted(missing)}")

    feature_frame = features_row.loc[required].to_frame().T
    dmatrix = xgb.DMatrix(feature_frame, feature_names=required)
    probs = np.asarray(booster.predict(dmatrix), dtype="float64")
    if probs.ndim != 2 or probs.shape[0] != 1:
        raise ValueError(f"Expected XGBoost classification probabilities, got shape {probs.shape}")

    resolved_metadata = _resolve_xgb_metadata(model, booster, metadata)
    resolved_label_mode = normalize_label_mode(str(resolved_metadata["label_mode"]))
    class_labels = [str(label) for label in resolved_metadata["class_labels"]]
    if probs.shape[1] != len(class_labels):
        raise ValueError("Classification artifact produced an unexpected probability shape.")
    class_idx = int(np.argmax(probs[0]))
    predicted_label = class_labels[class_idx]
    probabilities = {
        label: float(prob)
        for label, prob in zip(class_labels, probs[0], strict=True)
    }
    if resolved_label_mode == "triple_barrier":
        predicted_direction = {
            "take_profit": "up",
            "stop_loss": "down",
            "timeout": "flat",
        }.get(predicted_label, "flat")
        p_up = float(probabilities.get("take_profit", 0.0))
        p_down = float(probabilities.get("stop_loss", 0.0))
        p_flat = float(probabilities.get("timeout", 0.0))
        confidence = prediction_confidence(
            predicted_direction,
            p_up=p_up,
            p_down=p_down,
            p_flat=p_flat,
        )
    elif resolved_label_mode == "direction":
        predicted_direction = predicted_label
        p_up = float(probabilities.get("up", 0.0))
        p_down = float(probabilities.get("down", 0.0))
        p_flat = float(probabilities.get("flat", 0.0))
        confidence = prediction_confidence(
            predicted_direction,
            p_up=p_up,
            p_down=p_down,
            p_flat=p_flat,
        )
    else:
        predicted_direction = predicted_label
        p_up = 0.0
        p_down = 0.0
        p_flat = 0.0
        confidence = round(float(probabilities.get(predicted_label, 0.0)), 6)

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
        "predicted_label": predicted_label,
        "predicted_return": None,
        "predicted_price": predicted_price,
        "predicted_direction": predicted_direction,
        "probabilities": probabilities,
        "p_up": p_up,
        "p_down": p_down,
        "p_flat": p_flat,
        "confidence": confidence,
    }


def predict_with_metadata(
    model: Any,
    features_row: pd.Series,
    *,
    metadata: dict[str, Any],
    timestamp: Any | None = None,
    current_price: float | None = None,
    tau: float = 0.01,
) -> dict[str, Any]:
    """Predict using artifact metadata to normalize classifier and regressor outputs."""
    family = str(metadata.get("family", "xgb")).strip().lower()
    if family == "xgb":
        booster = _ensure_model(model)
        return _predict_xgb_classification(
            model,
            booster,
            features_row,
            metadata=metadata,
            timestamp=timestamp,
            current_price=current_price,
            tau=tau,
        )

    required = _required_feature_names(model, features_row)
    missing = set(required) - set(features_row.index)
    if missing:
        raise KeyError(f"Missing required features: {sorted(missing)}")

    feature_frame = features_row.loc[required].to_frame().T.astype(float)
    predicted_return = float(model.predict(feature_frame)[0])
    predicted_direction, p_up, p_down, p_flat = _regression_direction(predicted_return, tau=tau)
    confidence = prediction_confidence(
        predicted_direction,
        p_up=p_up,
        p_down=p_down,
        p_flat=p_flat,
    )
    predicted_price = None
    if current_price is not None:
        predicted_price = float(current_price) * (1.0 + predicted_return)
    return {
        "timestamp": timestamp,
        "predicted_label": None,
        "predicted_return": predicted_return,
        "predicted_price": predicted_price,
        "predicted_direction": predicted_direction,
        "probabilities": {
            "up": p_up,
            "down": p_down,
            "flat": p_flat,
        },
        "p_up": p_up,
        "p_down": p_down,
        "p_flat": p_flat,
        "confidence": confidence,
    }


def predict_classification(
    model: xgb.Booster | str | Path,
    features_row: pd.Series,
    timestamp: Any | None = None,
    *,
    class_labels: list[str] | tuple[str, ...] | None = None,
    label_mode: str = "meta_label",
) -> dict[str, Any]:
    """Compatibility helper for classification-only inference paths."""
    metadata: dict[str, Any] = {
        "family": "xgb",
        "label_mode": label_mode,
    }
    if class_labels is not None:
        metadata["class_labels"] = [str(label).strip().lower() for label in class_labels]
    result = predict_with_metadata(
        model,
        features_row,
        metadata=metadata,
        timestamp=timestamp,
    )
    return {
        "timestamp": result.get("timestamp"),
        "label_mode": normalize_label_mode(label_mode),
        "predicted_label": result.get("predicted_label"),
        "confidence": round(
            float(
                result.get("probabilities", {}).get(
                    str(result.get("predicted_label", "")).lower(),
                    result.get("confidence", 0.0) or 0.0,
                )
            ),
            6,
        ),
        "probabilities": dict(result.get("probabilities", {})),
    }


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
    metadata = _resolve_xgb_metadata(model, booster)
    if metadata["label_mode"] != "direction":
        raise ValueError(
            f"predict_bar only supports direction artifacts; got label_mode='{metadata['label_mode']}'."
        )
    if metadata["class_labels"] != ["up", "down", "flat"]:
        raise ValueError(
            "Direction artifacts must declare class_labels ['up', 'down', 'flat'] to infer safely."
        )
    return predict_with_metadata(
        booster,
        features_row,
        metadata=metadata,
        timestamp=timestamp,
        current_price=current_price,
        tau=tau,
    )
