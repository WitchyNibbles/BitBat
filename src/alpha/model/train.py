"""Model training routines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb


def fit_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    class_weights: bool = True,
    seed: int = 42,
) -> tuple[xgb.Booster, dict[str, float]]:
    """Train an XGBoost multi-class classifier."""
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    X = X_train.astype(float)
    y = pd.Series(y_train).astype("category")
    labels = y.cat.codes.to_numpy()
    num_class = len(y.cat.categories)

    weights = None
    if class_weights:
        counts = y.value_counts()
        total = len(y)
        weights_map = {cat: total / (len(counts) * count) for cat, count in counts.items()}
        weights = y.map(weights_map).to_numpy()

    dtrain = xgb.DMatrix(X, label=labels, weight=weights, feature_names=list(X.columns))

    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": num_class,
        "seed": seed,
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    booster = xgb.train(params, dtrain, num_boost_round=50)

    raw_importance = booster.get_score(importance_type="gain")
    importance: dict[str, float] = {
        feature: float(value[0] if isinstance(value, list) else value)
        for feature, value in {col: raw_importance.get(col, 0.0) for col in X.columns}.items()
    }

    freq = "unknown"
    horizon = "unknown"
    if hasattr(X_train, "attrs"):
        freq = X_train.attrs.get("freq", freq)
        horizon = X_train.attrs.get("horizon", horizon)

    output_dir = Path("models") / f"{freq}_{horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgb.json"
    booster.save_model(str(model_path))

    return booster, importance


def train_model(training_data: Any) -> Any:  # pragma: no cover - legacy stub
    """Train a model and return the trained artifact."""
    raise NotImplementedError("train_model is not implemented yet.")
