"""Model training routines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb


def fit_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int = 42,
) -> tuple[xgb.Booster, dict[str, float]]:
    """Train an XGBoost regression model and persist it to disk.

    Persists the trained booster to `models/{freq}_{horizon}/xgb.json` when
    `X_train` includes the `freq` and `horizon` attributes on `.attrs`.

    Args:
        X_train: Feature matrix for training, using numeric columns.
        y_train: Continuous forward returns (float64) aligned to `X_train`.
        seed: Random seed for model training.

    Returns:
        The trained booster and a gain-based importance mapping keyed by feature name.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    X = X_train.astype(float)
    y = y_train.astype("float64").to_numpy()

    dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": seed,
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    booster = xgb.train(params, dtrain, num_boost_round=100)

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
