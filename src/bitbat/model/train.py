"""Model training routines."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from bitbat.config.loader import resolve_models_dir

BaselineFamily = Literal["xgb", "random_forest"]
TreeBaselineModel = xgb.Booster | RandomForestRegressor

DIRECTION_CLASSES: dict[str, int] = {"up": 0, "down": 1, "flat": 2}


def _extract_freq_horizon(X_train: pd.DataFrame) -> tuple[str, str]:
    freq = "unknown"
    horizon = "unknown"
    if hasattr(X_train, "attrs"):
        freq = str(X_train.attrs.get("freq", freq))
        horizon = str(X_train.attrs.get("horizon", horizon))
    return freq, horizon


def _default_model_path(family: BaselineFamily, freq: str, horizon: str) -> Path:
    filename = "xgb.json" if family == "xgb" else "random_forest.pkl"
    model_dir = resolve_models_dir() / f"{freq}_{horizon}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / filename


def fit_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    family: BaselineFamily = "xgb",
    seed: int = 42,
    persist: bool = True,
) -> tuple[TreeBaselineModel, dict[str, float]]:
    """Train a baseline model family from a shared feature/target contract."""
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    X = X_train.astype(float)

    if family == "xgb":
        y_encoded = y_train.map(DIRECTION_CLASSES).astype(int)
        dtrain = xgb.DMatrix(X, label=y_encoded.to_numpy(), feature_names=list(X.columns))
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
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
        model: TreeBaselineModel = booster
    elif family == "random_forest":
        random_forest = RandomForestRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
        )
        y_float = y_train.astype("float64")
        random_forest.fit(X, y_float.to_numpy())
        importance = {
            feature: float(score)
            for feature, score in zip(
                X.columns,
                random_forest.feature_importances_,
                strict=True,
            )
        }
        model = random_forest
    else:
        raise ValueError(f"Unsupported baseline family: {family}")

    if persist:
        freq, horizon = _extract_freq_horizon(X_train)
        model_path = _default_model_path(family, freq, horizon)
        if family == "xgb":
            if not isinstance(model, xgb.Booster):
                raise TypeError(
                    f"Expected xgb.Booster for family 'xgb', got {type(model).__name__}"
                )  # noqa: E501
            model.save_model(str(model_path))
        else:
            with model_path.open("wb") as artifact:
                pickle.dump(model, artifact)

    return model, importance


def fit_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int = 42,
    persist: bool = True,
) -> tuple[xgb.Booster, dict[str, float]]:
    """Train an XGBoost classification model and persist it to disk."""
    model, importance = fit_baseline(
        X_train,
        y_train,
        family="xgb",
        seed=seed,
        persist=persist,
    )
    if not isinstance(model, xgb.Booster):
        raise TypeError(f"Expected xgb.Booster from fit_baseline, got {type(model).__name__}")
    return model, importance


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int = 42,
    persist: bool = True,
) -> tuple[RandomForestRegressor, dict[str, float]]:
    """Train a RandomForest baseline model using the same contract as XGBoost."""
    model, importance = fit_baseline(
        X_train,
        y_train,
        family="random_forest",
        seed=seed,
        persist=persist,
    )
    if not isinstance(model, RandomForestRegressor):
        raise TypeError(
            f"Expected RandomForestRegressor from fit_baseline, got {type(model).__name__}"
        )  # noqa: E501
    return model, importance
