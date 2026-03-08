"""
Multi-horizon ensemble — combine predictions from multiple horizons
to produce a more robust return signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb


@dataclass
class HorizonPrediction:
    """Prediction from a single horizon model."""

    horizon: str
    predicted_return: float
    predicted_direction: str
    weight: float = 1.0


@dataclass
class EnsemblePrediction:
    """Weighted-average ensemble prediction from multiple horizons."""

    predicted_return: float
    predicted_direction: str
    confidence: float
    horizon_predictions: list[HorizonPrediction]

    def summary(self) -> dict[str, Any]:
        return {
            "predicted_direction": self.predicted_direction,
            "predicted_return": round(self.predicted_return, 6),
            "confidence": round(self.confidence, 4),
            "horizons_used": len(self.horizon_predictions),
            "per_horizon": [
                {
                    "horizon": hp.horizon,
                    "direction": hp.predicted_direction,
                    "predicted_return": round(hp.predicted_return, 6),
                    "weight": round(hp.weight, 2),
                }
                for hp in self.horizon_predictions
            ],
        }


class MultiHorizonEnsemble:
    """Combine predictions from models trained on different horizons.

    Parameters
    ----------
    model_dir : Path | str
        Root models directory.  Expects sub-directories like
        ``{freq}_{horizon}/xgb.json``.
    freq : str
        Bar frequency (shared across all horizons).
    horizons : list[str]
        Horizons to include (e.g. ``["15m", "30m", "1h"]``).
    weights : dict[str, float] | None
        Per-horizon weights.  If ``None``, equal weighting is used.
    """

    def __init__(
        self,
        model_dir: Path | str,
        freq: str = "5m",
        horizons: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.freq = freq
        self.horizons = horizons or ["15m", "30m", "1h"]
        self.weights = weights or {h: 1.0 for h in self.horizons}
        self._boosters: dict[str, xgb.Booster] = {}

    def _load_booster(self, horizon: str) -> xgb.Booster | None:
        """Load and cache a booster for *horizon*."""
        if horizon in self._boosters:
            return self._boosters[horizon]
        path = self.model_dir / f"{self.freq}_{horizon}" / "xgb.json"
        if not path.exists():
            return None
        booster = xgb.Booster()
        booster.load_model(str(path))
        self._boosters[horizon] = booster
        return booster

    def available_horizons(self) -> list[str]:
        """Return the subset of requested horizons that have trained models."""
        return [
            h for h in self.horizons if (self.model_dir / f"{self.freq}_{h}" / "xgb.json").exists()
        ]

    def predict(self, features: pd.DataFrame) -> EnsemblePrediction:
        """Generate a weighted-average ensemble prediction for a single bar.

        Parameters
        ----------
        features : pd.DataFrame
            Single-row DataFrame with feature columns.

        Returns
        -------
        EnsemblePrediction with weighted predicted return and per-horizon details.

        Raises
        ------
        ValueError
            If no models are available for any requested horizon.
        """
        horizon_preds: list[HorizonPrediction] = []

        for horizon in self.horizons:
            booster = self._load_booster(horizon)
            if booster is None:
                continue

            dmatrix = xgb.DMatrix(features.astype(float), feature_names=list(features.columns))
            predicted_return = float(booster.predict(dmatrix)[0])
            direction = "up" if predicted_return > 0 else "down"
            w = self.weights.get(horizon, 1.0)

            horizon_preds.append(
                HorizonPrediction(
                    horizon=horizon,
                    predicted_return=predicted_return,
                    predicted_direction=direction,
                    weight=w,
                )
            )

        if not horizon_preds:
            raise ValueError("No models available for any requested horizon")

        # Weighted average of predicted returns
        total_weight = sum(hp.weight for hp in horizon_preds)
        predicted_return = (
            sum(hp.predicted_return * hp.weight for hp in horizon_preds) / total_weight
        )

        direction = "up" if predicted_return > 0 else "down"
        # Confidence: agreement among horizons (fraction pointing same direction)
        agree_count = sum(1 for hp in horizon_preds if hp.predicted_direction == direction)
        confidence = agree_count / len(horizon_preds)

        return EnsemblePrediction(
            predicted_return=predicted_return,
            predicted_direction=direction,
            confidence=confidence,
            horizon_predictions=horizon_preds,
        )

    def predict_batch(self, features: pd.DataFrame) -> list[EnsemblePrediction]:
        """Generate ensemble predictions for each row in *features*."""
        return [self.predict(features.iloc[[i]]) for i in range(len(features))]
