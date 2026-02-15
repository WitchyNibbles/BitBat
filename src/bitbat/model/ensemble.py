"""
Multi-horizon ensemble — combine predictions from multiple horizons
to produce a more robust directional signal.
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
    p_up: float
    p_down: float
    p_flat: float
    predicted_direction: str
    weight: float = 1.0


@dataclass
class EnsemblePrediction:
    """Weighted-average ensemble prediction from multiple horizons."""

    p_up: float
    p_down: float
    p_flat: float
    predicted_direction: str
    confidence: float
    horizon_predictions: list[HorizonPrediction]

    def summary(self) -> dict[str, Any]:
        return {
            "predicted_direction": self.predicted_direction,
            "confidence": round(self.confidence, 4),
            "p_up": round(self.p_up, 4),
            "p_down": round(self.p_down, 4),
            "p_flat": round(self.p_flat, 4),
            "horizons_used": len(self.horizon_predictions),
            "per_horizon": [
                {
                    "horizon": hp.horizon,
                    "direction": hp.predicted_direction,
                    "p_up": round(hp.p_up, 4),
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
        Horizons to include (e.g. ``["1h", "4h", "24h"]``).
    weights : dict[str, float] | None
        Per-horizon weights.  If ``None``, equal weighting is used.
    """

    CLASS_ORDER = ["down", "flat", "up"]

    def __init__(
        self,
        model_dir: Path | str,
        freq: str = "1h",
        horizons: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.freq = freq
        self.horizons = horizons or ["1h", "4h", "24h"]
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
        EnsemblePrediction with weighted probabilities and per-horizon details.

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
            probas = booster.predict(dmatrix)[0]

            # Map to class order: [p_down, p_flat, p_up]
            p_down, p_flat, p_up = float(probas[0]), float(probas[1]), float(probas[2])
            direction = self.CLASS_ORDER[int(probas.argmax())]
            w = self.weights.get(horizon, 1.0)

            horizon_preds.append(
                HorizonPrediction(
                    horizon=horizon,
                    p_up=p_up,
                    p_down=p_down,
                    p_flat=p_flat,
                    predicted_direction=direction,
                    weight=w,
                )
            )

        if not horizon_preds:
            raise ValueError("No models available for any requested horizon")

        # Weighted average
        total_weight = sum(hp.weight for hp in horizon_preds)
        p_up = sum(hp.p_up * hp.weight for hp in horizon_preds) / total_weight
        p_down = sum(hp.p_down * hp.weight for hp in horizon_preds) / total_weight
        p_flat = sum(hp.p_flat * hp.weight for hp in horizon_preds) / total_weight

        probs = {"up": p_up, "down": p_down, "flat": p_flat}
        direction = max(probs, key=probs.get)  # type: ignore[arg-type]
        confidence = max(probs.values())

        return EnsemblePrediction(
            p_up=p_up,
            p_down=p_down,
            p_flat=p_flat,
            predicted_direction=direction,
            confidence=confidence,
            horizon_predictions=horizon_preds,
        )

    def predict_batch(self, features: pd.DataFrame) -> list[EnsemblePrediction]:
        """Generate ensemble predictions for each row in *features*."""
        return [self.predict(features.iloc[[i]]) for i in range(len(features))]
