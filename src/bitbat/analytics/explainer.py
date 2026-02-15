"""
Prediction explainability for BitBat using XGBoost's native SHAP support.

XGBoost natively computes SHAP (Shapley Additive Explanations) values via
``booster.predict(dmatrix, pred_contribs=True)`` — no extra package required.

These values tell you: "For this prediction, feature X pushed the probability
up/down by this much."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


class PredictionExplainer:
    """Explain individual BitBat predictions using XGBoost's built-in SHAP.

    Parameters
    ----------
    model_path:
        Path to the saved ``xgb.json`` model file.
    """

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)
        self._booster: xgb.Booster | None = None

    def _load_model(self) -> xgb.Booster:
        if self._booster is None:
            booster = xgb.Booster()
            booster.load_model(str(self.model_path))
            self._booster = booster
        return self._booster

    # ------------------------------------------------------------------
    # SHAP contribution matrix
    # ------------------------------------------------------------------

    def shap_values(
        self, X: pd.DataFrame
    ) -> np.ndarray:
        """Return raw SHAP contribution array for *X*.

        Parameters
        ----------
        X:
            Feature matrix with the same columns the model was trained on.

        Returns
        -------
        ndarray of shape ``(n_samples, n_features + 1, n_classes)`` where the
        last feature column is the bias term.  We drop the bias for user-facing
        display.
        """
        booster = self._load_model()
        dmatrix = xgb.DMatrix(X.astype(float), feature_names=list(X.columns))
        contribs = booster.predict(dmatrix, pred_contribs=True)
        # contribs shape: (n_samples, n_features+1) for binary or
        # (n_samples, n_classes, n_features+1) for multi-class
        return contribs

    def explain_row(
        self,
        row: pd.Series | pd.DataFrame,
        label_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Explain a single prediction row in plain language.

        Parameters
        ----------
        row:
            A single-row DataFrame or Series of feature values.
        label_map:
            Mapping from class index to class name, e.g. ``{0:'down', 1:'flat', 2:'up'}``.

        Returns
        -------
        Dict with keys: ``contributions``, ``top_positive``, ``top_negative``,
        ``plain_english``.
        """
        if isinstance(row, pd.Series):
            X = row.to_frame().T
        else:
            X = row.head(1)

        contribs = self.shap_values(X)

        # For multi-class XGBoost the shape is (1, n_classes, n_features+1)
        # For binary it's (1, n_features+1)
        n_feats = X.shape[1]
        feat_names = list(X.columns)

        if contribs.ndim == 3:
            # Multi-class: pick the class with highest output contribution sum
            per_class = contribs[0, :, :n_feats]  # (n_classes, n_features)
            predicted_class = int(np.argmax(per_class.sum(axis=1)))
            feat_contribs = per_class[predicted_class]
        else:
            # Binary
            feat_contribs = contribs[0, :n_feats]
            predicted_class = 0

        # Build contribution series
        contrib_series = pd.Series(feat_contribs, index=feat_names).sort_values(
            key=abs, ascending=False
        )

        top_positive = contrib_series[contrib_series > 0].head(5)
        top_negative = contrib_series[contrib_series < 0].head(5)

        direction = label_map.get(predicted_class, str(predicted_class)) if label_map else str(predicted_class)

        plain = _build_plain_explanation(top_positive, top_negative, direction)

        return {
            "contributions": contrib_series,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "predicted_class": predicted_class,
            "predicted_direction": direction,
            "plain_english": plain,
        }

    def feature_importance_from_model(self) -> pd.Series:
        """Return gain-based feature importance from the model (no data needed)."""
        booster = self._load_model()
        scores = booster.get_score(importance_type="gain")
        if not scores:
            return pd.Series(dtype=float)
        return (
            pd.Series(scores)
            .sort_values(ascending=False)
        )

    def batch_mean_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute mean absolute SHAP contribution per feature over *X*.

        Returns
        -------
        DataFrame with columns: feature, mean_abs_shap, rank.
        """
        contribs = self.shap_values(X)
        n_feats = X.shape[1]
        feat_names = list(X.columns)

        if contribs.ndim == 3:
            # Average over classes
            feat_contribs = contribs[:, :, :n_feats].mean(axis=1)  # (n_samples, n_features)
        else:
            feat_contribs = contribs[:, :n_feats]

        mean_abs = np.abs(feat_contribs).mean(axis=0)
        df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df


# ---------------------------------------------------------------------------
# Plain-language explanation builder
# ---------------------------------------------------------------------------


def _readable_feature_name(feat: str) -> str:
    """Convert a technical feature name to a readable label."""
    translations = {
        "feat_ret_1": "1-hour price return",
        "feat_ret_4": "4-hour price return",
        "feat_ret_24": "24-hour price return",
        "feat_vol_24": "24-hour volatility",
        "feat_vol_4": "4-hour volatility",
        "feat_rsi_14": "RSI (momentum indicator)",
        "feat_macd": "MACD trend signal",
        "feat_sent_1h_mean": "1-hour news sentiment",
        "feat_sent_4h_mean": "4-hour news sentiment",
        "feat_sent_24h_mean": "24-hour news sentiment",
        "feat_sent_1h_count": "1-hour news volume",
    }
    return translations.get(feat, feat.replace("feat_", "").replace("_", " ").title())


def _build_plain_explanation(
    top_positive: pd.Series,
    top_negative: pd.Series,
    direction: str,
) -> str:
    """Build a plain-English explanation string."""
    dir_word = {"up": "UP ↑", "down": "DOWN ↓", "flat": "FLAT →"}.get(direction, direction.upper())
    lines = [f"BitBat predicted **{dir_word}** because:"]

    if not top_positive.empty:
        lines.append("\n**Supporting evidence:**")
        for feat, val in top_positive.items():
            name = _readable_feature_name(str(feat))
            lines.append(f"- {name} (weight: +{val:.3f})")

    if not top_negative.empty:
        lines.append("\n**Counter-evidence:**")
        for feat, val in top_negative.items():
            name = _readable_feature_name(str(feat))
            lines.append(f"- {name} (weight: {val:.3f})")

    return "\n".join(lines)
