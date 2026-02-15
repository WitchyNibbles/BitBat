"""
Walk-forward backtesting with model retraining at each fold.

Simulates how the model would have performed if retrained periodically
on expanding windows of data — the most realistic backtest possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from bitbat.dataset.splits import Fold


@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""

    fold_index: int
    train_size: int
    test_size: int
    accuracy: float
    logloss: float
    predictions: pd.DataFrame  # columns: predicted, actual, p_up, p_down, p_flat


@dataclass
class WalkForwardResult:
    """Aggregate results from a walk-forward backtest."""

    fold_results: list[FoldResult] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.fold_results)

    @property
    def mean_accuracy(self) -> float:
        if not self.fold_results:
            return 0.0
        return float(np.mean([f.accuracy for f in self.fold_results]))

    @property
    def mean_logloss(self) -> float:
        if not self.fold_results:
            return 0.0
        return float(np.mean([f.logloss for f in self.fold_results]))

    @property
    def all_predictions(self) -> pd.DataFrame:
        """Concatenate predictions from all folds."""
        if not self.fold_results:
            return pd.DataFrame()
        return pd.concat([f.predictions for f in self.fold_results], ignore_index=True)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "n_folds": self.n_folds,
            "mean_accuracy": round(self.mean_accuracy, 4),
            "mean_logloss": round(self.mean_logloss, 4),
            "fold_accuracies": [round(f.accuracy, 4) for f in self.fold_results],
            "fold_sizes": [f.test_size for f in self.fold_results],
            "total_test_samples": sum(f.test_size for f in self.fold_results),
        }


class WalkForwardValidator:
    """Walk-forward backtest: retrain XGBoost on each expanding fold.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (DatetimeIndex).
    y : pd.Series
        Labels aligned to *X* (string values).
    folds : list[Fold]
        Walk-forward folds from ``dataset.splits.walk_forward``.
    xgb_params : dict
        XGBoost training parameters (without ``objective``/``num_class``).
    num_boost_round : int
        Boosting rounds per fold.
    """

    CLASS_ORDER = ["down", "flat", "up"]

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        folds: list[Fold],
        *,
        xgb_params: dict[str, Any] | None = None,
        num_boost_round: int = 50,
    ) -> None:
        self.X = X.astype(float)
        self.y = pd.Series(y).astype("category")
        self.labels = self.y.cat.codes.to_numpy()
        self.categories = list(self.y.cat.categories)
        self.num_class = len(self.categories)
        self.folds = folds
        self.xgb_params = xgb_params or {}
        self.num_boost_round = num_boost_round

    def run(self) -> WalkForwardResult:
        """Execute the walk-forward backtest, retraining at each fold."""
        result = WalkForwardResult()

        for i, fold in enumerate(self.folds):
            train_mask = self.X.index.isin(fold.train)
            test_mask = self.X.index.isin(fold.test)

            if not train_mask.any() or not test_mask.any():
                continue

            X_tr = self.X.loc[train_mask]
            y_tr = self.labels[train_mask]
            X_te = self.X.loc[test_mask]
            y_te = self.labels[test_mask]

            dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(self.X.columns))
            dtest = xgb.DMatrix(X_te, label=y_te, feature_names=list(self.X.columns))

            params = {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": self.num_class,
                **self.xgb_params,
            }

            booster = xgb.train(params, dtrain, num_boost_round=self.num_boost_round, verbose_eval=False)
            probas = booster.predict(dtest)

            # Accuracy
            pred_classes = probas.argmax(axis=1)
            accuracy = float((pred_classes == y_te).mean())

            # Logloss
            eps = 1e-15
            clipped = np.clip(probas, eps, 1 - eps)
            logloss = -float(np.mean(np.log(clipped[np.arange(len(y_te)), y_te])))

            # Build predictions DataFrame
            pred_labels = [self.categories[c] for c in pred_classes]
            actual_labels = [self.categories[c] for c in y_te]

            preds_df = pd.DataFrame({
                "predicted": pred_labels,
                "actual": actual_labels,
            })
            for ci, cat in enumerate(self.categories):
                preds_df[f"p_{cat}"] = probas[:, ci]

            result.fold_results.append(
                FoldResult(
                    fold_index=i,
                    train_size=int(train_mask.sum()),
                    test_size=int(test_mask.sum()),
                    accuracy=accuracy,
                    logloss=logloss,
                    predictions=preds_df,
                )
            )

        return result
