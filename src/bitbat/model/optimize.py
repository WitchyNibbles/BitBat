"""
Hyperparameter optimization for XGBoost using Optuna.

Integrates with the existing ``walk_forward`` cross-validation
and ``fit_xgb`` training routines to find optimal model parameters.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from bitbat.dataset.splits import Fold

logger = logging.getLogger(__name__)


class HyperparamOptimizer:
    """Optuna-based hyperparameter search for the XGBoost classifier.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Labels aligned to *X* (string values: ``"up"``, ``"down"``, ``"flat"``).
    folds : list[Fold]
        Walk-forward cross-validation folds (from ``dataset.splits.walk_forward``).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        folds: list[Fold],
        *,
        seed: int = 42,
    ) -> None:
        self.X = X.astype(float)
        self.y = pd.Series(y).astype("category")
        self.labels = self.y.cat.codes.to_numpy()
        self.num_class = len(self.y.cat.categories)
        self.folds = folds
        self.seed = seed

    def _cv_score(self, params: dict[str, Any]) -> float:
        """Run walk-forward CV with *params* and return mean log-loss."""
        fold_scores: list[float] = []

        for fold in self.folds:
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

            num_rounds = params.pop("num_boost_round", 50)
            full_params = {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": self.num_class,
                "seed": self.seed,
                **params,
            }
            booster = xgb.train(
                full_params,
                dtrain,
                num_boost_round=int(num_rounds),
                evals=[(dtest, "val")],
                verbose_eval=False,
            )

            # Extract best logloss from eval results
            preds = booster.predict(dtest)
            # Manual mlogloss
            eps = 1e-15
            preds = np.clip(preds, eps, 1 - eps)
            logloss = -np.mean(
                np.log(preds[np.arange(len(y_te)), y_te])
            )
            fold_scores.append(float(logloss))

        return float(np.mean(fold_scores)) if fold_scores else 999.0

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function — minimize CV log-loss."""
        params = {
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_boost_round": trial.suggest_int("num_boost_round", 20, 200),
        }
        return self._cv_score(params)

    def optimize(
        self,
        n_trials: int = 50,
        timeout: int | None = None,
    ) -> OptimizationResult:
        """Run the hyperparameter search.

        Parameters
        ----------
        n_trials : int
            Number of Optuna trials.
        timeout : int | None
            Maximum seconds for the study; ``None`` = no limit.

        Returns
        -------
        OptimizationResult with best parameters and study reference.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(self._objective, n_trials=n_trials, timeout=timeout)

        best = study.best_trial
        logger.info(
            "Optuna finished: best logloss=%.4f after %d trials",
            best.value,
            len(study.trials),
        )

        return OptimizationResult(
            best_params=best.params,
            best_score=best.value,
            n_trials=len(study.trials),
            study=study,
        )


class OptimizationResult:
    """Container for hyperparameter optimization output."""

    def __init__(
        self,
        best_params: dict[str, Any],
        best_score: float,
        n_trials: int,
        study: optuna.Study,
    ) -> None:
        self.best_params = best_params
        self.best_score = best_score
        self.n_trials = n_trials
        self.study = study

    def to_xgb_params(self) -> tuple[dict[str, Any], int]:
        """Return XGBoost-compatible params dict and num_boost_round separately.

        The ``num_boost_round`` key is extracted from ``best_params`` (it is
        not a valid ``xgb.train`` parameter).
        """
        params = dict(self.best_params)
        num_rounds = int(params.pop("num_boost_round", 50))
        return params, num_rounds

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "best_params": self.best_params,
            "best_score": round(self.best_score, 6),
            "n_trials": self.n_trials,
        }
