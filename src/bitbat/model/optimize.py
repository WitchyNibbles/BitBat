"""Nested walk-forward hyperparameter optimization with deterministic provenance."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from bitbat.dataset.splits import Fold
from bitbat.model.evaluate import (
    classification_probability_metrics,
    compute_multiple_testing_safeguards,
)
from bitbat.model.train import DIRECTION_CLASSES

logger = logging.getLogger(__name__)


class HyperparamOptimizer:
    """Optuna-based nested walk-forward search for XGBoost.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Continuous forward returns aligned to *X* (float64).
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
        self.y = y.copy()
        self.folds = folds
        self.seed = seed

    @staticmethod
    def _is_classification_target(y: pd.Series) -> bool:
        from bitbat.model.utils import is_classification_target
        return is_classification_target(y)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert numpy/pandas scalar types to JSON-safe Python primitives."""
        if isinstance(value, (np.floating, np.integer)):  # noqa: UP038
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): HyperparamOptimizer._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [HyperparamOptimizer._json_safe(item) for item in value]
        return value

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest one hyperparameter candidate."""
        return {
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

    def _fold_score(self, params: dict[str, Any], fold: Fold, *, seed: int) -> float | None:
        """Evaluate one fold and return a minimized score, or None if the fold is empty."""
        train_mask = self.X.index.isin(fold.train)
        test_mask = self.X.index.isin(fold.test)

        if not train_mask.any() or not test_mask.any():
            return None

        X_tr = self.X.loc[train_mask]
        y_tr = self.y[train_mask]
        X_te = self.X.loc[test_mask]
        y_te = self.y[test_mask]
        classification_mode = self._is_classification_target(y_tr)

        from bitbat.model.utils import create_dmatrices
        dtrain, dtest, _y_tr, y_te_norm = create_dmatrices(
            X_tr, y_tr, X_te, y_te, list(self.X.columns), classification_mode
        )

        xgb_params = dict(params)
        num_rounds = int(xgb_params.pop("num_boost_round", 100))
        if classification_mode:
            full_params = {
                "objective": "multi:softprob",
                "num_class": len(DIRECTION_CLASSES),
                "eval_metric": "mlogloss",
                "seed": seed,
                "nthread": 1,
                **xgb_params,
            }
        else:
            full_params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "seed": seed,
                "nthread": 1,
                **xgb_params,
            }
        booster = xgb.train(
            full_params,
            dtrain,
            num_boost_round=num_rounds,
            evals=[(dtest, "val")],
            verbose_eval=False,
        )

        preds = booster.predict(dtest)
        if classification_mode:
            metrics = classification_probability_metrics(y_te_norm.to_numpy(dtype=object), preds)
            return float(1.0 - metrics["pr_auc"])
        return float(np.sqrt(np.mean((y_te_norm.to_numpy() - preds) ** 2)))

    def _cv_score(
        self,
        params: dict[str, Any],
        *,
        folds: list[Fold] | None = None,
        seed: int | None = None,
    ) -> float:
        """Run walk-forward CV with *params* and return mean minimized score."""
        selected_folds = folds if folds is not None else self.folds
        resolved_seed = self.seed if seed is None else seed
        fold_scores: list[float] = []
        for fold in selected_folds:
            score = self._fold_score(params, fold, seed=resolved_seed)
            if score is not None:
                fold_scores.append(score)
        return float(np.mean(fold_scores)) if fold_scores else 999.0

    @staticmethod
    def _fold_window(fold: Fold) -> dict[str, Any]:
        """Serialize one fold's window boundaries/sizes."""
        train_start = str(fold.train[0]) if len(fold.train) else None
        train_end = str(fold.train[-1]) if len(fold.train) else None
        test_start = str(fold.test[0]) if len(fold.test) else None
        test_end = str(fold.test[-1]) if len(fold.test) else None
        return {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_size": int(len(fold.train)),
            "test_size": int(len(fold.test)),
            "purge_bars": int(getattr(fold, "purge_bars", 0)),
            "embargo_bars": int(getattr(fold, "embargo_bars", 0)),
        }

    def _trial_history(self, study: optuna.Study) -> list[dict[str, Any]]:
        """Return deterministic JSON-safe trial history for one study."""
        payload: list[dict[str, Any]] = []
        for trial in sorted(study.trials, key=lambda item: item.number):
            payload.append({
                "number": int(trial.number),
                "state": str(trial.state.name).lower(),
                "value": round(float(trial.value), 6) if trial.value is not None else None,
                "params": self._json_safe(trial.params),
            })
        return payload

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
        outer_folds: list[dict[str, Any]] = []
        all_trial_history: list[dict[str, Any]] = []
        best_trial_lineage: list[dict[str, Any]] = []
        best_params: dict[str, Any] = {}
        best_study: optuna.Study | None = None
        best_outer_score = float("inf")
        outer_scores: list[float] = []
        classification_mode = self._is_classification_target(self.y)

        for outer_idx, outer_fold in enumerate(self.folds):
            inner_folds = self.folds[:outer_idx]
            if not inner_folds:
                # First outer fold has no prior folds; fall back to self-contained tuning.
                inner_folds = [outer_fold]

            outer_seed = self.seed + outer_idx
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=outer_seed),
            )
            study.optimize(
                lambda trial, fold_set=inner_folds, seed=outer_seed: self._cv_score(
                    self._suggest_params(trial),
                    folds=fold_set,
                    seed=seed,
                ),
                n_trials=n_trials,
                timeout=timeout,
            )
            selected_params = dict(study.best_trial.params)
            outer_score = self._cv_score(
                selected_params,
                folds=[outer_fold],
                seed=outer_seed,
            )
            outer_scores.append(outer_score)

            outer_folds.append({
                "outer_fold": outer_idx + 1,
                "inner_fold_count": int(len(inner_folds)),
                "window": self._fold_window(outer_fold),
                "selected_params": self._json_safe(selected_params),
                "inner_best_score": round(float(study.best_trial.value), 6),
                "outer_score": round(float(outer_score), 6),
                "outer_pr_auc": round(float(1.0 - outer_score), 6)
                if classification_mode
                else None,
            })
            all_trial_history.append({
                "outer_fold": outer_idx + 1,
                "seed": int(outer_seed),
                "trials": self._trial_history(study),
            })
            best_trial_lineage.append({
                "outer_fold": outer_idx + 1,
                "trial_number": int(study.best_trial.number),
                "inner_score": round(float(study.best_trial.value), 6),
                "params": self._json_safe(selected_params),
            })

            if outer_score < best_outer_score:
                best_outer_score = outer_score
                best_params = selected_params
                best_study = study

        if best_study is None:
            best_study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.seed),
            )

        aggregate_score = float(np.mean(outer_scores)) if outer_scores else 999.0
        if classification_mode:
            logger.info(
                "Nested optimization finished: aggregate outer PR-AUC=%.6f across %d folds",
                1.0 - aggregate_score,
                len(outer_folds),
            )
        else:
            logger.info(
                "Nested optimization finished: aggregate outer RMSE=%.6f across %d folds",
                aggregate_score,
                len(outer_folds),
            )

        search_space = {
            "eta": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 2, "high": 10},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
            "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0, "log": False},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
            "gamma": {"type": "float", "low": 0.0, "high": 5.0, "log": False},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "num_boost_round": {"type": "int", "low": 20, "high": 200},
        }
        provenance = {
            "mode": "nested_walk_forward",
            "seed": int(self.seed),
            "n_trials_requested": int(n_trials),
            "timeout_seconds": int(timeout) if timeout is not None else None,
            "search_space": search_space,
            "folds": [self._fold_window(fold) for fold in self.folds],
            "outer_folds": outer_folds,
            "trial_history": all_trial_history,
            "best_trial_lineage": best_trial_lineage,
            "aggregate_outer_score": round(aggregate_score, 6),
            "aggregate_outer_pr_auc": round(float(1.0 - aggregate_score), 6)
            if classification_mode
            else None,
            "objective_mode": "classification" if classification_mode else "regression",
            "score_metric": "1-pr_auc" if classification_mode else "rmse",
            # Keep deterministic outputs stable by not persisting runtime clock values.
            "wall_clock": {
                "clock_captured": False,
                "started_at_utc": None,
                "completed_at_utc": None,
            },  # noqa: E501
        }
        trial_count = int(sum(len(item.get("trials", [])) for item in all_trial_history))
        safeguards = compute_multiple_testing_safeguards(
            outer_folds,
            trial_count=trial_count if trial_count > 0 else int(n_trials),
        )
        provenance["safeguards"] = safeguards

        return OptimizationResult(
            best_params=self._json_safe(best_params),
            best_score=aggregate_score,
            n_trials=int(n_trials),
            study=best_study,
            mode="nested_walk_forward",
            outer_folds=outer_folds,
            provenance=provenance,
            safeguards=safeguards,
            objective_mode="classification" if classification_mode else "regression",
        )


class OptimizationResult:
    """Container for hyperparameter optimization output."""

    def __init__(
        self,
        best_params: dict[str, Any],
        best_score: float,
        n_trials: int,
        study: optuna.Study,
        *,
        mode: str = "nested_walk_forward",
        outer_folds: list[dict[str, Any]] | None = None,
        provenance: dict[str, Any] | None = None,
        safeguards: dict[str, Any] | None = None,
        objective_mode: str = "regression",
    ) -> None:
        self.best_params = best_params
        self.best_score = best_score
        self.n_trials = n_trials
        self.study = study
        self.mode = mode
        self.outer_folds = list(outer_folds or [])
        self.provenance = dict(provenance or {})
        self.safeguards = dict(safeguards or {})
        self.objective_mode = objective_mode

    def to_xgb_params(self) -> tuple[dict[str, Any], int]:
        """Return XGBoost-compatible params dict and num_boost_round separately.

        The ``num_boost_round`` key is extracted from ``best_params`` (it is
        not a valid ``xgb.train`` parameter).
        """
        params = dict(self.best_params)
        num_rounds = int(params.pop("num_boost_round", 100))
        return params, num_rounds

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "best_params": self.best_params,
            "best_score": round(self.best_score, 6),
            "n_trials": self.n_trials,
            "mode": self.mode,
            "objective_mode": self.objective_mode,
            "best_pr_auc": round(float(1.0 - self.best_score), 6)
            if self.objective_mode == "classification"
            else None,
            "outer_folds": self.outer_folds,
            "provenance": self.provenance,
            "safeguards": self.safeguards,
        }
