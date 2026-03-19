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
from bitbat.model.evaluate import (
    build_candidate_report,
    classification_probability_metrics,
    window_diagnostics,
)
from bitbat.model.train import DIRECTION_CLASSES

INT_TO_DIRECTION = {value: key for key, value in DIRECTION_CLASSES.items()}
CLASS_TO_SIGNAL = {"up": 1.0, "down": -1.0, "flat": 0.0}


@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""

    fold_index: int
    train_size: int
    test_size: int
    rmse: float
    mae: float
    directional_accuracy: float
    predictions: pd.DataFrame  # columns: predicted, actual
    net_sharpe: float = 0.0
    gross_sharpe: float = 0.0
    total_costs: float = 0.0
    total_fee_costs: float = 0.0
    total_slippage_costs: float = 0.0
    net_return: float = 0.0
    gross_return: float = 0.0
    window_metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    pr_auc: float | None = None
    logloss: float | None = None
    objective_mode: str = "regression"


@dataclass
class WalkForwardResult:
    """Aggregate results from a walk-forward backtest."""

    fold_results: list[FoldResult] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.fold_results)

    @property
    def mean_rmse(self) -> float:
        if not self.fold_results:
            return 0.0
        return float(np.mean([f.rmse for f in self.fold_results]))

    @property
    def mean_mae(self) -> float:
        if not self.fold_results:
            return 0.0
        return float(np.mean([f.mae for f in self.fold_results]))

    @property
    def mean_directional_accuracy(self) -> float:
        if not self.fold_results:
            return 0.0
        return float(np.mean([f.directional_accuracy for f in self.fold_results]))

    @property
    def all_predictions(self) -> pd.DataFrame:
        """Concatenate predictions from all folds."""
        if not self.fold_results:
            return pd.DataFrame()
        return pd.concat([f.predictions for f in self.fold_results], ignore_index=True)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        result: dict[str, Any] = {
            "n_folds": self.n_folds,
            "mean_rmse": round(self.mean_rmse, 6),
            "mean_mae": round(self.mean_mae, 6),
            "mean_directional_accuracy": round(self.mean_directional_accuracy, 4),
            "fold_rmses": [round(f.rmse, 6) for f in self.fold_results],
            "fold_sizes": [f.test_size for f in self.fold_results],
            "total_test_samples": sum(f.test_size for f in self.fold_results),
            "fold_windows": [f.window_metadata for f in self.fold_results],
            "fold_diagnostics": [f.diagnostics for f in self.fold_results],
        }
        if any(f.pr_auc is not None for f in self.fold_results):
            pr_aucs = [float(f.pr_auc) for f in self.fold_results if f.pr_auc is not None]
            loglosses = [float(f.logloss) for f in self.fold_results if f.logloss is not None]
            result["mean_pr_auc"] = round(float(np.mean(pr_aucs)), 6) if pr_aucs else 0.0
            result["fold_pr_aucs"] = [round(float(value), 6) for value in pr_aucs]
            result["mean_logloss"] = round(float(np.mean(loglosses)), 6) if loglosses else 0.0
            result["objective_mode"] = "classification"
        if any(f.net_sharpe != 0.0 or f.total_costs != 0.0 for f in self.fold_results):
            result["mean_net_sharpe"] = round(
                float(np.mean([f.net_sharpe for f in self.fold_results])), 4
            )
            result["mean_gross_sharpe"] = round(
                float(np.mean([f.gross_sharpe for f in self.fold_results])), 4
            )
            result["total_costs"] = round(sum(f.total_costs for f in self.fold_results), 6)
            result["total_fee_costs"] = round(sum(f.total_fee_costs for f in self.fold_results), 6)
            result["total_slippage_costs"] = round(
                sum(f.total_slippage_costs for f in self.fold_results), 6
            )
            result["mean_net_return"] = round(
                float(np.mean([f.net_return for f in self.fold_results])), 6
            )
            result["mean_gross_return"] = round(
                float(np.mean([f.gross_return for f in self.fold_results])), 6
            )

        if self.fold_results:
            family = str(self.fold_results[0].diagnostics.get("family", "xgb"))
            fold_metrics: list[dict[str, Any]] = []
            for fold in self.fold_results:
                fold_metrics.append({
                    "rmse": float(fold.rmse),
                    "mae": float(fold.mae),
                    "directional_accuracy": float(fold.directional_accuracy),
                    "correlation": float(fold.diagnostics.get("volatility_ratio", 0.0)),
                    "net_sharpe": float(fold.net_sharpe),
                    "gross_sharpe": float(fold.gross_sharpe),
                    "max_drawdown": float(fold.diagnostics.get("drift_score", 0.0) * -1.0),
                    "net_return": float(fold.net_return),
                    "gross_return": float(fold.gross_return),
                    "total_costs": float(fold.total_costs),
                    "total_fee_costs": float(fold.total_fee_costs),
                    "total_slippage_costs": float(fold.total_slippage_costs),
                })
            result["candidate_report"] = build_candidate_report(
                candidate_id=family,
                family=family,
                fold_metrics=fold_metrics,
            )["metrics"]
        return result


class WalkForwardValidator:
    """Walk-forward backtest: retrain XGBoost regression on each expanding fold.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (DatetimeIndex).
    y : pd.Series
        Continuous forward returns aligned to *X* (float64).
    folds : list[Fold]
        Walk-forward folds from ``dataset.splits.walk_forward``.
    xgb_params : dict
        XGBoost training parameters (without ``objective``).
    num_boost_round : int
        Boosting rounds per fold.
    prices : pd.Series | None
        Close prices aligned to *X* for cost-adjusted backtesting.
    cost_bps : float
        Round-trip transaction cost in basis points.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        folds: list[Fold],
        *,
        xgb_params: dict[str, Any] | None = None,
        num_boost_round: int = 100,
        prices: pd.Series | None = None,
        cost_bps: float = 4.0,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
    ) -> None:
        self.X = X.astype(float)
        self.y = y.copy()
        self.folds = folds
        self.xgb_params = xgb_params or {}
        self.num_boost_round = num_boost_round
        self.prices = prices
        self.cost_bps = cost_bps
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

    @staticmethod
    def _is_classification_target(y: pd.Series) -> bool:
        from bitbat.model.utils import is_classification_target
        return is_classification_target(y)

    def _cost_metrics(
        self, test_index: pd.Index, predicted_returns: np.ndarray
    ) -> tuple[float, float, float, float, float, float, float]:
        """Run fold backtest and return net/gross and cost-attribution metrics."""
        if self.prices is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        from bitbat.backtest.engine import run as bt_run

        fold_prices = self.prices.loc[self.prices.index.isin(test_index)]
        if len(fold_prices) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        pred_returns = pd.Series(predicted_returns, index=test_index)
        common = fold_prices.index.intersection(pred_returns.index)
        if len(common) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        trades, equity = bt_run(
            fold_prices.loc[common],
            pred_returns.loc[common],
            cost_bps=self.cost_bps,
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
        )

        net_pnl = trades["pnl"]
        gross_pnl = trades["gross_pnl"]
        std_net = net_pnl.std()
        std_gross = gross_pnl.std()
        net_sharpe = float(np.sqrt(252) * net_pnl.mean() / std_net) if std_net > 0 else 0.0
        gross_sharpe = float(np.sqrt(252) * gross_pnl.mean() / std_gross) if std_gross > 0 else 0.0
        total_costs = float(trades["costs"].sum())
        total_fee_costs = (
            float(trades["fee_costs"].sum()) if "fee_costs" in trades.columns else total_costs
        )
        total_slippage_costs = (
            float(trades["slippage_costs"].sum()) if "slippage_costs" in trades.columns else 0.0
        )
        net_return = float(equity.iloc[-1] - 1) if len(equity) > 0 else 0.0
        gross_return = (
            float((1.0 + gross_pnl).cumprod().iloc[-1] - 1.0) if len(gross_pnl) > 0 else 0.0
        )

        return (
            net_sharpe,
            gross_sharpe,
            total_costs,
            total_fee_costs,
            total_slippage_costs,
            net_return,
            gross_return,
        )

    def run(self) -> WalkForwardResult:
        """Execute the walk-forward backtest, retraining at each fold."""
        result = WalkForwardResult()

        for i, fold in enumerate(self.folds):
            train_mask = self.X.index.isin(fold.train)
            test_mask = self.X.index.isin(fold.test)

            if not train_mask.any() or not test_mask.any():
                continue

            X_tr = self.X.loc[train_mask]
            y_tr = self.y[train_mask]
            X_te = self.X.loc[test_mask]
            y_te = self.y[test_mask]
            classification_mode = self._is_classification_target(y_tr)

            from bitbat.model.utils import create_dmatrices
            dtrain, dtest, _y_tr, y_te_norm = create_dmatrices(
                X_tr, y_tr, X_te, y_te, list(self.X.columns), classification_mode
            )

            if classification_mode:
                params = {
                    "objective": "multi:softprob",
                    "num_class": len(DIRECTION_CLASSES),
                    "eval_metric": "mlogloss",
                    **self.xgb_params,
                }
            else:
                params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    **self.xgb_params,
                }

            booster = xgb.train(
                params, dtrain, num_boost_round=self.num_boost_round, verbose_eval=False
            )
            pr_auc: float | None = None
            logloss: float | None = None
            if classification_mode:
                probabilities = np.asarray(booster.predict(dtest), dtype="float64")
                predicted_idx = np.argmax(probabilities, axis=1)
                predicted_directions = np.asarray(
                    [INT_TO_DIRECTION[int(idx)] for idx in predicted_idx],
                    dtype=object,
                )
                actual_directions = y_te_norm.to_numpy(dtype=object)
                predicted = (
                    probabilities[:, DIRECTION_CLASSES["up"]]
                    - probabilities[:, DIRECTION_CLASSES["down"]]
                )
                actual = np.asarray(
                    [CLASS_TO_SIGNAL[str(label)] for label in actual_directions],
                    dtype="float64",
                )
                residuals = actual - predicted
                rmse = float(np.sqrt(np.mean(residuals**2)))
                mae = float(np.mean(np.abs(residuals)))
                cls_metrics = classification_probability_metrics(actual_directions, probabilities)
                directional_accuracy = float(cls_metrics["directional_accuracy"])
                pr_auc = float(cls_metrics["pr_auc"])
                logloss = float(cls_metrics["mlogloss"])
                preds_df = pd.DataFrame({
                    "predicted": predicted,
                    "actual": actual,
                    "p_up": probabilities[:, DIRECTION_CLASSES["up"]],
                    "p_down": probabilities[:, DIRECTION_CLASSES["down"]],
                    "p_flat": probabilities[:, DIRECTION_CLASSES["flat"]],
                    "predicted_direction": predicted_directions,
                    "actual_direction": actual_directions,
                })
            else:
                predicted = np.asarray(booster.predict(dtest), dtype="float64")
                actual = y_te_norm.to_numpy()
                residuals = actual - predicted
                rmse = float(np.sqrt(np.mean(residuals**2)))
                mae = float(np.mean(np.abs(residuals)))
                sign_match = np.sign(actual) == np.sign(predicted)
                directional_accuracy = float(np.mean(sign_match))
                preds_df = pd.DataFrame({
                    "predicted": predicted,
                    "actual": actual,
                })

            # Cost-adjusted metrics
            (
                net_sharpe,
                gross_sharpe,
                total_costs,
                total_fee_costs,
                total_slippage_costs,
                net_return,
                gross_return,
            ) = self._cost_metrics(X_te.index, predicted)
            window_metadata = {
                "train_start": X_tr.index.min().isoformat(),
                "train_end": X_tr.index.max().isoformat(),
                "test_start": X_te.index.min().isoformat(),
                "test_end": X_te.index.max().isoformat(),
                "embargo_bars": int(getattr(fold, "embargo_bars", 0)),
                "purge_bars": int(getattr(fold, "purge_bars", 0)),
            }
            diagnostics = window_diagnostics(
                actual,
                predicted,
                window_id=f"fold-{i + 1}",
                family=str(self.xgb_params.get("family", "xgb")),
            )

            result.fold_results.append(
                FoldResult(
                    fold_index=i,
                    train_size=int(train_mask.sum()),
                    test_size=int(test_mask.sum()),
                    rmse=rmse,
                    mae=mae,
                    directional_accuracy=directional_accuracy,
                    predictions=preds_df,
                    net_sharpe=net_sharpe,
                    gross_sharpe=gross_sharpe,
                    total_costs=total_costs,
                    total_fee_costs=total_fee_costs,
                    total_slippage_costs=total_slippage_costs,
                    net_return=net_return,
                    gross_return=gross_return,
                    window_metadata=window_metadata,
                    diagnostics=diagnostics,
                    pr_auc=pr_auc,
                    logloss=logloss,
                    objective_mode="classification" if classification_mode else "regression",
                )
            )

        return result
