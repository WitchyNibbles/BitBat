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
from bitbat.model.evaluate import window_diagnostics


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
    net_return: float = 0.0
    window_metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


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
        if any(f.net_sharpe != 0.0 or f.total_costs != 0.0 for f in self.fold_results):
            result["mean_net_sharpe"] = round(
                float(np.mean([f.net_sharpe for f in self.fold_results])), 4
            )
            result["mean_gross_sharpe"] = round(
                float(np.mean([f.gross_sharpe for f in self.fold_results])), 4
            )
            result["total_costs"] = round(sum(f.total_costs for f in self.fold_results), 6)
            result["mean_net_return"] = round(
                float(np.mean([f.net_return for f in self.fold_results])), 6
            )
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
    ) -> None:
        self.X = X.astype(float)
        self.y = y.astype("float64")
        self.folds = folds
        self.xgb_params = xgb_params or {}
        self.num_boost_round = num_boost_round
        self.prices = prices
        self.cost_bps = cost_bps

    def _cost_metrics(
        self, test_index: pd.Index, predicted_returns: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Run backtest on fold and return (net_sharpe, gross_sharpe, costs, net_return)."""
        if self.prices is None:
            return 0.0, 0.0, 0.0, 0.0

        from bitbat.backtest.engine import run as bt_run

        fold_prices = self.prices.loc[self.prices.index.isin(test_index)]
        if len(fold_prices) < 2:
            return 0.0, 0.0, 0.0, 0.0

        pred_returns = pd.Series(predicted_returns, index=test_index)
        common = fold_prices.index.intersection(pred_returns.index)
        if len(common) < 2:
            return 0.0, 0.0, 0.0, 0.0

        trades, equity = bt_run(
            fold_prices.loc[common],
            pred_returns.loc[common],
            cost_bps=self.cost_bps,
        )

        net_pnl = trades["pnl"]
        gross_pnl = trades["gross_pnl"]
        std_net = net_pnl.std()
        std_gross = gross_pnl.std()
        net_sharpe = float(np.sqrt(252) * net_pnl.mean() / std_net) if std_net > 0 else 0.0
        gross_sharpe = float(np.sqrt(252) * gross_pnl.mean() / std_gross) if std_gross > 0 else 0.0
        total_costs = float(trades["costs"].sum())
        net_return = float(equity.iloc[-1] - 1) if len(equity) > 0 else 0.0

        return net_sharpe, gross_sharpe, total_costs, net_return

    def run(self) -> WalkForwardResult:
        """Execute the walk-forward backtest, retraining at each fold."""
        result = WalkForwardResult()

        for i, fold in enumerate(self.folds):
            train_mask = self.X.index.isin(fold.train)
            test_mask = self.X.index.isin(fold.test)

            if not train_mask.any() or not test_mask.any():
                continue

            X_tr = self.X.loc[train_mask]
            y_tr = self.y[train_mask].to_numpy()
            X_te = self.X.loc[test_mask]
            y_te = self.y[test_mask].to_numpy()

            dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(self.X.columns))
            dtest = xgb.DMatrix(X_te, label=y_te, feature_names=list(self.X.columns))

            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                **self.xgb_params,
            }

            booster = xgb.train(
                params, dtrain, num_boost_round=self.num_boost_round, verbose_eval=False
            )
            predicted = booster.predict(dtest)

            # RMSE
            residuals = y_te - predicted
            rmse = float(np.sqrt(np.mean(residuals**2)))

            # MAE
            mae = float(np.mean(np.abs(residuals)))

            # Directional accuracy
            sign_match = np.sign(y_te) == np.sign(predicted)
            directional_accuracy = float(np.mean(sign_match))

            # Build predictions DataFrame
            preds_df = pd.DataFrame({
                "predicted": predicted,
                "actual": y_te,
            })

            # Cost-adjusted metrics
            net_sharpe, gross_sharpe, total_costs, net_return = self._cost_metrics(
                X_te.index, predicted
            )
            window_metadata = {
                "train_start": X_tr.index.min().isoformat(),
                "train_end": X_tr.index.max().isoformat(),
                "test_start": X_te.index.min().isoformat(),
                "test_end": X_te.index.max().isoformat(),
            }
            diagnostics = window_diagnostics(
                y_te,
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
                    net_return=net_return,
                    window_metadata=window_metadata,
                    diagnostics=diagnostics,
                )
            )

        return result
