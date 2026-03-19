"""
Phase 5 Complete Integration Test.

End-to-end: optimize → walk-forward
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitbat.dataset.splits import Fold
from bitbat.model.optimize import HyperparamOptimizer
from bitbat.model.walk_forward import WalkForwardValidator

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dataset() -> tuple[pd.DataFrame, pd.Series, list[Fold]]:
    rng = np.random.default_rng(2024)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    X = pd.DataFrame(
        {
            "feat_ret_1": rng.normal(0, 0.01, n),
            "feat_vol_24": rng.uniform(0.01, 0.05, n),
            "feat_rsi_14": rng.uniform(25, 75, n),
        },
        index=idx,
    )
    y = pd.Series(rng.normal(0, 0.01, n), index=idx)
    folds = [
        Fold(train=idx[:150], test=idx[150:225]),
        Fold(train=idx[:225], test=idx[225:300]),
    ]
    return X, y, folds


class TestPhase5Integration:
    def test_optuna_finds_params(self, dataset: tuple) -> None:
        X, y, folds = dataset
        opt = HyperparamOptimizer(X, y, folds, seed=42)
        result = opt.optimize(n_trials=3, timeout=60)
        assert result.best_score > 0
        params, rounds = result.to_xgb_params()
        assert "eta" in params
        assert rounds >= 20

    def test_walk_forward_with_optimized_params(self, dataset: tuple) -> None:
        X, y, folds = dataset
        # Use simple params (avoid full Optuna for speed)
        v = WalkForwardValidator(
            X,
            y,
            folds,
            xgb_params={"max_depth": 3, "eta": 0.1},
            num_boost_round=10,
        )
        result = v.run()
        assert result.n_folds == 2
        assert 0.0 <= result.mean_directional_accuracy <= 1.0
        preds = result.all_predictions
        assert "predicted" in preds.columns

    def test_classification_mode_keeps_summary_compatibility(self) -> None:
        rng = np.random.default_rng(404)
        idx = pd.date_range("2024-02-01", periods=120, freq="1h")
        X = pd.DataFrame(
            {
                "feat_ret_1": rng.normal(0, 0.01, len(idx)),
                "feat_vol_24": rng.uniform(0.01, 0.05, len(idx)),
            },
            index=idx,
        )
        y = pd.Series(
            np.where(
                X["feat_ret_1"] > 0.003,
                "up",
                np.where(X["feat_ret_1"] < -0.003, "down", "flat"),
            ),
            index=idx,
        )
        folds = [
            Fold(train=idx[:80], test=idx[80:100]),
            Fold(train=idx[:100], test=idx[100:120]),
        ]

        optimization_summary = HyperparamOptimizer(X, y, folds, seed=21).optimize(
            n_trials=2,
            timeout=30,
        ).summary()
        walk_forward_summary = WalkForwardValidator(X, y, folds, num_boost_round=5).run().summary()

        assert optimization_summary["objective_mode"] == "classification"
        assert "best_score" in optimization_summary
        assert 0.0 <= float(optimization_summary["best_pr_auc"]) <= 1.0
        assert walk_forward_summary["objective_mode"] == "classification"
        assert "mean_rmse" in walk_forward_summary
        assert 0.0 <= float(walk_forward_summary["mean_pr_auc"]) <= 1.0
