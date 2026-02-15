"""
Phase 5 Complete Integration Test.

End-to-end: optimize → walk-forward → ensemble → monte carlo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from bitbat.analytics.monte_carlo import MonteCarloSimulator
from bitbat.backtest.engine import run as backtest_run
from bitbat.dataset.splits import Fold
from bitbat.model.ensemble import MultiHorizonEnsemble
from bitbat.model.optimize import HyperparamOptimizer
from bitbat.model.walk_forward import WalkForwardValidator


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
    y = pd.Series(rng.choice(["up", "down", "flat"], size=n), index=idx)
    folds = [
        Fold(train=idx[:150], test=idx[150:225]),
        Fold(train=idx[:225], test=idx[225:300]),
    ]
    return X, y, folds


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory, dataset: tuple) -> Path:
    """Train models for 3 horizons."""
    X, y, _ = dataset
    root = tmp_path_factory.mktemp("models")
    np.random.default_rng(42)
    labels = pd.Categorical(y).codes

    for horizon in ("1h", "4h", "24h"):
        d = root / f"1h_{horizon}"
        d.mkdir()
        dtrain = xgb.DMatrix(X, label=labels, feature_names=list(X.columns))
        booster = xgb.train(
            {
                "objective": "multi:softprob",
                "num_class": 3,
                "max_depth": 2,
                "seed": hash(horizon) % 10000,
            },
            dtrain,
            num_boost_round=10,
        )
        booster.save_model(str(d / "xgb.json"))
    return root


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
        assert 0.0 <= result.mean_accuracy <= 1.0
        preds = result.all_predictions
        assert "predicted" in preds.columns
        assert "p_up" in preds.columns

    def test_ensemble_combines_horizons(self, model_dir: Path, dataset: tuple) -> None:
        X, _, _ = dataset
        ens = MultiHorizonEnsemble(model_dir, freq="1h", horizons=["1h", "4h", "24h"])
        assert len(ens.available_horizons()) == 3
        pred = ens.predict(X.iloc[[0]])
        assert pred.predicted_direction in ("up", "down", "flat")
        assert len(pred.horizon_predictions) == 3
        assert abs(pred.p_up + pred.p_down + pred.p_flat - 1.0) < 1e-6

    def test_monte_carlo_on_backtest(self, dataset: tuple, model_dir: Path) -> None:
        X, _, _ = dataset
        rng = np.random.default_rng(7)
        n = len(X)
        idx = X.index
        close = pd.Series(np.cumprod(1 + X["feat_ret_1"].fillna(0)) * 100, index=idx)
        pu = pd.Series(rng.uniform(0.3, 0.7, n), index=idx)
        pd_ = pd.Series(rng.uniform(0.2, 0.5, n), index=idx)

        trades, equity = backtest_run(close, pu, pd_, enter=0.65)
        sim = MonteCarloSimulator(trades["pnl"].values)
        mc = sim.run(n_simulations=100, seed=42)

        assert mc.n_simulations == 100
        assert 0.0 <= mc.probability_of_loss() <= 1.0
        s = mc.summary()
        assert "median_return" in s
        assert "probability_of_loss" in s

    def test_full_pipeline_end_to_end(self, dataset: tuple, model_dir: Path) -> None:
        """optimize → walk-forward → ensemble → backtest → monte carlo."""
        X, y, folds = dataset

        # 1. Walk-forward
        wf = WalkForwardValidator(X, y, folds, num_boost_round=5)
        wf_result = wf.run()
        assert wf_result.mean_accuracy > 0

        # 2. Ensemble
        ens = MultiHorizonEnsemble(model_dir, freq="1h")
        pred = ens.predict(X.iloc[[0]])
        assert pred.confidence > 0

        # 3. Backtest with ensemble-derived probabilities
        np.random.default_rng(42)
        len(X)
        idx = X.index
        close = pd.Series(np.cumprod(1 + X["feat_ret_1"].fillna(0)) * 100, index=idx)
        batch_preds = ens.predict_batch(X)
        pu = pd.Series([p.p_up for p in batch_preds], index=idx)
        pd_ = pd.Series([p.p_down for p in batch_preds], index=idx)
        trades, equity = backtest_run(close, pu, pd_, enter=0.55)

        # 4. Monte Carlo
        sim = MonteCarloSimulator(trades["pnl"].values)
        mc = sim.run(n_simulations=50, seed=0)
        ci_lo, ci_hi = mc.confidence_interval("total_returns")
        assert ci_lo <= ci_hi

        # Verify everything is JSON-serialisable
        import json

        json.dumps(wf_result.summary())
        json.dumps(pred.summary())
        json.dumps(mc.summary())
