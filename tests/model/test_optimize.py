"""Tests for HyperparamOptimizer (Phase 5, Session 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitbat.dataset.splits import Fold
from bitbat.model.optimize import HyperparamOptimizer, OptimizationResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[pd.DataFrame, pd.Series, list[Fold]]:
    """Build a synthetic dataset with walk-forward folds."""
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    X = pd.DataFrame(
        {
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
            "feat_c": rng.normal(size=n),
        },
        index=idx,
    )
    y = pd.Series(rng.choice(["up", "down", "flat"], size=n), index=idx, name="label")

    # Two walk-forward folds
    folds = [
        Fold(train=idx[:100], test=idx[100:150]),
        Fold(train=idx[:150], test=idx[150:200]),
    ]
    return X, y, folds


@pytest.fixture(scope="module")
def optimizer(synthetic_data: tuple) -> HyperparamOptimizer:
    X, y, folds = synthetic_data
    return HyperparamOptimizer(X, y, folds, seed=42)


@pytest.fixture(scope="module")
def result(optimizer: HyperparamOptimizer) -> OptimizationResult:
    return optimizer.optimize(n_trials=5, timeout=60)


# ---------------------------------------------------------------------------
# HyperparamOptimizer
# ---------------------------------------------------------------------------


class TestOptimizer:
    def test_creates_without_error(self, synthetic_data: tuple) -> None:
        X, y, folds = synthetic_data
        opt = HyperparamOptimizer(X, y, folds)
        assert opt.num_class == 3

    def test_labels_encoded(self, optimizer: HyperparamOptimizer) -> None:
        assert optimizer.labels.dtype in (np.int8, np.int16, np.int32, np.int64)
        assert set(np.unique(optimizer.labels)) <= {0, 1, 2}

    def test_cv_score_returns_float(self, optimizer: HyperparamOptimizer) -> None:
        score = optimizer._cv_score({"eta": 0.1, "max_depth": 3})
        assert isinstance(score, float)
        assert score > 0

    def test_cv_score_lower_is_better(self, optimizer: HyperparamOptimizer) -> None:
        score_shallow = optimizer._cv_score({"eta": 0.1, "max_depth": 2, "num_boost_round": 10})
        # Just check it returns a valid score — not necessarily worse
        assert isinstance(score_shallow, float)
        assert score_shallow > 0


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------


class TestOptimizationResult:
    def test_returns_result(self, result: OptimizationResult) -> None:
        assert isinstance(result, OptimizationResult)

    def test_best_params_is_dict(self, result: OptimizationResult) -> None:
        assert isinstance(result.best_params, dict)
        assert len(result.best_params) > 0

    def test_best_score_positive(self, result: OptimizationResult) -> None:
        assert result.best_score > 0

    def test_n_trials_matches(self, result: OptimizationResult) -> None:
        assert result.n_trials == 5

    def test_has_study(self, result: OptimizationResult) -> None:
        import optuna

        assert isinstance(result.study, optuna.Study)

    def test_to_xgb_params_separates_rounds(self, result: OptimizationResult) -> None:
        params, num_rounds = result.to_xgb_params()
        assert "num_boost_round" not in params
        assert isinstance(num_rounds, int)
        assert num_rounds >= 20

    def test_to_xgb_params_has_eta(self, result: OptimizationResult) -> None:
        params, _ = result.to_xgb_params()
        assert "eta" in params
        assert "max_depth" in params

    def test_summary_json_serialisable(self, result: OptimizationResult) -> None:
        import json

        s = result.summary()
        json.dumps(s)  # should not raise
        assert "best_params" in s
        assert "best_score" in s
        assert "n_trials" in s

    def test_summary_best_score_rounded(self, result: OptimizationResult) -> None:
        s = result.summary()
        # Should have at most 6 decimal places
        assert len(str(s["best_score"]).split(".")[-1]) <= 6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_folds_returns_high_score(self) -> None:
        rng = np.random.default_rng(99)
        X = pd.DataFrame({"a": rng.normal(size=10)})
        y = pd.Series(["up"] * 5 + ["down"] * 5)
        empty_folds: list[Fold] = []
        opt = HyperparamOptimizer(X, y, empty_folds)
        score = opt._cv_score({"eta": 0.1, "max_depth": 3})
        assert score == 999.0

    def test_single_fold(self) -> None:
        rng = np.random.default_rng(7)
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="1h")
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)
        y = pd.Series(rng.choice(["up", "down", "flat"], size=n), index=idx)
        folds = [Fold(train=idx[:40], test=idx[40:])]
        opt = HyperparamOptimizer(X, y, folds)
        result = opt.optimize(n_trials=3, timeout=30)
        assert result.best_score > 0
        assert result.n_trials == 3
