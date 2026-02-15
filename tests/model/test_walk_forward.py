"""Tests for WalkForwardValidator (Phase 5, Session 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitbat.dataset.splits import Fold
from bitbat.model.walk_forward import FoldResult, WalkForwardResult, WalkForwardValidator


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[pd.DataFrame, pd.Series, list[Fold]]:
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)
    y = pd.Series(rng.choice(["up", "down", "flat"], size=n), index=idx)
    folds = [
        Fold(train=idx[:100], test=idx[100:150]),
        Fold(train=idx[:150], test=idx[150:200]),
    ]
    return X, y, folds


@pytest.fixture(scope="module")
def result(synthetic_data: tuple) -> WalkForwardResult:
    X, y, folds = synthetic_data
    v = WalkForwardValidator(X, y, folds, num_boost_round=10)
    return v.run()


class TestWalkForwardValidator:
    def test_returns_result(self, result: WalkForwardResult) -> None:
        assert isinstance(result, WalkForwardResult)

    def test_two_folds(self, result: WalkForwardResult) -> None:
        assert result.n_folds == 2

    def test_mean_accuracy_in_range(self, result: WalkForwardResult) -> None:
        assert 0.0 <= result.mean_accuracy <= 1.0

    def test_mean_logloss_positive(self, result: WalkForwardResult) -> None:
        assert result.mean_logloss > 0

    def test_fold_results_have_data(self, result: WalkForwardResult) -> None:
        for fr in result.fold_results:
            assert isinstance(fr, FoldResult)
            assert fr.train_size > 0
            assert fr.test_size > 0
            assert 0.0 <= fr.accuracy <= 1.0

    def test_all_predictions_concatenated(self, result: WalkForwardResult) -> None:
        preds = result.all_predictions
        assert isinstance(preds, pd.DataFrame)
        assert len(preds) == 100  # 50 + 50 test bars

    def test_predictions_have_probabilities(self, result: WalkForwardResult) -> None:
        preds = result.all_predictions
        for col in ("p_down", "p_flat", "p_up"):
            assert col in preds.columns

    def test_summary_dict(self, result: WalkForwardResult) -> None:
        s = result.summary()
        assert "n_folds" in s
        assert "mean_accuracy" in s
        assert "total_test_samples" in s
        assert s["n_folds"] == 2

    def test_custom_xgb_params(self, synthetic_data: tuple) -> None:
        X, y, folds = synthetic_data
        v = WalkForwardValidator(
            X,
            y,
            folds,
            xgb_params={"max_depth": 3, "eta": 0.05},
            num_boost_round=5,
        )
        r = v.run()
        assert r.n_folds == 2

    def test_empty_folds(self) -> None:
        X = pd.DataFrame({"a": [1.0]})
        y = pd.Series(["up"])
        v = WalkForwardValidator(X, y, [])
        r = v.run()
        assert r.n_folds == 0
        assert r.mean_accuracy == 0.0
