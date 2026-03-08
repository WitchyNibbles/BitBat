"""Tests for WalkForwardValidator (regression mode)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bitbat.dataset.splits import Fold, generate_rolling_windows, walk_forward
from bitbat.model.walk_forward import (
    FoldResult,
    WalkForwardResult,
    WalkForwardValidator,
)

pytestmark = pytest.mark.behavioral

@pytest.fixture(scope="module")
def synthetic_data() -> (
    tuple[pd.DataFrame, pd.Series, list[Fold]]
):
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    X = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n)},
        index=idx,
    )
    y = pd.Series(
        rng.normal(0.0, 0.01, size=n),
        index=idx,
        dtype="float64",
    )
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
    def test_returns_result(
        self, result: WalkForwardResult
    ) -> None:
        assert isinstance(result, WalkForwardResult)

    def test_two_folds(
        self, result: WalkForwardResult
    ) -> None:
        assert result.n_folds == 2

    def test_mean_rmse_positive(
        self, result: WalkForwardResult
    ) -> None:
        assert result.mean_rmse > 0

    def test_mean_mae_positive(
        self, result: WalkForwardResult
    ) -> None:
        assert result.mean_mae > 0

    def test_mean_directional_accuracy_in_range(
        self, result: WalkForwardResult
    ) -> None:
        assert 0.0 <= result.mean_directional_accuracy <= 1.0

    def test_fold_results_have_data(
        self, result: WalkForwardResult
    ) -> None:
        for fr in result.fold_results:
            assert isinstance(fr, FoldResult)
            assert fr.train_size > 0
            assert fr.test_size > 0
            assert fr.rmse > 0
            assert fr.diagnostics["window_id"].startswith("fold-")
            assert "regime" in fr.diagnostics
            assert fr.diagnostics["n_samples"] == fr.test_size

    def test_fold_metadata_captures_leakage_controls(
        self, result: WalkForwardResult
    ) -> None:
        for fr in result.fold_results:
            assert "embargo_bars" in fr.window_metadata
            assert "purge_bars" in fr.window_metadata
            assert fr.window_metadata["embargo_bars"] >= 0
            assert fr.window_metadata["purge_bars"] >= 0

    def test_all_predictions_concatenated(
        self, result: WalkForwardResult
    ) -> None:
        preds = result.all_predictions
        assert isinstance(preds, pd.DataFrame)
        assert len(preds) == 100  # 50 + 50 test bars

    def test_predictions_have_columns(
        self, result: WalkForwardResult
    ) -> None:
        preds = result.all_predictions
        assert "predicted" in preds.columns
        assert "actual" in preds.columns

    def test_summary_dict(
        self, result: WalkForwardResult
    ) -> None:
        s = result.summary()
        assert "n_folds" in s
        assert "mean_rmse" in s
        assert "total_test_samples" in s
        assert "fold_diagnostics" in s
        assert s["n_folds"] == 2

    def test_summary_includes_candidate_report_payload(
        self, result: WalkForwardResult
    ) -> None:
        s = result.summary()
        assert "candidate_report" in s
        assert "regression" in s["candidate_report"]
        assert "directional" in s["candidate_report"]
        assert "risk" in s["candidate_report"]

    def test_custom_xgb_params(
        self, synthetic_data: tuple
    ) -> None:
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
        y = pd.Series([0.01])
        v = WalkForwardValidator(X, y, [])
        r = v.run()
        assert r.n_folds == 0
        assert r.mean_rmse == 0.0


def test_generate_rolling_windows_is_deterministic() -> None:
    idx = pd.date_range("2024-01-01 00:00:00", periods=24 * 25, freq="1h")
    windows = generate_rolling_windows(
        idx,
        train_window="5D",
        backtest_window="2D",
        step="2D",
        start="2024-01-01 00:00:00",
        end="2024-01-20 00:00:00",
    )

    assert len(windows) == 7
    assert windows[0] == (
        "2024-01-01 00:00:00",
        "2024-01-06 00:00:00",
        "2024-01-06 00:00:00",
        "2024-01-08 00:00:00",
    )
    assert windows[-1] == (
        "2024-01-13 00:00:00",
        "2024-01-18 00:00:00",
        "2024-01-18 00:00:00",
        "2024-01-20 00:00:00",
    )


def test_walk_forward_with_generated_windows_preserves_ordering() -> None:
    idx = pd.date_range("2024-01-01 00:00:00", periods=24 * 30, freq="1h")
    windows = generate_rolling_windows(
        idx,
        train_window="4D",
        backtest_window="2D",
        step="2D",
        start="2024-01-01 00:00:00",
        end="2024-01-21 00:00:00",
    )

    folds = walk_forward(indices=idx, windows=windows, embargo_bars=1)
    assert len(folds) == len(windows)
    assert len(folds) > 0
    assert any(not fold.train.empty and not fold.test.empty for fold in folds)
    for fold in folds:
        if fold.train.empty or fold.test.empty:
            continue
        assert fold.train.max() < fold.test.min()


def test_walk_forward_summary_includes_cost_fee_and_slippage_totals() -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=120, freq="1h")
    X = pd.DataFrame(
        {"a": rng.normal(size=len(idx)), "b": rng.normal(size=len(idx))},
        index=idx,
    )
    y = pd.Series(rng.normal(0.0, 0.01, size=len(idx)), index=idx, dtype="float64")
    prices = pd.Series(100.0 * np.cumprod(1.0 + y.to_numpy()), index=idx)
    folds = [Fold(train=idx[:80], test=idx[80:120])]

    validator = WalkForwardValidator(
        X,
        y,
        folds,
        num_boost_round=5,
        prices=prices,
        fee_bps=3.0,
        slippage_bps=2.0,
    )
    result = validator.run().summary()

    assert "total_fee_costs" in result
    assert "total_slippage_costs" in result
    assert "mean_gross_return" in result
