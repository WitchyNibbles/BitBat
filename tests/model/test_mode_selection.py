from __future__ import annotations

import pytest

from bitbat.model.mode_selection import summarize_candidate_selection

pytestmark = pytest.mark.behavioral


def test_summarize_candidate_selection_prefers_stronger_candidate() -> None:
    summary_by_family = {
        "xgb": [
            {
                "rmse": 0.012,
                "mae": 0.009,
                "directional_accuracy": 0.56,
                "net_sharpe": 0.8,
                "gross_sharpe": 1.0,
                "max_drawdown": -0.18,
                "net_return": 0.08,
                "gross_return": 0.10,
                "total_costs": 0.02,
            }
        ],
        "random_forest": [
            {
                "rmse": 0.010,
                "mae": 0.008,
                "directional_accuracy": 0.61,
                "net_sharpe": 1.2,
                "gross_sharpe": 1.4,
                "max_drawdown": -0.12,
                "net_return": 0.14,
                "gross_return": 0.16,
                "total_costs": 0.02,
            },
            {
                "rmse": 0.0105,
                "mae": 0.0082,
                "directional_accuracy": 0.60,
                "net_sharpe": 1.1,
                "gross_sharpe": 1.3,
                "max_drawdown": -0.13,
                "net_return": 0.13,
                "gross_return": 0.15,
                "total_costs": 0.02,
            },
        ],
    }

    selection = summarize_candidate_selection(
        summary_by_family,
        candidate_families=["xgb", "random_forest"],
        incumbent_family="xgb",
        min_consecutive_outperformance=1,
    )

    assert selection["selected_family"] == "random_forest"
    assert selection["champion_decision"]["winner"] == "random_forest"
    assert (
        selection["candidate_reports"]["random_forest"]["metrics"]["risk"]["mean_net_sharpe"] > 1.0
    )


def test_summarize_candidate_selection_falls_back_to_incumbent_when_gate_fails() -> None:
    summary_by_family = {
        "xgb": [
            {
                "rmse": 0.012,
                "mae": 0.009,
                "directional_accuracy": 0.58,
                "net_sharpe": 0.9,
                "gross_sharpe": 1.0,
                "max_drawdown": -0.15,
                "net_return": 0.09,
                "gross_return": 0.11,
                "total_costs": 0.02,
            }
        ],
        "random_forest": [
            {
                "rmse": 0.009,
                "mae": 0.007,
                "directional_accuracy": 0.62,
                "net_sharpe": 1.4,
                "gross_sharpe": 1.5,
                "max_drawdown": -0.45,
                "net_return": 0.18,
                "gross_return": 0.20,
                "total_costs": 0.02,
            }
        ],
    }

    selection = summarize_candidate_selection(
        summary_by_family,
        candidate_families=["xgb", "random_forest"],
        incumbent_family="xgb",
    )

    assert selection["selected_family"] == "xgb"
    assert selection["champion_decision"]["promotion_gate"]["pass"] is False
