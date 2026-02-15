from __future__ import annotations

from datetime import datetime

from bitbat.autonomous.metrics import PerformanceMetrics
from bitbat.autonomous.models import PredictionOutcome


def _prediction(
    idx: int,
    *,
    ret: float,
    correct: bool,
    p_up: float = 0.7,
    p_down: float = 0.2,
) -> PredictionOutcome:
    timestamp = datetime(2024, 1, 1, 0, 0)  # deterministic
    return PredictionOutcome(
        id=idx,
        timestamp_utc=timestamp,
        prediction_timestamp=timestamp,
        predicted_direction="up" if correct else "down",
        p_up=p_up,
        p_down=p_down,
        p_flat=1.0 - p_up - p_down,
        actual_return=ret,
        actual_direction="up" if ret > 0 else "down",
        correct=correct,
        model_version="test-v1",
        freq="1h",
        horizon="4h",
    )


def test_metrics_basic_values() -> None:
    predictions = [
        _prediction(1, ret=0.01, correct=True),
        _prediction(2, ret=-0.01, correct=False),
        _prediction(3, ret=0.02, correct=True),
        _prediction(4, ret=-0.005, correct=False),
    ]

    metrics = PerformanceMetrics(predictions)
    payload = metrics.to_dict()

    assert payload["total_predictions"] == 4
    assert payload["realized_predictions"] == 4
    assert payload["hit_rate"] == 0.5
    assert payload["win_streak"] == 1
    assert payload["lose_streak"] == 1
    assert isinstance(payload["sharpe_ratio"], float)
    assert isinstance(payload["max_drawdown"], float)


def test_metrics_handles_empty() -> None:
    metrics = PerformanceMetrics([])
    payload = metrics.to_dict()

    assert payload["total_predictions"] == 0
    assert payload["realized_predictions"] == 0
    assert payload["hit_rate"] == 0.0
    assert payload["sharpe_ratio"] == 0.0
    assert payload["max_drawdown"] == 0.0
