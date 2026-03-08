"""Behavioral tests for the accuracy guardrail feature.

Tests call check_accuracy_guardrail() directly — MonitoringAgent is NOT constructed
because its constructor requires a real model artifact on disk.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from bitbat.autonomous.agent import check_accuracy_guardrail
from bitbat.config.loader import load_config

pytestmark = pytest.mark.behavioral


def _config(
    *,
    enabled: bool = True,
    min_predictions_required: int = 10,
    realized_accuracy_threshold: float = 0.40,
) -> dict[str, Any]:
    """Build a minimal config dict for guardrail tests."""
    return {
        "autonomous": {
            "accuracy_guardrail": {
                "enabled": enabled,
                "min_predictions_required": min_predictions_required,
                "realized_accuracy_threshold": realized_accuracy_threshold,
            }
        }
    }


def _metrics(*, hit_rate: float, realized_predictions: int) -> dict[str, Any]:
    return {
        "hit_rate": hit_rate,
        "realized_predictions": realized_predictions,
    }


# ---------------------------------------------------------------------------
# Test 1: guardrail fires on low accuracy
# ---------------------------------------------------------------------------


def test_guardrail_fires_on_low_accuracy() -> None:
    """hit_rate=0.0, realized=20, threshold=0.40 (default) → alert fired, returns True."""
    metrics = _metrics(hit_rate=0.0, realized_predictions=20)
    config = _config(realized_accuracy_threshold=0.40)

    with patch("bitbat.autonomous.agent.send_alert") as mock_alert:
        result = check_accuracy_guardrail(metrics, "1h", "4h", config=config)

    assert result is True
    mock_alert.assert_called_once()
    call_args = mock_alert.call_args
    assert call_args[0][0] == "WARNING"


# ---------------------------------------------------------------------------
# Test 2: guardrail respects a custom (lower) threshold
# ---------------------------------------------------------------------------


def test_guardrail_respects_custom_threshold() -> None:
    """hit_rate=0.35, realized=20, threshold=0.30 → 0.35 >= 0.30 so no alert, returns False."""
    metrics = _metrics(hit_rate=0.35, realized_predictions=20)
    config = _config(realized_accuracy_threshold=0.30)

    with patch("bitbat.autonomous.agent.send_alert") as mock_alert:
        result = check_accuracy_guardrail(metrics, "1h", "4h", config=config)

    assert result is False
    mock_alert.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: guardrail skips when realized count < min_predictions_required
# ---------------------------------------------------------------------------


def test_guardrail_skips_insufficient_samples() -> None:
    """hit_rate=0.0, realized=5, min_predictions_required=10 → not enough samples, returns False."""
    metrics = _metrics(hit_rate=0.0, realized_predictions=5)
    config = _config(min_predictions_required=10)

    with patch("bitbat.autonomous.agent.send_alert") as mock_alert:
        result = check_accuracy_guardrail(metrics, "1h", "4h", config=config)

    assert result is False
    mock_alert.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: alert details contain required keys
# ---------------------------------------------------------------------------


def test_guardrail_alert_details() -> None:
    """Alert details dict must contain required keys.

    Keys: observed_accuracy, threshold, realized_predictions, freq, horizon.
    """
    metrics = _metrics(hit_rate=0.10, realized_predictions=20)
    config = _config(realized_accuracy_threshold=0.40)

    with patch("bitbat.autonomous.agent.send_alert") as mock_alert:
        check_accuracy_guardrail(metrics, "1h", "4h", config=config)

    mock_alert.assert_called_once()
    _level, _message, details = mock_alert.call_args[0]
    assert "observed_accuracy" in details
    assert "threshold" in details
    assert "realized_predictions" in details
    assert "freq" in details
    assert "horizon" in details
    assert details["observed_accuracy"] == pytest.approx(0.10)
    assert details["threshold"] == pytest.approx(0.40)
    assert details["realized_predictions"] == 20
    assert details["freq"] == "1h"
    assert details["horizon"] == "4h"


# ---------------------------------------------------------------------------
# Test 5: config key exists in default.yaml
# ---------------------------------------------------------------------------


def test_guardrail_config_key_in_default_yaml() -> None:
    """default.yaml must have autonomous.accuracy_guardrail.realized_accuracy_threshold == 0.40."""
    config = load_config()
    guardrail = config["autonomous"]["accuracy_guardrail"]
    assert guardrail["realized_accuracy_threshold"] == pytest.approx(0.40)
