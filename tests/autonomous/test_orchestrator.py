"""Tests for the one-click training orchestrator."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bitbat.autonomous.orchestrator import one_click_train


pytestmark = pytest.mark.behavioral

@pytest.fixture(autouse=True)
def _mock_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a minimal runtime config for all tests."""
    cfg: dict[str, Any] = {
        "data_dir": "data",
        "enable_sentiment": False,
        "enable_garch": False,
        "enable_macro": False,
        "enable_onchain": False,
        "autonomous": {"database_url": "sqlite:///:memory:"},
    }
    import bitbat.config.loader as loader_mod

    monkeypatch.setattr(loader_mod, "get_runtime_config", lambda: cfg)
    monkeypatch.setattr(loader_mod, "load_config", lambda **kw: cfg)


def _fake_xy() -> tuple[pd.DataFrame, pd.Series, Any]:
    """Minimal feature matrix + labels for mocking build_xy."""
    X = pd.DataFrame(
        {"feat_a": [1.0, 2.0, 3.0], "feat_b": [4.0, 5.0, 6.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )
    y = pd.Series(["up", "down", "up"], index=X.index)
    meta = MagicMock()
    return X, y, meta


def test_one_click_success(tmp_path: Any) -> None:
    """Happy path: all steps succeed."""
    # Create the expected parquet file
    prices_dir = tmp_path / "data" / "raw" / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=5, freq="h"),
        "close": range(5),
    }).to_parquet(prices_dir / "btcusd_yf_1h.parquet")

    # Override config to use tmp_path (via monkeypatch for safe cleanup)
    import bitbat.config.loader as loader_mod

    loader_mod._ACTIVE_CONFIG = {
        "data_dir": str(tmp_path / "data"),
        "enable_sentiment": False,
        "enable_garch": False,
        "enable_macro": False,
        "enable_onchain": False,
        "autonomous": {"database_url": "sqlite:///:memory:"},
    }

    X, y, meta = _fake_xy()
    mock_db = MagicMock()
    mock_db.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_db.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_db.get_active_model.return_value = None

    progress_calls: list[tuple[str, float]] = []

    with (
        patch("bitbat.ingest.prices.fetch_yf", return_value=pd.DataFrame()),
        patch("bitbat.dataset.build.build_xy", return_value=(X, y, meta)),
        patch("bitbat.model.train.fit_xgb", return_value=(MagicMock(), {"feat_a": 1.0})),
        patch("bitbat.autonomous.models.init_database"),
        patch("bitbat.autonomous.db.AutonomousDB", return_value=mock_db),
        patch("bitbat.autonomous.predictor.LivePredictor") as mock_predictor_cls,
    ):
        mock_predictor_cls.return_value.predict_latest.return_value = {"direction": "up"}

        result = one_click_train(
            preset_name="balanced",
            progress_callback=lambda msg, frac: progress_calls.append((msg, frac)),
        )

    # Restore config to avoid leaking into other tests
    loader_mod.reset_runtime_config()

    assert result["status"] == "success"
    assert result["training_samples"] == 3
    assert "model_version" in result
    assert result["duration_seconds"] >= 0

    # Progress should increase monotonically
    fracs = [f for _, f in progress_calls]
    assert fracs == sorted(fracs)
    assert fracs[-1] == 1.0


def test_one_click_fails_gracefully_on_ingest_error() -> None:
    """When price ingestion fails, return status=failed with step info."""
    with patch(
        "bitbat.ingest.prices.fetch_yf",
        side_effect=RuntimeError("network down"),
    ):
        result = one_click_train(preset_name="balanced")

    assert result["status"] == "failed"
    assert result["step"] == "ingest_prices"
    assert "network down" in result["error"]
