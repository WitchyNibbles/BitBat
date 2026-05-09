"""Tests for the one-click training orchestrator."""

from __future__ import annotations

from pathlib import Path
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


def _config_for(tmp_path: Any) -> dict[str, Any]:
    return {
        "data_dir": str(tmp_path / "data"),
        "enable_sentiment": False,
        "enable_garch": False,
        "enable_macro": False,
        "enable_onchain": False,
        "autonomous": {"database_url": "sqlite:///:memory:"},
    }


def _stub_saved_artifact(tmp_path: Any, family: str = "xgb") -> Path:
    artifact = tmp_path / "models" / "artifact" / (
        "xgb.json" if family == "xgb" else "random_forest.pkl"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("stub", encoding="utf-8")
    return artifact


def test_one_click_success(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: all steps succeed."""
    # Create the expected parquet file
    prices_dir = tmp_path / "data" / "raw" / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=5, freq="h"),
        "close": range(5),
    }).to_parquet(prices_dir / "btcusd_yf_1h.parquet")

    import bitbat.config.loader as loader_mod

    cfg = _config_for(tmp_path)
    monkeypatch.setattr(loader_mod, "get_runtime_config", lambda: cfg)
    monkeypatch.setattr(loader_mod, "load_config", lambda **kw: cfg)

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
        patch(
            "bitbat.model.persist.save_baseline_artifact",
            return_value=_stub_saved_artifact(tmp_path),
        ),
        patch("bitbat.autonomous.models.init_database"),
        patch("bitbat.autonomous.db.AutonomousDB", return_value=mock_db),
        patch("bitbat.autonomous.predictor.LivePredictor") as mock_predictor_cls,
    ):
        mock_predictor_cls.return_value.predict_latest.return_value = {"direction": "up"}

        result = one_click_train(
            preset_name="balanced",
            progress_callback=lambda msg, frac: progress_calls.append((msg, frac)),
        )

    assert result["status"] == "success"
    assert result["training_samples"] == 3
    assert "model_version" in result
    assert result["duration_seconds"] >= 0

    # Progress should increase monotonically
    fracs = [f for _, f in progress_calls]
    assert fracs == sorted(fracs)
    assert fracs[-1] == 1.0


def test_one_click_fails_gracefully_on_ingest_error(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When price ingestion fails with no cache, return status=failed with step info."""
    import bitbat.config.loader as loader_mod

    cfg = _config_for(tmp_path)
    monkeypatch.setattr(loader_mod, "get_runtime_config", lambda: cfg)
    monkeypatch.setattr(loader_mod, "load_config", lambda **kw: cfg)

    with patch(
        "bitbat.ingest.prices.fetch_yf",
        side_effect=RuntimeError("network down"),
    ):
        result = one_click_train(preset_name="balanced")

    assert result["status"] == "failed"
    assert result["step"] == "ingest_prices"
    assert "network down" in result["error"]


def test_one_click_uses_cached_prices_when_ingest_fails(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If cached prices exist, the orchestrator should continue after an API failure."""
    prices_dir = tmp_path / "data" / "raw" / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=5, freq="h"),
        "close": range(5),
    }).to_parquet(prices_dir / "btcusd_yf_1h.parquet")

    import bitbat.config.loader as loader_mod

    cfg = _config_for(tmp_path)
    monkeypatch.setattr(loader_mod, "get_runtime_config", lambda: cfg)
    monkeypatch.setattr(loader_mod, "load_config", lambda **kw: cfg)

    X, y, meta = _fake_xy()
    mock_db = MagicMock()
    mock_db.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_db.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_db.get_active_model.return_value = None

    with (
        patch("bitbat.ingest.prices.fetch_yf", side_effect=RuntimeError("network down")),
        patch("bitbat.dataset.build.build_xy", return_value=(X, y, meta)),
        patch("bitbat.model.train.fit_xgb", return_value=(MagicMock(), {"feat_a": 1.0})),
        patch(
            "bitbat.model.persist.save_baseline_artifact",
            return_value=_stub_saved_artifact(tmp_path),
        ),
        patch("bitbat.autonomous.models.init_database"),
        patch("bitbat.autonomous.db.AutonomousDB", return_value=mock_db),
        patch("bitbat.autonomous.predictor.LivePredictor") as mock_predictor_cls,
    ):
        mock_predictor_cls.return_value.predict_latest.return_value = {"direction": "up"}
        result = one_click_train(preset_name="balanced")

    assert result["status"] == "success"


@pytest.mark.parametrize(
    ("preset_name", "expected_family", "expected_label_mode", "expected_labels"),
    [
        ("scalper", "xgb", "triple_barrier", ["take_profit", "stop_loss", "timeout"]),
        ("balanced", "xgb", "return_direction", ["up", "down", "flat"]),
        ("conservative", "random_forest", "return_direction", None),
    ],
)
def test_one_click_train_uses_mode_specific_profile(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    preset_name: str,
    expected_family: str,
    expected_label_mode: str,
    expected_labels: list[str] | None,
) -> None:
    prices_dir = tmp_path / "data" / "raw" / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=5, freq="h"),
        "close": range(5),
    }).to_parquet(prices_dir / "btcusd_yf_1h.parquet")
    pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=5, freq="5min"),
        "close": range(5),
    }).to_parquet(prices_dir / "btcusd_yf_5m.parquet")
    pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=5, freq="15min"),
        "close": range(5),
    }).to_parquet(prices_dir / "btcusd_yf_15m.parquet")

    import bitbat.config.loader as loader_mod

    cfg = _config_for(tmp_path)
    monkeypatch.setattr(loader_mod, "get_runtime_config", lambda: cfg)
    monkeypatch.setattr(loader_mod, "load_config", lambda **kw: cfg)

    X, y, meta = _fake_xy()
    mode_labels = (
        expected_labels
        if expected_labels is not None
        else ["up", "down", "flat"]
    )

    if preset_name == "scalper":
        freq = "5m"
        horizon = "30m"
    elif preset_name == "balanced":
        freq = "1h"
        horizon = "4h"
    else:
        freq = "1h"
        horizon = "24h"

    feature_dir = tmp_path / "data" / "features" / f"{freq}_{horizon}"
    feature_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="h"),
        "label": mode_labels,
        "r_forward": [0.01, -0.01, 0.0],
        "feat_a": [1.0, 2.0, 3.0],
    }).to_parquet(feature_dir / "dataset.parquet", index=False)

    mock_db = MagicMock()
    mock_db.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_db.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_db.get_active_model.return_value = None

    captured: dict[str, Any] = {}

    def _fake_build_xy(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.Series, Any]:
        captured["build_xy_kwargs"] = kwargs
        return X, y, meta

    def _fake_fit_xgb(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs: Any,
    ) -> tuple[MagicMock, dict[str, float]]:
        captured["xgb_labels"] = list(y_train.astype(str))
        captured["xgb_kwargs"] = kwargs
        return MagicMock(), {"feat_a": 1.0}

    def _fake_fit_random_forest(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs: Any,
    ) -> tuple[MagicMock, dict[str, float]]:
        captured["rf_targets_numeric"] = bool(pd.api.types.is_numeric_dtype(y_train))
        return MagicMock(), {"feat_a": 1.0}

    def _fake_save_baseline_artifact(
        model: Any,
        *,
        family: str,
        freq: str,
        horizon: str,
        root: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        captured["saved_family"] = family
        captured["saved_metadata"] = metadata or {}
        artifact = tmp_path / "models" / f"{freq}_{horizon}" / (
            "xgb.json" if family == "xgb" else "random_forest.pkl"
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("stub", encoding="utf-8")
        return artifact

    with (
        patch("bitbat.ingest.prices.fetch_yf", return_value=pd.DataFrame()),
        patch("bitbat.dataset.build.build_xy", side_effect=_fake_build_xy),
        patch("bitbat.model.train.fit_xgb", side_effect=_fake_fit_xgb),
        patch("bitbat.model.train.fit_random_forest", side_effect=_fake_fit_random_forest),
        patch(
            "bitbat.model.persist.save_baseline_artifact",
            side_effect=_fake_save_baseline_artifact,
        ),
        patch("bitbat.autonomous.models.init_database"),
        patch("bitbat.autonomous.db.AutonomousDB", return_value=mock_db),
        patch("bitbat.autonomous.predictor.LivePredictor") as mock_predictor_cls,
    ):
        mock_predictor_cls.return_value.predict_latest.return_value = {"direction": "up"}
        result = one_click_train(preset_name=preset_name)

    assert result["status"] == "success"
    assert captured["saved_family"] == expected_family
    assert captured["build_xy_kwargs"]["label_mode"] == expected_label_mode
    assert captured["saved_metadata"]["preset_name"] == preset_name
    assert captured["saved_metadata"]["label_mode"] == expected_label_mode
    if expected_labels is not None:
        assert captured["xgb_labels"] == expected_labels
        assert captured["xgb_kwargs"]["class_labels"] == expected_labels
    else:
        assert captured["rf_targets_numeric"] is True


def test_one_click_train_uses_walk_forward_selected_family(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices_dir = tmp_path / "data" / "raw" / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=5, freq="h"),
        "close": range(5),
    }).to_parquet(prices_dir / "btcusd_yf_1h.parquet")

    feature_dir = tmp_path / "data" / "features" / "1h_4h"
    feature_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="h"),
        "label": ["up", "down", "flat"],
        "r_forward": [0.01, -0.01, 0.0],
        "feat_a": [1.0, 2.0, 3.0],
    }).to_parquet(feature_dir / "dataset.parquet", index=False)

    import bitbat.config.loader as loader_mod

    cfg = _config_for(tmp_path)
    monkeypatch.setattr(loader_mod, "get_runtime_config", lambda: cfg)
    monkeypatch.setattr(loader_mod, "load_config", lambda **kw: cfg)

    X, y, meta = _fake_xy()
    mock_db = MagicMock()
    mock_db.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_db.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_db.get_active_model.return_value = None

    captured: dict[str, Any] = {}

    def _fake_save_baseline_artifact(
        model: Any,
        *,
        family: str,
        freq: str,
        horizon: str,
        root: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        captured["saved_family"] = family
        captured["saved_metadata"] = metadata or {}
        artifact = tmp_path / "models" / f"{freq}_{horizon}" / "random_forest.pkl"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("stub", encoding="utf-8")
        return artifact

    with (
        patch("bitbat.ingest.prices.fetch_yf", return_value=pd.DataFrame()),
        patch("bitbat.dataset.build.build_xy", return_value=(X, y, meta)),
        patch("bitbat.model.train.fit_random_forest", return_value=(MagicMock(), {"feat_a": 1.0})),
        patch(
            "bitbat.model.mode_selection.select_mode_candidate",
            return_value={
                "selected_family": "random_forest",
                "candidate_reports": {
                    "random_forest": {
                        "metrics": {
                            "directional": {"mean_directional_accuracy": 0.61},
                        }
                    }
                },
                "champion_decision": {"winner": "random_forest", "promote_candidate": True},
            },
        ),
        patch(
            "bitbat.model.persist.save_baseline_artifact",
            side_effect=_fake_save_baseline_artifact,
        ),
        patch("bitbat.autonomous.models.init_database"),
        patch("bitbat.autonomous.db.AutonomousDB", return_value=mock_db),
        patch("bitbat.autonomous.predictor.LivePredictor") as mock_predictor_cls,
    ):
        mock_predictor_cls.return_value.predict_latest.return_value = {"direction": "up"}
        result = one_click_train(preset_name="balanced")

    assert result["status"] == "success"
    assert captured["saved_family"] == "random_forest"
    assert captured["saved_metadata"]["selected_family"] == "random_forest"
    assert captured["saved_metadata"]["champion_decision"]["winner"] == "random_forest"
