from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

import pandas as pd
import pytest

from bitbat.autonomous.continuous_trainer import ContinuousTrainer
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import init_database

pytestmark = pytest.mark.behavioral


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'continuous_trainer.db'}"


def test_continuous_trainer_dynamically_scales_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    # Initialize trainer with large required windows
    config = {
        "data_dir": str(tmp_path),
        "continuous_training": {
            "train_window_bars": 3000,
            "backtest_window_bars": 1000,
        },
    }
    trainer = ContinuousTrainer(db, freq="1h", horizon="4h", config=config)

    # Provide only 500 fake samples — strictly less than 3000 + 1000
    times = pd.date_range("2024-01-01 10:00", periods=500, freq="h", tz="UTC")
    # Use variable data to prevent technical indicators from computing as NaN
    base_prices = 40_000.0 + np.arange(500) * 10.0 + np.random.randn(500) * 50
    fake_prices = pd.DataFrame(
        {
            "open": base_prices,
            "high": base_prices + 500.0,
            "low": base_prices - 500.0,
            "close": base_prices + 100.0,
            "volume": 1_000.0 + np.arange(500),
        },
        index=times,
    )

    monkeypatch.setattr(trainer, "_load_prices", lambda: fake_prices)

    class MockBooster:
        def predict(self, dtest):
            # return a simulated probability array (N, 3)
            return np.ones((dtest.num_row(), 3)) / 3.0
        def save_model(self, *args, **kwargs):
            pass

    def fake_fit(X, y, seed):
        m = MockBooster()
        return m, None

    monkeypatch.setattr("bitbat.model.train.fit_xgb", fake_fit)
    
    import xgboost as xgb
    class MockDMatrix:
        def __init__(self, data, *args, **kwargs):
            self.n = len(data)
        def num_row(self):
            return self.n
    monkeypatch.setattr(xgb, "DMatrix", MockDMatrix)
    
    monkeypatch.setattr("bitbat.model.evaluate.classification_probability_metrics", lambda *args, **kwargs: {
        "pr_auc": 0.8, "mlogloss": 1.2, "directional_accuracy": 0.55, "n_samples": 100
    })
    
    # Mock settings directory lookups to avoid crash on yaml load bypasses
    monkeypatch.setattr("bitbat.autonomous.continuous_trainer.resolve_metrics_dir", lambda: tmp_path)
    monkeypatch.setattr("bitbat.autonomous.continuous_trainer.resolve_models_dir", lambda: tmp_path)

    logger_warnings = []
    monkeypatch.setattr("bitbat.autonomous.continuous_trainer.logger.warning", lambda msg, *args: logger_warnings.append(msg))

    # Should not raise ValueError! Should scale and deploy new model.
    result = trainer._do_retrain("old_v1")

    assert result["deployed"] is True
    assert "Scaling down retraining windows" in logger_warnings[0]
    
    meta = result["window_metadata"]
    assert meta["train_window_bars"] < 3000
    assert meta["backtest_window_bars"] < 1000
    assert meta["train_window_bars"] + meta["backtest_window_bars"] <= 500


def test_continuous_trainer_raises_on_absolute_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    config = {
        "data_dir": str(tmp_path),
        "continuous_training": {
            "train_window_bars": 3000,
            "backtest_window_bars": 1000,
        },
    }
    trainer = ContinuousTrainer(db, freq="1h", horizon="4h", config=config)

    # Provide 110 fake price samples to pass the first check (len(prices) >= 100) 
    # but fail the second (len(features) < 80) since feature generation drops ~37 rows.
    # 110 - 37 = 73 < 80 minimum.
    times = pd.date_range("2024-01-01 10:00", periods=110, freq="h", tz="UTC")
    base_prices = 40_000.0 + np.arange(110) * 10.0 + np.random.randn(110) * 50
    fake_prices = pd.DataFrame(
        {
            "open": base_prices,
            "high": base_prices + 500.0,
            "low": base_prices - 500.0,
            "close": base_prices + 100.0,
            "volume": 1_000.0 + np.arange(110),
        },
        index=times,
    )

    monkeypatch.setattr(trainer, "_load_prices", lambda: fake_prices)

    with pytest.raises(ValueError, match="Not enough samples to retrain"):
        trainer._do_retrain("old_v1")
