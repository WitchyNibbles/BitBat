from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import init_database
from bitbat.autonomous.predictor import LivePredictor

pytestmark = pytest.mark.integration


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'predictor.db'}"


def _runtime_config(tmp_path: Path) -> dict[str, object]:
    return {
        "data_dir": str(tmp_path / "data"),
        "tau": 0.01,
        "enable_garch": False,
        "enable_sentiment": False,
        "enable_macro": False,
        "enable_onchain": False,
    }


def _bars_and_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    now = datetime.now(UTC).replace(tzinfo=None)
    index = pd.date_range(now - timedelta(hours=39), periods=40, freq="h")
    bars = pd.DataFrame({"close": [100.0 + idx for idx in range(40)]}, index=index)
    features = pd.DataFrame(
        {"feat_close": [0.1 + idx / 1000 for idx in range(40)]},
        index=index,
    )
    return bars, features


@pytest.mark.parametrize(
    ("direction", "expected_multiplier"),
    [
        ("up", 1.01),
        ("down", 0.99),
        ("flat", 1.0),
    ],
)
def test_predict_latest_derives_classifier_predicted_price_from_tau(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    expected_multiplier: float,
) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    monkeypatch.setattr(
        "bitbat.autonomous.predictor.get_runtime_config",
        lambda: _runtime_config(tmp_path),
    )

    predictor = LivePredictor(db=db, freq="1h", horizon="4h")
    bars, features = _bars_and_features()
    latest_price = float(bars.iloc[-1]["close"])

    class DummyBooster:
        feature_names = ["feat_close"]

    monkeypatch.setattr(predictor, "_load_model", lambda: DummyBooster())
    monkeypatch.setattr(
        "bitbat.autonomous.predictor._load_ingested_prices",
        lambda data_dir, freq: bars,
    )
    monkeypatch.setattr(
        "bitbat.autonomous.predictor.generate_price_features",
        lambda prices, enable_garch, freq: features,
    )
    monkeypatch.setattr(
        "bitbat.autonomous.predictor.predict_bar",
        lambda *args, **kwargs: {
            "predicted_direction": direction,
            "predicted_return": None,
            "predicted_price": None,
            "p_up": 0.8 if direction == "up" else 0.1,
            "p_down": 0.8 if direction == "down" else 0.1,
            "p_flat": 0.8 if direction == "flat" else 0.1,
            "confidence": 0.8,
        },
    )

    result = predictor.predict_latest()

    assert result["status"] == "generated"
    assert result["predicted_price"] == pytest.approx(latest_price * expected_multiplier)
    assert result["predicted_return"] is None

    with db.session() as session:
        stored = db.get_recent_predictions(session, "1h", "4h", days=7, realized_only=False)[0]

    assert stored.predicted_direction == direction
    assert stored.predicted_price == pytest.approx(latest_price * expected_multiplier)


def test_predict_latest_preserves_model_supplied_predicted_price(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    monkeypatch.setattr(
        "bitbat.autonomous.predictor.get_runtime_config",
        lambda: _runtime_config(tmp_path),
    )

    predictor = LivePredictor(db=db, freq="1h", horizon="4h")
    bars, features = _bars_and_features()

    class DummyBooster:
        feature_names = ["feat_close"]

    monkeypatch.setattr(predictor, "_load_model", lambda: DummyBooster())
    monkeypatch.setattr(
        "bitbat.autonomous.predictor._load_ingested_prices",
        lambda data_dir, freq: bars,
    )
    monkeypatch.setattr(
        "bitbat.autonomous.predictor.generate_price_features",
        lambda prices, enable_garch, freq: features,
    )
    monkeypatch.setattr(
        "bitbat.autonomous.predictor.predict_bar",
        lambda *args, **kwargs: {
            "predicted_direction": "up",
            "predicted_return": 0.02,
            "predicted_price": 123.45,
            "p_up": 0.7,
            "p_down": 0.2,
            "p_flat": 0.1,
            "confidence": 0.7,
        },
    )

    result = predictor.predict_latest()

    assert result["status"] == "generated"
    assert result["predicted_price"] == pytest.approx(123.45)

    with db.session() as session:
        stored = db.get_recent_predictions(session, "1h", "4h", days=7, realized_only=False)[0]

    assert stored.predicted_price == pytest.approx(123.45)
