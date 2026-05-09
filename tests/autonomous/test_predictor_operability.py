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


def test_predict_latest_skips_when_unrealized_prediction_exists(
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

    with db.session() as session:
        db.store_prediction(
            session=session,
            timestamp_utc=datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1),
            predicted_direction="up",
            p_up=0.7,
            p_down=0.2,
            model_version="v1.0.0",
            freq="1h",
            horizon="4h",
            predicted_return=0.01,
            predicted_price=101.0,
        )

    predictor = LivePredictor(db=db, freq="1h", horizon="4h")

    class DummyBooster:
        feature_names = ["feat_close"]

    monkeypatch.setattr(predictor, "_load_model", lambda: DummyBooster())

    def _unexpected_load_prices(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        raise AssertionError("price loading should not run while a prediction is still pending")

    monkeypatch.setattr(
        "bitbat.autonomous.predictor._load_ingested_prices",
        _unexpected_load_prices,
    )

    result = predictor.predict_latest()

    assert result["status"] == "no_prediction"
    assert result["reason"] == "pending_realization"
    assert result["details"]["pending_predictions"] == 1

    with db.session() as session:
        stored = db.get_recent_predictions(session, "1h", "4h", days=7, realized_only=False)

    assert len(stored) == 1


def test_model_path_uses_active_model_family_metadata(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    with db.session() as session:
        db.store_model_version(
            session=session,
            version="v-rf",
            freq="1h",
            horizon="4h",
            training_start=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1),
            training_end=datetime.now(UTC).replace(tzinfo=None),
            training_samples=100,
            cv_score=0.61,
            features=["feat_close"],
            hyperparameters=None,
            training_metadata={"family": "random_forest"},
            is_active=True,
        )

    predictor = LivePredictor(db=db, freq="1h", horizon="4h")

    assert predictor._model_path().name == "random_forest.pkl"
