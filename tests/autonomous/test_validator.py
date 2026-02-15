from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import init_database
from bitbat.autonomous.validator import PredictionValidator


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'validator.db'}"


def _seed_prices(tmp_path: Path, freq: str = "1h") -> None:
    prices_dir = tmp_path / "data" / "raw" / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    index = pd.date_range("2024-01-01 00:00:00", periods=16, freq=freq)
    frame = pd.DataFrame({
        "timestamp_utc": index,
        "open": [43000 + i for i in range(len(index))],
        "high": [43010 + i for i in range(len(index))],
        "low": [42990 + i for i in range(len(index))],
        "close": [43000 + (i * 10) for i in range(len(index))],
        "volume": [1000.0] * len(index),
        "source": ["yfinance"] * len(index),
    })
    frame.to_parquet(prices_dir / f"btcusd_yf_{freq}.parquet", index=False)


def test_parse_horizon() -> None:
    db = AutonomousDB("sqlite:///:memory:")
    validator = PredictionValidator(db=db, freq="1h", horizon="4h")

    assert validator.horizon_delta == timedelta(hours=4)
    assert validator._parse_horizon("2d") == timedelta(days=2)  # noqa: SLF001
    assert validator._parse_horizon("30m") == timedelta(minutes=30)  # noqa: SLF001

    with pytest.raises(ValueError):
        validator._parse_horizon("4x")  # noqa: SLF001


def test_find_predictions_to_validate(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    old_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=8)
    recent_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)

    with db.session() as session:
        db.store_prediction(
            session=session,
            timestamp_utc=old_time,
            predicted_direction="up",
            p_up=0.7,
            p_down=0.2,
            model_version="v1",
            freq="1h",
            horizon="4h",
        )
        db.store_prediction(
            session=session,
            timestamp_utc=recent_time,
            predicted_direction="down",
            p_up=0.2,
            p_down=0.7,
            model_version="v1",
            freq="1h",
            horizon="4h",
        )

    validator = PredictionValidator(db=db, freq="1h", horizon="4h")
    predictions = validator.find_predictions_to_validate()
    assert len(predictions) == 1


def test_fetch_price_data_and_matching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_prices(tmp_path)
    monkeypatch.chdir(tmp_path)
    db = AutonomousDB("sqlite:///:memory:")
    validator = PredictionValidator(db=db, freq="1h", horizon="4h")

    start = datetime(2024, 1, 1, 1, 0, 0)
    end = datetime(2024, 1, 1, 6, 0, 0)
    price_data = validator.fetch_price_data(start, end)
    assert len(price_data) == 6

    exact = validator.get_price_at_timestamp(price_data, datetime(2024, 1, 1, 2, 0, 0))
    assert exact is not None

    nearby = validator.get_price_at_timestamp(price_data, datetime(2024, 1, 1, 2, 30, 0))
    assert nearby is not None

    missing = validator.get_price_at_timestamp(
        price_data,
        datetime(2024, 1, 2, 2, 0, 0),
        tolerance_minutes=10,
    )
    assert missing is None


def test_validate_prediction(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    validator = PredictionValidator(db=db, freq="1h", horizon="4h", tau=0.01)

    with db.session() as session:
        pred = db.store_prediction(
            session=session,
            timestamp_utc=datetime(2024, 1, 1, 10, 0, 0),
            predicted_direction="up",
            p_up=0.7,
            p_down=0.2,
            model_version="v1",
            freq="1h",
            horizon="4h",
        )
        prediction = pred

    price_data = {
        datetime(2024, 1, 1, 10, 0, 0): 43000.0,
        datetime(2024, 1, 1, 14, 0, 0): 43500.0,
    }
    result = validator.validate_prediction(prediction, price_data)

    assert result is not None
    assert result["actual_direction"] == "up"
    assert result["correct"] is True


def test_validate_batch_updates_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_prices(tmp_path)
    monkeypatch.chdir(tmp_path)
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    timestamps = [
        datetime(2024, 1, 1, 1, 0, 0),
        datetime(2024, 1, 1, 2, 0, 0),
    ]
    with db.session() as session:
        for ts in timestamps:
            db.store_prediction(
                session=session,
                timestamp_utc=ts,
                predicted_direction="up",
                p_up=0.7,
                p_down=0.2,
                model_version="v1",
                freq="1h",
                horizon="4h",
            )

    validator = PredictionValidator(db=db, freq="1h", horizon="4h", tau=0.01)
    pending = validator.find_predictions_to_validate()
    results = validator.validate_batch(pending)

    assert results["validated_count"] == 2
    assert results["errors"] == []

    with db.session() as session:
        realized = db.get_recent_predictions(
            session=session,
            freq="1h",
            horizon="4h",
            days=1000,
            realized_only=True,
        )
    assert len(realized) == 2
