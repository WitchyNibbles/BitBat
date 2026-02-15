from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

try:  # pragma: no cover - dependency guard
    import sqlalchemy  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

from sqlalchemy import inspect

from bitbat.autonomous.models import (
    PredictionOutcome,
    get_session,
    init_database,
)


def test_init_database_creates_all_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "autonomous.db"
    engine = init_database(f"sqlite:///{db_path}")
    inspector = inspect(engine)

    assert set(inspector.get_table_names()) == {
        "model_versions",
        "performance_snapshots",
        "prediction_outcomes",
        "retraining_events",
        "system_logs",
    }


def test_prediction_outcome_to_dict_and_is_realized(tmp_path: Path) -> None:
    db_path = tmp_path / "autonomous.db"
    engine = init_database(f"sqlite:///{db_path}")
    session = get_session(engine)

    try:
        pred = PredictionOutcome(
            timestamp_utc=datetime(2026, 1, 1, 0, 0, 0),
            prediction_timestamp=datetime(2026, 1, 1, 0, 0, 0),
            predicted_direction="up",
            p_up=0.70,
            p_down=0.20,
            p_flat=0.10,
            model_version="v1",
            freq="1h",
            horizon="4h",
        )
        session.add(pred)
        session.commit()
        session.refresh(pred)

        payload = pred.to_dict()
        assert payload["predicted_direction"] == "up"
        assert payload["model_version"] == "v1"
        assert pred.is_realized is False
    finally:
        session.close()
