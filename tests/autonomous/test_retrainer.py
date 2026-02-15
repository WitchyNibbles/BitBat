from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import RetrainingEvent, init_database
from bitbat.autonomous.retrainer import AutoRetrainer


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'retrainer.db'}"


def test_should_deploy_logic(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    retrainer = AutoRetrainer(db, "1h", "4h")

    assert (
        retrainer.should_deploy(
            {"cv_score": 0.62, "holdout_hit_rate": 0.60},
            {"cv_score": 0.58},
        )
        is True
    )
    assert (
        retrainer.should_deploy(
            {"cv_score": 0.59, "holdout_hit_rate": 0.70},
            {"cv_score": 0.58},
        )
        is False
    )


def test_retrainer_records_failed_event(tmp_path: Path, monkeypatch) -> None:
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)

    now = datetime.now(UTC).replace(tzinfo=None)
    with db.session() as session:
        db.store_model_version(
            session=session,
            version="v1",
            freq="1h",
            horizon="4h",
            training_start=now - timedelta(days=30),
            training_end=now,
            training_samples=500,
            cv_score=0.60,
            features=[],
            hyperparameters={},
            training_metadata={},
            is_active=True,
        )

    retrainer = AutoRetrainer(db, "1h", "4h")

    def _raise(_: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(returncode=1, cmd=["poetry", "run", "bitbat"])

    monkeypatch.setattr(retrainer, "_run_command", _raise)
    result = retrainer.retrain()

    assert result["status"] == "failed"
    with db.session() as session:
        event = session.query(RetrainingEvent).order_by(RetrainingEvent.id.desc()).first()
    assert event is not None
    assert event.status == "failed"
