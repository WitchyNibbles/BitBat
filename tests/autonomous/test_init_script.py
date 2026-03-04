from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from bitbat.autonomous.models import create_database_engine

try:  # pragma: no cover - dependency guard
    import sqlalchemy  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "init_autonomous_db.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


pytestmark = pytest.mark.integration

def _run_init_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _create_legacy_prediction_outcomes(database_url: str) -> None:
    engine = create_database_engine(database_url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS prediction_outcomes"))
        conn.execute(text(
            """
            CREATE TABLE prediction_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc DATETIME NOT NULL,
                prediction_timestamp DATETIME NOT NULL,
                predicted_direction VARCHAR(10) NOT NULL,
                p_up FLOAT,
                p_down FLOAT,
                p_flat FLOAT,
                predicted_return FLOAT,
                actual_return FLOAT,
                actual_direction VARCHAR(10),
                correct BOOLEAN,
                model_version VARCHAR(64) NOT NULL,
                freq VARCHAR(16) NOT NULL,
                horizon VARCHAR(16) NOT NULL,
                features_used JSON,
                created_at DATETIME NOT NULL,
                realized_at DATETIME
            )
            """
        ))
        conn.execute(text(
            """
            INSERT INTO prediction_outcomes (
                timestamp_utc,
                prediction_timestamp,
                predicted_direction,
                p_up,
                p_down,
                p_flat,
                predicted_return,
                actual_return,
                actual_direction,
                correct,
                model_version,
                freq,
                horizon,
                features_used,
                created_at,
                realized_at
            ) VALUES (
                '2026-02-24 00:00:00',
                '2026-02-24 00:00:00',
                'up',
                0.6,
                0.3,
                0.1,
                0.005,
                NULL,
                NULL,
                NULL,
                'legacy-v1',
                '1h',
                '4h',
                NULL,
                '2026-02-24 00:00:00',
                NULL
            )
            """
        ))
    engine.dispose()


def _prediction_row_snapshot(database_url: str) -> tuple[int, str, float | None, str]:
    engine = create_database_engine(database_url)
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT id, predicted_direction, predicted_return, model_version "
            "FROM prediction_outcomes LIMIT 1"
        )).one()
        total = int(conn.execute(text("SELECT COUNT(*) FROM prediction_outcomes")).scalar() or 0)
    engine.dispose()
    return total, str(row[1]), float(row[2]) if row[2] is not None else None, str(row[3])


def test_init_script_create_then_force_reset(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'script_autonomous.db'}"

    first_run = _run_init_script("--database-url", db_url)
    assert first_run.returncode == 0, first_run.stdout + first_run.stderr
    assert "Autonomous database initialization complete." in first_run.stdout

    second_run = _run_init_script("--database-url", db_url)
    assert second_run.returncode == 1
    assert "Found existing autonomous tables" in second_run.stdout

    third_run = _run_init_script("--database-url", db_url, "--force")
    assert third_run.returncode == 0, third_run.stdout + third_run.stderr


def test_init_script_upgrade_is_repeat_safe_and_reports_status(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'script_upgrade_repeat.db'}"
    _create_legacy_prediction_outcomes(db_url)

    first_run = _run_init_script("--database-url", db_url, "--upgrade")
    assert first_run.returncode == 0, first_run.stdout + first_run.stderr
    assert (
        "Upgrade status: upgraded (operations=1, missing_before=1, missing_after=0)"
        in first_run.stdout
    )
    assert "Compatibility status: PASS" in first_run.stdout

    second_run = _run_init_script("--database-url", db_url, "--upgrade")
    assert second_run.returncode == 0, second_run.stdout + second_run.stderr
    assert (
        "Upgrade status: already_compatible (operations=0, missing_before=0, missing_after=0)"
        in second_run.stdout
    )
    assert "Compatibility status: PASS" in second_run.stdout

    total, predicted_direction, predicted_return, model_version = _prediction_row_snapshot(db_url)
    assert total == 1
    assert predicted_direction == "up"
    assert predicted_return == pytest.approx(0.005, abs=1e-9)
    assert model_version == "legacy-v1"
