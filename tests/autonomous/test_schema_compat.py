from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

try:  # pragma: no cover - dependency guard
    import sqlalchemy  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

from sqlalchemy import text

from bitbat.autonomous.models import create_database_engine
from bitbat.autonomous.schema_compat import (
    audit_schema_compatibility,
    format_schema_audit,
    required_columns_for_table,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "init_autonomous_db.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _db_url(tmp_path: Path, filename: str) -> str:
    return f"sqlite:///{tmp_path / filename}"


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
    engine.dispose()


def _prediction_columns(database_url: str) -> set[str]:
    engine = create_database_engine(database_url)
    with engine.connect() as conn:
        rows = conn.execute(text("PRAGMA table_info(prediction_outcomes)"))
        columns = {str(row[1]) for row in rows}
    engine.dispose()
    return columns


def test_required_contract_contains_runtime_columns() -> None:
    required = required_columns_for_table("prediction_outcomes")
    assert "predicted_price" in required
    assert "predicted_return" in required
    assert "prediction_timestamp" in required


def test_audit_detects_legacy_missing_column(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path, "legacy_missing_predicted_price.db")
    _create_legacy_prediction_outcomes(database_url)

    report = audit_schema_compatibility(database_url=database_url)

    assert report.is_compatible is False
    assert report.can_auto_upgrade is True
    assert report.missing_columns == {"prediction_outcomes": ("predicted_price",)}

    table = report.tables[0]
    assert table.addable_missing_columns == ("predicted_price",)
    assert table.blocking_missing_columns == ()


def test_formatted_audit_is_actionable(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path, "legacy_audit_output.db")
    _create_legacy_prediction_outcomes(database_url)

    output = format_schema_audit(audit_schema_compatibility(database_url=database_url))

    assert "Autonomous schema compatibility audit" in output
    assert "prediction_outcomes: incompatible" in output
    assert "missing: predicted_price" in output


def test_init_script_audit_is_non_destructive(tmp_path: Path) -> None:
    database_url = _db_url(tmp_path, "legacy_script_audit.db")
    _create_legacy_prediction_outcomes(database_url)

    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--database-url", database_url, "--audit"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Compatibility status: FAIL" in result.stdout
    assert "missing: predicted_price" in result.stdout
    assert "predicted_price" not in _prediction_columns(database_url)
