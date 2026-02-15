from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

try:  # pragma: no cover - dependency guard
    import sqlalchemy  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "init_autonomous_db.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_init_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


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
