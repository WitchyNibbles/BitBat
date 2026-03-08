"""Contract tests: retrainer subprocess commands must match the real CLI interface."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import init_database
from bitbat.autonomous.retrainer import AutoRetrainer
from bitbat.cli import _cli

pytestmark = pytest.mark.behavioral


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'contract.db'}"


def _make_retrainer(tmp_path: Path, monkeypatch) -> tuple[AutoRetrainer, list[list[str]]]:
    """Create an AutoRetrainer with captured subprocess commands."""
    database_url = _db_url(tmp_path)
    init_database(database_url)
    db = AutonomousDB(database_url)
    retrainer = AutoRetrainer(db, "1h", "4h")

    commands: list[list[str]] = []

    def _capture(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(retrainer, "_run_command", _capture)
    monkeypatch.setattr(retrainer, "_read_cv_score", lambda: 0.5)
    monkeypatch.setattr(retrainer, "_training_sample_count", lambda: 1000)

    return retrainer, commands


def _find_features_build_command(commands: list[list[str]]) -> list[str]:
    """Find the 'features build' subprocess invocation."""
    for cmd in commands:
        if len(cmd) >= 5 and cmd[3] == "features" and cmd[4] == "build":
            return cmd
    raise AssertionError(f"No 'features build' command found among captured commands: {commands}")


def test_retrainer_features_build_has_no_tau_arg(tmp_path: Path, monkeypatch) -> None:
    """The retrainer must not pass --tau to 'features build' (it reads tau from config)."""
    retrainer, commands = _make_retrainer(tmp_path, monkeypatch)
    retrainer.retrain()

    features_cmd = _find_features_build_command(commands)
    assert (
        "--tau" not in features_cmd
    ), f"'--tau' must not appear in the features build command: {features_cmd}"


def test_retrainer_subprocess_args_match_cli(tmp_path: Path, monkeypatch) -> None:
    """Every --flag the retrainer passes to 'features build' must be a valid CLI option."""
    retrainer, commands = _make_retrainer(tmp_path, monkeypatch)
    retrainer.retrain()

    features_cmd = _find_features_build_command(commands)

    # Extract valid option names from the real CLI 'features build' --help
    runner = CliRunner()
    result = runner.invoke(_cli, ["features", "build", "--help"])
    assert result.exit_code == 0, f"features build --help failed: {result.output}"

    # Parse valid options from the help text (lines like "  --start TEXT  ...")
    valid_options: set[str] = set()
    for line in result.output.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            option_name = stripped.split()[0].split("/")[0]
            valid_options.add(option_name)

    # Check every --flag in the subprocess command is valid
    invalid_flags = []
    for arg in features_cmd:
        if arg.startswith("--"):  # noqa: SIM102
            if arg not in valid_options:
                invalid_flags.append(arg)

    assert not invalid_flags, (
        f"Retrainer passes invalid flags to 'features build': {invalid_flags}. "
        f"Valid options are: {sorted(valid_options)}"
    )
