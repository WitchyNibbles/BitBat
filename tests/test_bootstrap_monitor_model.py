from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from scripts import bootstrap_monitor_model


def test_bootstrap_runtime_model_runs_expected_cli_sequence(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text('freq: "5m"\nhorizon: "30m"\n', encoding="utf-8")

    calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def _runner(
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, check
        calls.append((cmd, cwd, env))
        if cmd[3:] == ["model", "train", "--freq", "5m", "--horizon", "30m"]:
            model_dir = cwd / "models" / "5m_30m"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "xgb.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    artifact = bootstrap_monitor_model.bootstrap_runtime_model(
        config_path=config_path,
        start_date="2026-01-01",
        symbol="BTC-USD",
        root_dir=tmp_path,
        runner=_runner,
    )

    assert artifact == tmp_path / "models" / "5m_30m" / "xgb.json"
    assert len(calls) == 3

    prices_cmd = calls[0][0]
    assert prices_cmd[:3] == [sys.executable, "-m", "bitbat.cli"]
    assert prices_cmd[3:] == [
        "prices",
        "pull",
        "--symbol",
        "BTC-USD",
        "--interval",
        "5m",
        "--start",
        "2026-01-01",
    ]

    features_cmd = calls[1][0]
    assert features_cmd[3:] == ["features", "build"]

    train_cmd = calls[2][0]
    assert train_cmd[3:] == ["model", "train", "--freq", "5m", "--horizon", "30m"]

    for _, _, env in calls:
        assert env["BITBAT_CONFIG"] == str(config_path)
        assert "PYTHONPATH" in env


def test_bootstrap_runtime_model_raises_if_training_did_not_create_artifact(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text('freq: "5m"\nhorizon: "30m"\n', encoding="utf-8")

    def _runner(
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cmd, cwd, env, capture_output, text, check
        return subprocess.CompletedProcess(["ok"], 0, "ok", "")

    with pytest.raises(FileNotFoundError, match="models/5m_30m/xgb.json"):
        bootstrap_monitor_model.bootstrap_runtime_model(
            config_path=config_path,
            start_date="2026-01-01",
            symbol="BTC-USD",
            root_dir=tmp_path,
            runner=_runner,
        )
