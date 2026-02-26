from __future__ import annotations

import json
from pathlib import Path

from scripts import run_monitoring_agent


def test_write_heartbeat_includes_runtime_and_config_metadata(tmp_path: Path) -> None:
    heartbeat = tmp_path / "monitoring_agent_heartbeat.json"

    run_monitoring_agent._write_heartbeat(  # noqa: SLF001
        heartbeat,
        status="ok",
        freq="1h",
        horizon="4h",
        interval=300,
        db_url="sqlite:///data/autonomous.db",
        config_source="--config",
        config_path=str(tmp_path / "monitor.yaml"),
    )

    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["freq"] == "1h"
    assert payload["horizon"] == "4h"
    assert payload["config_source"] == "--config"
    assert payload["config_path"] == str(tmp_path / "monitor.yaml")


def test_write_heartbeat_error_payload_keeps_metadata(tmp_path: Path) -> None:
    heartbeat = tmp_path / "monitoring_agent_heartbeat.json"
    runtime_config_path = tmp_path / "runtime.yaml"

    run_monitoring_agent._write_heartbeat(  # noqa: SLF001
        heartbeat,
        status="error",
        freq="5m",
        horizon="30m",
        interval=60,
        db_url="sqlite:///tmp/agent.db",
        config_source="BITBAT_CONFIG",
        config_path=str(runtime_config_path),
        error="predict.store_prediction failed",
    )

    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["config_source"] == "BITBAT_CONFIG"
    assert payload["config_path"] == str(runtime_config_path)
    assert "predict.store_prediction" in payload["error"]
