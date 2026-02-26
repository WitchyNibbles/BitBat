"""Phase 19 D1 gate: monitor alignment and diagnostics integrity."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_UPSTREAM_SUITES = [
    "tests/autonomous/test_agent_integration.py",
    "tests/test_cli.py",
    "tests/autonomous/test_schema_compat.py",
    "tests/test_run_monitoring_agent.py",
]


def _source(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_phase19_gate_required_upstream_suites_exist() -> None:
    for rel_path in REQUIRED_UPSTREAM_SUITES:
        assert (ROOT / rel_path).exists(), f"Missing monitor alignment suite: {rel_path}"


def test_phase19_startup_guardrail_contract_remains_anchored() -> None:
    source = _source("tests/autonomous/test_agent_integration.py")

    assert "test_monitoring_agent_blocks_startup_without_model_artifact" in source
    assert "with pytest.raises(FileNotFoundError" in source
    assert "xgb.json" in source
    assert "BITBAT_CONFIG" in source


def test_phase19_cycle_and_status_semantics_contract_remains_anchored() -> None:
    source = _source("tests/test_cli.py")

    assert "test_cli_monitor_run_once_outputs_cycle_state_semantics" in source
    assert 'assert "Prediction state: none" in out' in source
    assert 'assert "Prediction reason: insufficient_data" in out' in source
    assert 'assert "Realization state: pending" in out' in source
    assert 'assert "Cycle diagnostic: insufficient_data" in out' in source

    assert "test_cli_monitor_status_outputs_pair_counts_without_snapshot" in source
    assert 'assert "Total predictions: 2" in status_out' in source
    assert 'assert "Unrealized predictions: 1" in status_out' in source
    assert 'assert "Realized predictions: 1" in status_out' in source


def test_phase19_schema_compatibility_contract_remains_anchored() -> None:
    source = _source("tests/autonomous/test_schema_compat.py")

    assert "test_required_contract_contains_performance_snapshot_runtime_columns" in source
    assert 'assert "directional_accuracy" in required' in source
    assert 'assert "mae" in required' in source
    assert 'assert "rmse" in required' in source
    assert "test_audit_detects_legacy_performance_snapshots_missing_columns" in source


def test_phase19_heartbeat_diagnostic_contract_remains_anchored() -> None:
    source = _source("tests/test_run_monitoring_agent.py")

    assert "test_write_heartbeat_includes_cycle_diagnostic_fields" in source
    assert 'assert payload["cycle_prediction_state"] == "none"' in source
    assert 'assert payload["cycle_prediction_reason"] == "missing_model"' in source
    assert 'assert payload["cycle_realization_state"] == "pending"' in source
    assert 'assert "missing_model" in payload["cycle_diagnostic"]' in source
