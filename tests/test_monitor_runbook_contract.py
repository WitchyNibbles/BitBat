from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "monitor-operations-runbook.md"
DOCS_README = ROOT / "docs" / "README.md"
USAGE_GUIDE = ROOT / "docs" / "usage-guide.md"
MONITORING_STRATEGY = ROOT / "docs" / "monitoring_strategy.md"
TESTING_QUALITY = ROOT / "docs" / "testing-quality.md"
SERVICE_TEMPLATE = ROOT / "deployment" / "bitbat-monitor.service"


pytestmark = pytest.mark.structural


def test_monitor_runbook_contains_required_operator_contracts() -> None:
    content = RUNBOOK.read_text(encoding="utf-8")

    assert "--config" in content
    assert "BITBAT_CONFIG" in content
    assert "monitor run-once" in content
    assert "monitor status" in content
    assert "cycle diagnostic" in content
    assert "schema remediation" in content
    assert "make test-release" in content
    assert "bootstrap_monitor_model.py" in content


def test_monitor_service_template_matches_documented_config_wiring() -> None:
    service = SERVICE_TEMPLATE.read_text(encoding="utf-8")
    runbook = RUNBOOK.read_text(encoding="utf-8")
    strategy = MONITORING_STRATEGY.read_text(encoding="utf-8")

    assert "Environment=BITBAT_CONFIG=" in service
    assert "run_monitoring_agent.py --config ${BITBAT_CONFIG}" in service
    assert "deployment/bitbat-monitor.service" in runbook
    assert 'scripts/run_monitoring_agent.py --config "$BITBAT_CONFIG"' in strategy


def test_docs_hub_and_testing_quality_reference_monitor_runbook_contract() -> None:
    docs_readme = DOCS_README.read_text(encoding="utf-8")
    usage = USAGE_GUIDE.read_text(encoding="utf-8")
    testing_quality = TESTING_QUALITY.read_text(encoding="utf-8")

    assert "[Monitor Operations Runbook](./monitor-operations-runbook.md)" in docs_readme
    assert "[Monitor Operations Runbook](./monitor-operations-runbook.md)" in usage
    assert "tests/test_monitor_runbook_contract.py" in testing_quality
    assert "make test-release" in testing_quality
