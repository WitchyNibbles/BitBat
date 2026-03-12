from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
USAGE_GUIDE = ROOT / "docs" / "usage-guide.md"
TESTING_QUALITY = ROOT / "docs" / "testing-quality.md"
RECOVERY_SCRIPT = ROOT / "scripts" / "build_recovery_evidence.py"

pytestmark = pytest.mark.structural


def test_recovery_docs_describe_reset_retrain_diagnosis_flow() -> None:
    usage = USAGE_GUIDE.read_text(encoding="utf-8")
    quality = TESTING_QUALITY.read_text(encoding="utf-8")

    assert "system reset --yes" in usage
    assert "model train --freq 1h --horizon 1h" in usage
    assert "scripts/build_recovery_evidence.py stage" in usage
    assert "scripts/build_recovery_evidence.py realize" in usage
    assert "tests/diagnosis/test_pipeline_stage_trace.py" in usage
    assert "BITBAT_CONFIG" in usage
    assert "recovery evidence" in quality.lower()


def test_recovery_script_exists() -> None:
    assert RECOVERY_SCRIPT.exists()
