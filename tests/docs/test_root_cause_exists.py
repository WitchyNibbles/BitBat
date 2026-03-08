"""Structural tests for ROOT_CAUSE.md — Phase 29 DIAG-02.

These tests enforce that ROOT_CAUSE.md is committed before any Phase 30
fix code. test_root_cause_md_exists is in RED state until Plan 29-02
Task 1 creates ROOT_CAUSE.md.
"""

from pathlib import Path

import pytest

ROOT_CAUSE_PATH = Path("ROOT_CAUSE.md")

REQUIRED_SECTIONS = [
    "## Observed Symptom",
    "## Pipeline Stage Trace",
    "### Stage",
    "## Summary Table",
    "Phase 30",
]


def test_root_cause_md_exists():
    """DIAG-02: ROOT_CAUSE.md must be committed before any fix code.

    This test is RED until Plan 29-02 Task 1 creates ROOT_CAUSE.md.
    """
    assert ROOT_CAUSE_PATH.exists(), (
        "ROOT_CAUSE.md not found at repo root. "
        "This document must be committed before any Phase 30 fix code. "
        "See .planning/phases/29-diagnosis/29-02-PLAN.md Task 1."
    )


def test_root_cause_has_required_sections():
    """DIAG-02: ROOT_CAUSE.md must contain all required sections."""
    if not ROOT_CAUSE_PATH.exists():
        pytest.skip("ROOT_CAUSE.md not yet created — run 29-01 first")
    content = ROOT_CAUSE_PATH.read_text()
    for section in REQUIRED_SECTIONS:
        assert section in content, f"ROOT_CAUSE.md is missing required section: '{section}'"
