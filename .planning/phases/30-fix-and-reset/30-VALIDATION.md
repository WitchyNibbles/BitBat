---
phase: 30
slug: fix-and-reset
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 30 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `poetry run pytest tests/diagnosis/ tests/model/test_train.py tests/model/test_infer.py -x` |
| **Full suite command** | `poetry run pytest` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `poetry run pytest tests/model/test_train.py tests/model/test_infer.py -x`
- **After every plan wave:** Run `poetry run pytest`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 30-01-01 | 01 | 1 | FIXR-01 | unit | `poetry run pytest tests/model/test_train.py::test_fit_xgb_uses_classification_objective tests/model/test_train.py::test_fit_xgb_classification_output_shape -x` | ❌ Wave 0 | ⬜ pending |
| 30-01-02 | 01 | 1 | FIXR-01 | unit | `poetry run pytest tests/model/test_infer.py::test_predict_bar_returns_three_classes -x` | ❌ Wave 0 | ⬜ pending |
| 30-01-03 | 01 | 1 | FIXR-01 | unit | `poetry run pytest tests/autonomous/test_validator.py::test_validator_uses_constructor_tau -x` | ❌ Wave 0 | ⬜ pending |
| 30-02-01 | 02 | 2 | FIXR-01 | integration | `poetry run pytest tests/diagnosis/test_pipeline_stage_trace.py -x` | ✅ (invert) | ⬜ pending |
| 30-02-02 | 02 | 2 | FIXR-02 | unit | `poetry run pytest tests/test_cli.py::test_system_reset_command tests/test_cli.py::test_system_reset_handles_missing_dirs tests/test_cli.py::test_system_reset_prompts_without_yes_flag -x` | ❌ Wave 0 | ⬜ pending |
| 30-03-01 | 03 | 3 | FIXR-03 | integration | `poetry run pytest` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/model/test_train.py` — add `test_fit_xgb_uses_classification_objective` and `test_fit_xgb_classification_output_shape`
- [ ] `tests/model/test_infer.py` — add `test_predict_bar_returns_three_classes`
- [ ] `tests/autonomous/test_validator.py` — add `test_validator_uses_constructor_tau` (create file if missing)
- [ ] `tests/test_cli.py` — add `test_system_reset_command`, `test_system_reset_handles_missing_dirs`, `test_system_reset_prompts_without_yes_flag` (using `tmp_path` fixtures)
- [ ] `tests/diagnosis/test_pipeline_stage_trace.py` — invert all 4 existing assertions from bug-present to bug-fixed

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Realized accuracy >33% after live reset + retrain | FIXR-03 | Requires live data ingestion and horizon passage | Operator runs `bitbat system reset --yes && poetry run bitbat prices ingest && poetry run bitbat model train && wait for horizon` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
