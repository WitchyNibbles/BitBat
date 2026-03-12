---
phase: 25
slug: critical-correctness-remediation
status: complete
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-07
reconstructed_from: artifacts
---

# Phase 25 — Validation Strategy

> Reconstructed from phase artifacts (State B — VALIDATION.md not created during execution).
> All tests confirmed passing: 23/23 tests green as of 2026-03-07.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.2 |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `poetry run pytest tests/autonomous/test_retrainer_cli_contract.py tests/model/test_cv_metric_roundtrip.py tests/model/test_regression_metrics_purity.py tests/model/test_assert_guards.py tests/features/test_leakage.py tests/api/test_api_config_defaults.py tests/features/test_obv_leakage_assessment.py tests/features/test_obv_fold_aware.py -v` |
| **Full suite command** | `poetry run pytest -x -q` |
| **Estimated runtime** | ~11 seconds (phase tests only); ~108 seconds (full suite) |

---

## Sampling Rate

- **After every task commit:** Run phase-specific test files above
- **After every plan wave:** Run full suite (`poetry run pytest -x -q`)
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~11 seconds (phase tests only)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 25-01-01 | 01 | 1 | CORR-01 | behavioral | `poetry run pytest tests/autonomous/test_retrainer_cli_contract.py -v` | ✅ | ✅ green |
| 25-01-02 | 01 | 1 | CORR-02 | behavioral | `poetry run pytest tests/model/test_cv_metric_roundtrip.py -v` | ✅ | ✅ green |
| 25-02-01 | 02 | 1 | CORR-03 | behavioral | `poetry run pytest tests/model/test_regression_metrics_purity.py -v` | ✅ | ✅ green |
| 25-02-02 | 02 | 1 | CORR-04 | behavioral | `poetry run pytest tests/model/test_assert_guards.py -v` | ✅ | ✅ green |
| 25-03-01 | 03 | 1 | CORR-05 | behavioral | `poetry run pytest tests/features/test_leakage.py -v` | ✅ | ✅ green |
| 25-03-02 | 03 | 1 | CORR-06 | behavioral | `poetry run pytest tests/api/test_api_config_defaults.py -v` | ✅ | ✅ green |
| 25-04-01 | 04 | 1 | LEAK-01 | behavioral | `poetry run pytest tests/features/test_obv_leakage_assessment.py -v` | ✅ | ✅ green |
| 25-04-02 | 04 | 1 | LEAK-02 | behavioral | `poetry run pytest tests/features/test_obv_fold_aware.py -v` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Gap Analysis

No gaps found. All 8 requirements have dedicated automated test files. All 23 tests pass.

| Requirement | Test File | Tests | Status |
|-------------|-----------|-------|--------|
| CORR-01 | `tests/autonomous/test_retrainer_cli_contract.py` | 2 | COVERED ✅ |
| CORR-02 | `tests/model/test_cv_metric_roundtrip.py` | 2 | COVERED ✅ |
| CORR-03 | `tests/model/test_regression_metrics_purity.py` | 3 | COVERED ✅ |
| CORR-04 | `tests/model/test_assert_guards.py` | 3 | COVERED ✅ |
| CORR-05 | `tests/features/test_leakage.py` | 3 | COVERED ✅ |
| CORR-06 | `tests/api/test_api_config_defaults.py` | 3 | COVERED ✅ |
| LEAK-01 | `tests/features/test_obv_leakage_assessment.py` | 3 | COVERED ✅ |
| LEAK-02 | `tests/features/test_obv_fold_aware.py` | 4 | COVERED ✅ |

**Total: 23/23 tests passing**

---

## Wave 0 Requirements

None — existing infrastructure covered all phase requirements. pytest and all required packages were already installed.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Verify `--tau` visually absent from retrainer subprocess | CORR-01 | Code inspection cross-check | `grep -n '"--tau"' src/bitbat/autonomous/retrainer.py` — should return nothing |
| Verify `average_balanced_accuracy` removed from CLI writer | CORR-02 | Code inspection cross-check | `grep -n 'average_balanced_accuracy' src/bitbat/cli.py` — should return nothing |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify commands
- [x] Sampling continuity: every task has automated verification
- [x] Wave 0 not needed — infrastructure already present
- [x] No watch-mode flags
- [x] Feedback latency < 15s (phase tests: ~11s)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-03-07 (reconstructed from artifacts — 23/23 tests passing)

---

## Validation Audit 2026-03-07

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Tests confirmed passing | 23 |
| Nyquist compliant | true |

## Validation Audit 2026-03-07 (re-audit via /gsd:validate-phase 25)

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
| Tests confirmed passing | 23 |
| Nyquist compliant | true |
| Notes | All 23 phase-25 tests confirmed green. test_no_hardcoded_1h_4h_in_api_routes passes (metrics.py fixed in quick task 1 — commit c03721f). |
