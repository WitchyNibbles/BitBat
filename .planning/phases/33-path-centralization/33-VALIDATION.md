---
phase: 33
slug: path-centralization
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-12
---

# Phase 33 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `poetry run pytest tests/config/ tests/model/test_train.py tests/backtest/test_metrics.py -x` |
| **Full suite command** | `poetry run pytest` |
| **Estimated runtime** | ~30 seconds (quick), ~130 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `poetry run pytest tests/config/ tests/model/test_train.py tests/backtest/test_metrics.py -x`
- **After every plan wave:** Run `poetry run pytest`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 33-01-01 | 01 | 1 | DEBT-02 | unit+structural | `poetry run pytest tests/config/test_path_resolution.py -x` | ❌ Wave 0 | ⬜ pending |
| 33-01-02 | 01 | 1 | DEBT-02 | unit | `poetry run pytest tests/config/test_path_resolution.py -x` | ❌ Wave 0 | ⬜ pending |
| 33-02-01 | 02 | 2 | DEBT-02 | behavioral | `poetry run pytest tests/config/ tests/model/test_train.py -x` | ✅ | ⬜ pending |
| 33-02-02 | 02 | 2 | DEBT-02 | structural+behavioral | `poetry run pytest tests/config/test_path_resolution.py tests/backtest/test_metrics.py -x` | ✅ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/config/test_path_resolution.py` — stubs for DEBT-02 (structural grep tests + config redirect unit tests); created in Plan 01 Task 1

*Wave 0 is satisfied by Plan 01 Task 1 (TDD red phase). The test file is the Wave 0 deliverable.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
