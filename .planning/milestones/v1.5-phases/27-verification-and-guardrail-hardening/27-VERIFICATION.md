---
phase: 27-verification-and-guardrail-hardening
verified: 2026-03-07T11:30:00Z
status: passed
score: 6/6 must-haves verified
re_verification: false
---

# Phase 27: Verification and Guardrail Hardening — Verification Report

**Phase Goal:** CI gates prevent recurrence of the architecture drift and complexity creep found during this audit
**Verified:** 2026-03-07T11:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth                                                                                                    | Status     | Evidence                                                                                          |
|-----|----------------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------|
| 1   | `poetry run ruff check src/` exits 0 after C901 is enabled (all 11 pre-existing violations suppressed)  | VERIFIED   | `RUFF_EXIT:0`, "All checks passed!" confirmed live                                                |
| 2   | `poetry run ruff check src/ --select C901` identifies zero unresolved violations                         | VERIFIED   | ruff exits 0; all 11 noqa suppressions present on exact def lines across 9 files                  |
| 3   | `poetry run lint-imports` exits 0 on the current clean codebase                                          | VERIFIED   | "Contracts: 1 kept, 0 broken." confirmed live                                                     |
| 4   | Adding `from bitbat.gui import anything` inside `bitbat.api` causes `lint-imports` to exit non-zero     | VERIFIED   | Smoke-test performed: "Contracts: 0 kept, 1 broken." when temp import added to api/routes/system.py |
| 5   | CI lint job runs `lint-imports` and blocks merges that introduce api->gui imports                        | VERIFIED   | `.github/workflows/ci.yml` line 25-26: "Import architecture contracts" step with `poetry run lint-imports`; test job has `needs: lint` dependency |
| 6   | A function added to any src module with CC > 10 (without noqa) causes `ruff check` to fail              | VERIFIED   | Smoke-test performed: C901 `_test_complexity_gate` flagged as "too complex (12 > 10)", EXIT:1     |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact                                       | Expected                                                              | Status     | Details                                                                   |
|------------------------------------------------|-----------------------------------------------------------------------|------------|---------------------------------------------------------------------------|
| `pyproject.toml`                               | C901 in ruff select, max-complexity=10, importlinter forbidden contract | VERIFIED  | Lines 54, 57-58, 88-95 — all three configurations present                 |
| `.github/workflows/ci.yml`                     | lint-imports step in lint job                                         | VERIFIED   | Lines 25-26: "Import architecture contracts" / `poetry run lint-imports`   |
| `src/bitbat/autonomous/agent.py`               | `# noqa: C901` on run_once def (CC=14)                               | VERIFIED   | Line 182: `def run_once(self) -> dict[str, Any]:  # noqa: C901`           |
| `src/bitbat/autonomous/orchestrator.py`        | `# noqa: C901` on one_click_train def (CC=17)                        | VERIFIED   | Line 29: `def one_click_train(  # noqa: C901`                             |
| `src/bitbat/autonomous/predictor.py`           | `# noqa: C901` on predict_latest def (CC=18)                         | VERIFIED   | Line 109: `def predict_latest(self) -> dict[str, Any]:  # noqa: C901`    |
| `src/bitbat/backtest/metrics.py`               | `# noqa: C901` on summary def (CC=12)                                | VERIFIED   | Line 20: `def summary(  # noqa: C901`                                     |
| `src/bitbat/cli.py`                            | `# noqa: C901` on model_cv def (CC=14)                               | VERIFIED   | Line 544: `def model_cv(  # noqa: C901`                                   |
| `src/bitbat/contracts.py`                      | `# noqa: C901` on ensure_feature_contract def (CC=19)                | VERIFIED   | Line 79: `def ensure_feature_contract(  # noqa: C901`                     |
| `src/bitbat/ingest/news_cryptocompare.py`      | `# noqa: C901` on _fetch_page (CC=14) and fetch (CC=18)              | VERIFIED   | Lines 80, 259: both def lines suppressed                                   |
| `src/bitbat/ingest/news_gdelt.py`              | `# noqa: C901` on _fetch_chunk (CC=11) and fetch (CC=15)             | VERIFIED   | Lines 95, 244: both def lines suppressed                                   |
| `src/bitbat/ingest/prices.py`                  | `# noqa: C901` on fetch_yf def (CC=11)                               | VERIFIED   | Line 94: `def fetch_yf(  # noqa: C901`                                    |

All 11 artifacts verified at all three levels (exists, substantive, wired).

### Key Link Verification

| From                                    | To                               | Via                              | Status   | Details                                                                           |
|-----------------------------------------|----------------------------------|----------------------------------|----------|-----------------------------------------------------------------------------------|
| `pyproject.toml [tool.ruff.lint] select`| C901 rule                        | `ruff check src/`                | WIRED    | "C901" present in select list; `ruff check src/ tests/` exits 0                   |
| `pyproject.toml [tool.importlinter]`    | bitbat.api -> bitbat.gui contract| `lint-imports`                   | WIRED    | `forbidden_modules = ["bitbat.gui"]` present; lint-imports exits 0 on clean code  |
| `.github/workflows/ci.yml` lint job     | `poetry run lint-imports`        | CI step after ruff steps         | WIRED    | Step "Import architecture contracts" at lines 25-26; `test` job has `needs: lint` ensuring failures block |

### Requirements Coverage

| Requirement | Source Plan | Description                                                           | Status    | Evidence                                                                      |
|-------------|-------------|-----------------------------------------------------------------------|-----------|-------------------------------------------------------------------------------|
| ARCH-05     | 27-01-PLAN  | import-linter contracts added to CI preventing future cross-layer import drift | SATISFIED | `[tool.importlinter]` forbidden contract in pyproject.toml; CI step `poetry run lint-imports`; smoke-test confirms gate blocks api->gui imports |
| ARCH-06     | 27-01-PLAN  | ruff C901 complexity gate added to CI preventing new high-complexity functions | SATISFIED | `C901` in ruff select, `max-complexity=10` in mccabe config; 11 suppressions on pre-existing violations; smoke-test confirms gate blocks CC>10 |

No orphaned requirements — both ARCH-05 and ARCH-06 are fully satisfied by 27-01-PLAN. REQUIREMENTS.md marks both as Complete at Phase 27.

### Anti-Patterns Found

None. No TODO/FIXME/placeholder comments or stub implementations were found in the phase-modified files. The noqa suppressions are intentional, documented, and required.

### Human Verification Required

None. All truths are verifiable programmatically. Both gate behaviors (C901 blocking and import-linter blocking) were confirmed via live smoke-tests during this verification session. The CI job ordering (`test` needs `lint`) ensures import-linter failures block the test job from running.

### Summary

Phase 27 fully achieved its goal. Both CI guardrails are active and correctly configured:

1. **ARCH-06 (C901 gate):** `ruff check src/ tests/` exits 0 with C901 enabled at max-complexity=10. All 11 pre-existing violations are suppressed with `# noqa: C901` on the exact def lines. A new function with CC > 10 without a noqa comment causes ruff to exit non-zero — confirmed by smoke-test.

2. **ARCH-05 (import-linter gate):** The `forbidden` contract `bitbat.api -> bitbat.gui` is defined in pyproject.toml and enforced by `poetry run lint-imports` (exits 0, "Contracts: 1 kept, 0 broken"). A transitive violation in `orchestrator.py` (missed in Phase 26) was fixed during this phase. Adding an api->gui import causes lint-imports to exit non-zero — confirmed by smoke-test.

3. **CI integration:** The lint job in `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`, and `lint-imports` in sequence. The test job has `needs: lint`, so any lint-imports failure blocks the test job and prevents merge.

Both ARCH-05 and ARCH-06 are durably prevented from recurrence at the CI gate level.

---

_Verified: 2026-03-07T11:30:00Z_
_Verifier: Claude (gsd-verifier)_
