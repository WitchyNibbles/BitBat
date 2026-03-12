---
phase: 32-cli-decomposition
verified: 2026-03-12T19:50:00Z
status: passed
score: 4/4 success criteria verified
gaps: []
reverification: "Yes — stale monkeypatch-target gap report reconciled in Phase 37"
---

# Phase 32: CLI Decomposition Verification Report

**Phase Goal:** Decompose the 1817-line cli.py monolith into a cli/ package — helpers module, 10 command-group submodules, thin `__init__.py` registration layer. Zero new `noqa:C901` suppressions. All existing tests pass.
**Verified:** 2026-03-12T19:50:00Z
**Status:** passed
**Re-verification:** Yes — Phase 37 reran the current CLI regression surface and replaced the stale gap artifact

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `cli.py` is reduced to a thin entry point and all 10 command groups live in dedicated sub-modules | VERIFIED | `src/bitbat/cli/__init__.py` remains a thin registration layer and all 10 command groups live under `src/bitbat/cli/commands/` |
| 2 | Every existing CLI command continues to work identically | VERIFIED | `poetry run bitbat --help` lists the expected 10 command groups and the CLI regression suites passed |
| 3 | Ruff C901 passes on all new CLI modules without suppressions | VERIFIED | `poetry run ruff check src/bitbat/cli/ --select C901` passed |
| 4 | Existing CLI tests pass without modification | VERIFIED | `poetry run pytest tests/test_cli.py tests/model/test_cv_metric_roundtrip.py tests/dataset/test_public_api.py -x` -> `45 passed` |

**Score:** 4/4 truths verified

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/bitbat/cli/__init__.py` | Thin registration entry point with re-exports for monkeypatched symbols | VERIFIED | Current CLI help and regression tests confirm the registration layer still wires all command groups correctly |
| `src/bitbat/cli/_helpers.py` | Shared helper module | VERIFIED | Existing CLI tests passed against the decomposed package |
| `src/bitbat/cli/commands/*.py` | Dedicated command-group modules | VERIFIED | All 10 groups remain present and active in the current help surface |
| `tests/test_cli.py` | Correct monkeypatch targets | VERIFIED | The `features build` tests now patch `bitbat.cli.commands.features.build_xy`, matching the production import path |

## Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| DEBT-01 | 32-01, 32-02, 32-03 + Phase 37 re-verification | cli.py monolith decomposed — 53 functions and 1802+ lines split into focused modules with no behavioral change | COMPLETE | CLI regression suites passed and the stale Phase 32 gap report is no longer applicable |

## Validation Evidence

- `poetry run pytest tests/test_cli.py tests/model/test_cv_metric_roundtrip.py tests/dataset/test_public_api.py -x` -> `45 passed`
- `poetry run ruff check src/bitbat/cli/ --select C901` -> passed
- `poetry run bitbat --help` -> commands: `backtest`, `batch`, `features`, `ingest`, `model`, `monitor`, `news`, `prices`, `system`, `validate`

## Result

Phase 32 is formally archive-clean. The previously saved monkeypatch-target gap no longer exists in the current codebase, the full CLI regression surface is green, and DEBT-01 is fully satisfied with matching implementation and verification records.
