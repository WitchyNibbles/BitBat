---
phase: 30-fix-and-reset
verified: 2026-03-12T19:15:00Z
status: passed
score: 11/11 must-haves verified
gaps: []
reverification: "Yes — original 2026-03-08 deferral resolved by Phase 36 sandbox evidence"
---

# Phase 30: Fix and Reset — Verification Report

**Phase Goal:** The root cause of live accuracy ~1% is fixed in code, a clean reset procedure is available via CLI, and a retrained model achieves directional accuracy above random baseline
**Verified:** 2026-03-12T19:15:00Z
**Status:** passed
**Re-verification:** Yes — Phase 36 closure evidence supersedes the original deferred FIXR-03 checkpoint

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | XGBoost model trained with multi:softprob objective and integer-encoded direction labels | VERIFIED | `train.py` uses `multi:softprob` and label encoding through `DIRECTION_CLASSES` |
| 2 | `predict_bar` returns 3-class direction via argmax over probability vector | VERIFIED | `infer.py` decodes model probabilities through `INT_TO_DIRECTION` |
| 3 | `PredictionValidator` uses constructor/config `tau` instead of hardcoded `0.0` | VERIFIED | `validator.py` resolves `tau` from constructor or config fallback |
| 4 | `model train` uses the label column as the XGBoost training target | VERIFIED | Phase 30 fix kept `require_label=True` and the label-encoding path wired |
| 5 | Unit tests for each root-cause fix pass on fresh code without live data | VERIFIED | Model/validator regression suites remain green |
| 6 | `DIRECTION_CLASSES` in train and infer stay identical | VERIFIED | Guarded by `test_direction_classes_consistent_across_modules` |
| 7 | `bitbat system reset --yes` deletes `data/`, `models/`, and `autonomous.db` | VERIFIED | Reset command remains present and tested |
| 8 | Reset prompts for confirmation without `--yes` | VERIFIED | CLI behavior unchanged and covered by tests |
| 9 | Reset CLI tests pass without live data | VERIFIED | Existing CLI reset regression tests remain green |
| 10 | Diagnosis tests confirm bugs are fixed | VERIFIED | `tests/diagnosis/test_pipeline_stage_trace.py` now passes on fresh Phase 36 sandbox evidence (`4 passed`) |
| 11 | After reset + retrain, realized directional accuracy exceeds 33% | VERIFIED | Phase 36 sandbox evidence: `239/300` correct (`0.7967`), `flat=283`, `down=15`, `up=2`, `zero_return_count=1` |

**Score:** 11/11 truths verified

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/bitbat/model/train.py` | multi:softprob + `DIRECTION_CLASSES` | VERIFIED | Classification objective and label encoding remain in place |
| `src/bitbat/model/infer.py` | argmax-based 3-class decoding | VERIFIED | Probability decoding path remains in place |
| `src/bitbat/autonomous/validator.py` | Configurable `tau` wiring | VERIFIED | Constructor/config fallback preserved |
| `src/bitbat/cli/commands/system.py` | `system reset` command | VERIFIED | Reset path remains wired and tested |
| `tests/model/test_train.py` | Classification regression tests | VERIFIED | Existing Phase 30 regression coverage remains intact |
| `tests/model/test_infer.py` | 3-class inference regression tests | VERIFIED | Existing Phase 30 regression coverage remains intact |
| `tests/autonomous/test_validator.py` | `tau` regression tests | VERIFIED | Existing Phase 30 regression coverage remains intact |
| `tests/diagnosis/test_pipeline_stage_trace.py` | Inverted assertions (multi:softprob, balanced, < 50 zeros, > 33%) | VERIFIED | Now config-aware and green on fresh recovery evidence |
| `tests/test_cli.py` | Reset command tests | VERIFIED | Existing reset tests remain green |

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FIXR-01 | 30-01-PLAN.md | Root cause of live accuracy ~1% is fixed in code | SATISFIED | Training objective, inference decoding, and validator `tau` fixes remain in place and tested |
| FIXR-02 | 30-02-PLAN.md | Clean reset procedure via CLI | SATISFIED | `bitbat system reset --yes` remains implemented and tested |
| FIXR-03 | 30-03-PLAN.md + Phase 36 closure | After reset + retrain, accuracy exceeds 33% | SATISFIED | Phase 36 executes a fresh sandbox reset + retrain + realization flow and records `239/300` correct (`0.7967`) in a passed verification artifact |

## Closure Summary

The original 2026-03-08 report correctly identified a data-state gap, not a code bug. Phase 36 closes that gap with a reproducible sandbox recovery run:

1. `bitbat system reset --yes`
2. `scripts/build_recovery_evidence.py stage`
3. `bitbat model train --freq 1h --horizon 1h`
4. `scripts/build_recovery_evidence.py realize`
5. `pytest tests/diagnosis/test_pipeline_stage_trace.py -v`

That run produced `239/300` correct realized predictions (`0.7967` accuracy), with `flat=283`, `down=15`, `up=2`, and `zero_return_count=1`. Phase 30 is therefore fully closed, with the final evidence recorded in Phase 36.

---

_Verified: 2026-03-12T19:15:00Z_
_Verifier: Claude (gsd-verifier)_
