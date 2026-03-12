---
phase: 36-live-recovery-evidence-closure
plan: 02
subsystem: verification
tags: [verification, sandbox, reset, retrain, diagnosis]

requires:
  - phase: 36-01
    provides: Recovery evidence harness and config-aware diagnosis path
provides:
  - saved passed FIXR-03 evidence from a fresh reset + retrain sandbox run
  - updated Phase 30 verification artifact with the resolved status
  - milestone traceability updates showing Phase 36 complete
affects: [30-VERIFICATION, ROADMAP, STATE, REQUIREMENTS, PROJECT]

requirements-completed: [FIXR-03]

duration: 18min
completed: 2026-03-12
---

# Phase 36 Plan 02: Live Recovery Evidence Closure Summary

**FIXR-03 is no longer deferred: the reset + retrain path now has saved fresh evidence**

## Sandbox Run

- Sandbox config: `/tmp/bitbat-phase36-93srjn/recovery.yaml`
- Pair: `1h_1h`
- `system reset`: passed (`Nothing to delete — already clean.` on a new sandbox)
- `stage`: wrote `1056` training rows and `300` evaluation rows
- `model train`: wrote `/tmp/bitbat-phase36-93srjn/models/1h_1h/xgb.json`
- `realize`: wrote fresh `prediction_outcomes` and `/tmp/bitbat-phase36-93srjn/metrics/recovery_evidence.json`

## Key Evidence

- Realized predictions: `300`
- Correct predictions: `239`
- Realized accuracy: `0.7967`
- Zero-return rows: `1`
- Predicted direction counts: `flat=283`, `down=15`, `up=2`
- Diagnosis suite under the sandbox config: `4 passed`

## Verification

- `poetry run bitbat --config /tmp/bitbat-phase36-93srjn/recovery.yaml system reset --yes`
- `poetry run python scripts/build_recovery_evidence.py stage --config /tmp/bitbat-phase36-93srjn/recovery.yaml --source-dataset data/features/1h_1h/dataset.parquet --evaluation-rows 300`
- `poetry run bitbat --config /tmp/bitbat-phase36-93srjn/recovery.yaml model train --freq 1h --horizon 1h`
- `poetry run python scripts/build_recovery_evidence.py realize --config /tmp/bitbat-phase36-93srjn/recovery.yaml`
- `BITBAT_CONFIG=/tmp/bitbat-phase36-93srjn/recovery.yaml poetry run pytest tests/diagnosis/test_pipeline_stage_trace.py -v`

## Decisions Made

- The closure evidence is saved in Phase 36 rather than trying to reinterpret the original Phase 30 report retroactively
- Phase 30's verification artifact is re-marked as passed with explicit reference to the new Phase 36 sandbox evidence

## Deviations from Plan

None.
