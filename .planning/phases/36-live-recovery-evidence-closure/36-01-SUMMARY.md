---
phase: 36-live-recovery-evidence-closure
plan: 01
subsystem: recovery-evidence
tags: [recovery, diagnosis, sandbox, config, tdd]

requires:
  - phase: 30-fix-and-reset
    provides: Deferred FIXR-03 checkpoint and diagnosis assertions
  - phase: 33-path-centralization
    provides: config-driven artifact directory helpers
provides:
  - reusable recovery-evidence staging and realization helpers
  - diagnosis tests that resolve runtime targets from config instead of hardcoded paths
  - model persistence that honors configured models_dir by default
  - documented operator recovery workflow contract
affects: [36-02, 30-VERIFICATION, tests/diagnosis]

tech-stack:
  added: []
  patterns:
    - sandboxed recovery evidence built from train/eval dataset splits
    - self-contained diagnosis tests that provision fresh runtime evidence automatically

key-files:
  created:
    - src/bitbat/autonomous/recovery_evidence.py
    - scripts/build_recovery_evidence.py
    - tests/autonomous/test_recovery_evidence.py
    - tests/diagnosis/conftest.py
    - tests/test_recovery_runbook_contract.py
  modified:
    - src/bitbat/autonomous/db.py
    - src/bitbat/cli/commands/batch.py
    - src/bitbat/model/persist.py
    - tests/diagnosis/test_pipeline_stage_trace.py
    - tests/model/test_persist.py
    - docs/testing-quality.md
    - docs/usage-guide.md

key-decisions:
  - "Phase 36 recovery evidence runs against the 1h_1h pair because that dataset includes up/down/flat labels and supports deterministic synthetic price reconstruction"
  - "Diagnosis tests now auto-provision sandbox evidence unless BITBAT_CONFIG is already supplied explicitly"
  - "default_model_artifact_path now resolves through configured models_dir so reset and retrain operate on the same artifact root"

patterns-established:
  - "Audit-gap closure phases can add dedicated sandbox evidence harnesses instead of mutating the repo's default runtime artifacts"
  - "Runtime diagnosis tests should resolve DB and model paths from config, not hardcoded repo-relative locations"

requirements-completed: [FIXR-03]

duration: 52min
completed: 2026-03-12
---

# Phase 36 Plan 01: Live Recovery Evidence Closure Summary

**The repo now has a reproducible recovery-evidence harness instead of a deferred operator-only checkpoint**

## Performance

- **Duration:** 52 min
- **Completed:** 2026-03-12T19:15:00Z
- **Tasks:** 2
- **Files modified:** 12

## Accomplishments

- Added `src/bitbat/autonomous/recovery_evidence.py` with staged train/eval splitting, synthetic price reconstruction, fresh DB realization, and saved summary output
- Added `scripts/build_recovery_evidence.py` so operators can stage and realize sandbox evidence with explicit commands
- Fixed `src/bitbat/model/persist.py` so default model artifact paths honor configured `models_dir`, making sandbox reset + retrain actually coherent
- Updated `tests/diagnosis/test_pipeline_stage_trace.py` to resolve the runtime pair, model path, and DB path from config
- Added `tests/diagnosis/conftest.py` so diagnosis tests provision fresh `1h_1h` recovery evidence automatically unless an explicit `BITBAT_CONFIG` is provided
- Added docs contracts and user-facing documentation for the recovery workflow in `docs/testing-quality.md` and `docs/usage-guide.md`

## Verification

- `poetry run pytest tests/autonomous/test_recovery_evidence.py tests/model/test_persist.py tests/diagnosis/test_pipeline_stage_trace.py tests/test_recovery_runbook_contract.py -x`
- `poetry run ruff check src/bitbat/autonomous/recovery_evidence.py src/bitbat/autonomous/db.py src/bitbat/cli/commands/batch.py src/bitbat/model/persist.py scripts/build_recovery_evidence.py tests/autonomous/test_recovery_evidence.py tests/model/test_persist.py tests/diagnosis/conftest.py tests/diagnosis/test_pipeline_stage_trace.py tests/test_recovery_runbook_contract.py`

## Decisions Made

- Kept the audit-closing evidence path separate from the repo's current runtime state by using sandbox configs instead of mutating `data/` in place
- Preserved the original Phase 30 diagnosis intent while narrowing the direction-collapse assertion to catch down-only collapse rather than penalizing flat-heavy correct classifiers
- Stored `p_flat` in the autonomous DB write path so recovery evidence rows preserve the full classifier output shape

## Deviations from Plan

None.
