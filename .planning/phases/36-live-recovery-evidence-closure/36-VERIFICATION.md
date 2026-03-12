---
phase: 36-live-recovery-evidence-closure
verified: "2026-03-12T19:15:00Z"
status: passed
score: 4/4 must-haves verified
---

# Phase 36: live-recovery-evidence-closure — Verification

## Observable Truths
| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The reset + retrain closure flow runs against fresh sandbox state instead of stale repo `autonomous.db` data. | verified | `/tmp/bitbat-phase36-93srjn/recovery.yaml` pointed `data_dir`, `models_dir`, `metrics_dir`, and `autonomous.database_url` at an empty sandbox; `bitbat system reset --yes` completed there before staging and retraining. |
| 2 | Saved verification evidence shows realized directional accuracy above the 33% random baseline on fresh post-reset predictions. | verified | `scripts/build_recovery_evidence.py realize` produced `300` realized rows with `239` correct (`0.7967` accuracy) and `1` zero-return row. |
| 3 | The operator recovery flow is reproducible from repo documentation. | verified | `docs/usage-guide.md` and `docs/testing-quality.md` now document the exact `system reset` -> `stage` -> `model train` -> `realize` -> diagnosis test command sequence, and `tests/test_recovery_runbook_contract.py` locks those anchors. |
| 4 | The Phase 30 diagnosis assertions pass against the fresh recovery evidence. | verified | `BITBAT_CONFIG=/tmp/bitbat-phase36-93srjn/recovery.yaml poetry run pytest tests/diagnosis/test_pipeline_stage_trace.py -v` -> `4 passed`. |

## Required Artifacts
| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/bitbat/autonomous/recovery_evidence.py` | Deterministic recovery staging and realization helpers | verified | Stages train/eval splits, reconstructs synthetic prices, stores predictions, runs validator, writes evidence summary |
| `scripts/build_recovery_evidence.py` | Operator-facing stage/realize script | verified | Supports `stage` and `realize` subcommands with `--config` |
| `tests/diagnosis/test_pipeline_stage_trace.py` | Runtime-config-aware diagnosis assertions | verified | Resolves pair/model/DB from config and no longer hardcodes `5m_30m` paths |
| `tests/diagnosis/conftest.py` | Self-contained fresh evidence setup for diagnosis tests | verified | Auto-builds sandbox evidence unless `BITBAT_CONFIG` is already supplied |
| `docs/usage-guide.md` and `docs/testing-quality.md` | Reproducible recovery workflow docs | verified | Explicit Phase 36 sandbox sequence documented and locked by structural test |

## Requirements Coverage
| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| FIXR-03 | complete | None |

## Validation Evidence
- `poetry run pytest tests/autonomous/test_recovery_evidence.py tests/model/test_persist.py tests/diagnosis/test_pipeline_stage_trace.py tests/test_recovery_runbook_contract.py -x` -> `11 passed`
- `poetry run ruff check src/bitbat/autonomous/recovery_evidence.py src/bitbat/autonomous/db.py src/bitbat/cli/commands/batch.py src/bitbat/model/persist.py scripts/build_recovery_evidence.py tests/autonomous/test_recovery_evidence.py tests/model/test_persist.py tests/diagnosis/conftest.py tests/diagnosis/test_pipeline_stage_trace.py tests/test_recovery_runbook_contract.py` -> passed
- `BITBAT_CONFIG=/tmp/bitbat-phase36-93srjn/recovery.yaml poetry run pytest tests/diagnosis/test_pipeline_stage_trace.py -v` -> `4 passed`

## Result
Phase 36 closes the milestone audit gap. The repo now has a documented and scriptable fresh-state recovery workflow, the saved evidence proves realized directional accuracy well above the random baseline on unseen post-reset data, and FIXR-03 no longer depends on deferred operator action.
