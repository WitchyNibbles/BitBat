---
phase: 14-baseline-models-and-retraining-cadence
verified: "2026-02-26T06:56:18Z"
status: passed
score: 3/3 must-haves verified
---

# Phase 14: baseline-models-and-retraining-cadence — Verification

## Observable Truths
| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | XGBoost and RandomForest baselines train from one dataset contract and persist comparable artifacts. | verified | `src/bitbat/model/train.py` exposes shared baseline family training; `src/bitbat/model/persist.py` provides family-aware artifact paths/metadata; CLI `model train/cv` supports `--family xgb|random_forest|both` with regression coverage in `tests/model/test_train.py`, `tests/model/test_persist.py`, and `tests/test_cli.py`. |
| 2 | Retraining/backtest windows are configurable and applied consistently across CV and retraining flows. | verified | `src/bitbat/dataset/splits.py` adds deterministic rolling window generation; `src/bitbat/cli.py` supports duration-based rolling windows; `src/bitbat/autonomous/retrainer.py` and `src/bitbat/autonomous/continuous_trainer.py` consume explicit window cadence with metadata; covered by `tests/model/test_walk_forward.py`, `tests/autonomous/test_retrainer.py`, and CLI window tests. |
| 3 | Regime/drift diagnostics are emitted and surfaced per retraining window. | verified | `src/bitbat/model/evaluate.py` adds deterministic window diagnostics + writer; `src/bitbat/model/walk_forward.py` and `src/bitbat/autonomous/continuous_trainer.py` emit diagnostics artifacts/metadata; `src/bitbat/autonomous/drift.py` and monitor CLI report regime/drift score; covered by `tests/model/test_evaluate.py`, `tests/autonomous/test_drift.py`, and `tests/test_cli.py`. |

## Required Artifacts
| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/bitbat/model/train.py` | Shared dual-baseline trainer entrypoints | verified | `fit_baseline`, `fit_xgb`, and `fit_random_forest` support one contract with deterministic seed handling |
| `src/bitbat/model/persist.py` | Family-aware persistence + loading helpers | verified | Stable artifact path helpers and metadata sidecar writes for `xgb`/`random_forest` |
| `src/bitbat/dataset/splits.py` | Rolling train/backtest window generation | verified | `generate_rolling_windows` produces deterministic schedules from duration controls |
| `src/bitbat/autonomous/retrainer.py` | Retraining orchestration consuming configurable windows | verified | CV command generation now uses explicit rolling `--windows` tuples |
| `src/bitbat/model/evaluate.py` | Per-window regime/drift diagnostics | verified | `window_diagnostics` and `write_window_diagnostics` provide deterministic, persisted diagnostics payloads |
| `src/bitbat/autonomous/drift.py` | Drift checks aligned to diagnostic artifacts | verified | Detector metrics include `window_diagnostics`, `regime`, and `drift_score` |

## Requirements Coverage
| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| MODL-01 | complete | None |
| MODL-02 | complete | None |
| MODL-03 | complete | None |

## Validation Evidence
- `poetry run pytest tests/model/test_train.py -q -k "xgb or random or baseline"` -> 4 passed
- `poetry run pytest tests/model/test_persist.py tests/model/test_train.py -q -k "persist or random or xgb"` -> 7 passed
- `poetry run pytest tests/test_cli.py -q -k "model and (train or cv) and (baseline or random or family)"` -> 2 passed, 14 deselected
- `poetry run pytest tests/model/test_walk_forward.py -q -k "window or rolling or fold"` -> 5 passed, 8 deselected
- `poetry run pytest tests/autonomous/test_retrainer.py tests/model/test_walk_forward.py -q -k "retrain or window or rolling"` -> 5 passed, 11 deselected
- `poetry run pytest tests/test_cli.py -q -k "model and (cv or train) and (window or rolling or retrain)"` -> 1 passed, 16 deselected
- `poetry run pytest tests/model/test_evaluate.py -q -k "drift or regime or diagnostic"` -> 2 passed, 2 deselected
- `poetry run pytest tests/model/test_walk_forward.py tests/model/test_evaluate.py -q -k "diagnostic or regime or fold"` -> 5 passed, 12 deselected
- `poetry run pytest tests/autonomous/test_drift.py tests/test_cli.py -q -k "drift or regime or diagnostic"` -> 4 passed, 17 deselected
- `poetry run pytest tests/model/test_train.py tests/model/test_persist.py tests/model/test_walk_forward.py tests/model/test_evaluate.py tests/autonomous/test_retrainer.py tests/autonomous/test_drift.py tests/test_cli.py -q -k "baseline or random or family or window or rolling or retrain or drift or regime or diagnostic"` -> 19 passed, 29 deselected

## Result
Phase 14 goal is achieved. Baseline model families, configurable rolling retraining cadence, and per-window regime/drift diagnostics are implemented with regression coverage and traceability artifacts.
