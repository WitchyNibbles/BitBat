---
phase: 35-xgboost-fix
verified: "2026-03-12T16:47:26Z"
status: passed
score: 4/4 must-haves verified
---

# Phase 35: xgboost-fix — Verification

## Observable Truths
| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Walk-forward CV and nested optimization now switch XGBoost to `multi:softprob` whenever the target is the direction label. | verified | [walk_forward.py](/home/eimi/projects/ai-btc-predictor/src/bitbat/model/walk_forward.py) and [optimize.py](/home/eimi/projects/ai-btc-predictor/src/bitbat/model/optimize.py) detect label targets, encode them through `DIRECTION_CLASSES`, and the classification-mode tests passed. |
| 2 | Classification-mode predictions are valid probability outputs and the scoring path records PR-AUC/log-loss evidence. | verified | [evaluate.py](/home/eimi/projects/ai-btc-predictor/src/bitbat/model/evaluate.py) computes multiclass probability metrics with stable label ordering, [test_walk_forward.py](/home/eimi/projects/ai-btc-predictor/tests/model/test_walk_forward.py) verifies probability rows sum to 1.0, and the deterministic guardrail test now locks `mean_pr_auc >= 0.7`. |
| 3 | `model cv` and `model optimize` now use labels for the XGBoost path and surface classification-aware summaries to operators. | verified | [model.py](/home/eimi/projects/ai-btc-predictor/src/bitbat/cli/commands/model.py) now loads labels for XGBoost CV/optimization, [test_cli.py](/home/eimi/projects/ai-btc-predictor/tests/test_cli.py) verifies PR-AUC-aware output and JSON payloads, and [test_phase5_complete.py](/home/eimi/projects/ai-btc-predictor/tests/model/test_phase5_complete.py) verifies classification-mode summary compatibility. |
| 4 | Existing training/inference compatibility remains intact for the saved XGBoost artifact path and numeric-target regression fixtures. | verified | `fit_xgb()` / inference tests still passed, and the CLI helper fallback in [_helpers.py](/home/eimi/projects/ai-btc-predictor/src/bitbat/cli/_helpers.py) keeps 1-D XGBoost stub predictions working in older tests. |

## Required Artifacts
| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/bitbat/model/walk_forward.py` | Classification-aware walk-forward evaluation | verified | Emits probability columns, PR-AUC/log-loss summaries, and keeps regression fallback |
| `src/bitbat/model/optimize.py` | Classification-aware nested optimization | verified | Uses PR-AUC-based minimized score for label targets and records objective mode in provenance |
| `src/bitbat/model/evaluate.py` | Stable multiclass probability metrics | verified | Adds `classification_probability_metrics()` with normalized probabilities and consistent label encoding |
| `src/bitbat/cli/commands/model.py` and `src/bitbat/cli/_helpers.py` | Label-driven CLI CV/optimization flow | verified | XGBoost uses `label`, RandomForest keeps `r_forward`, and summaries emit PR-AUC for classification mode |
| `tests/model/test_walk_forward.py`, `tests/model/test_optimize.py`, `tests/test_cli.py`, `tests/model/test_phase5_complete.py` | Regression coverage for Phase 35 behavior | verified | All targeted suites passed |

## Requirements Coverage
| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| DEBT-04 | complete | None |

## Validation Evidence
- `poetry run pytest tests/model/test_train.py tests/model/test_infer.py tests/model/test_walk_forward.py tests/model/test_optimize.py tests/model/test_phase5_complete.py tests/test_cli.py -x` -> 92 passed
- `poetry run ruff check src/bitbat/model src/bitbat/cli/_helpers.py src/bitbat/cli/commands/model.py tests/model tests/test_cli.py` -> passed

## Result
Phase 35 goal is achieved. The XGBoost evaluation and selection stack now uses the classification objective for label targets, CLI CV/optimization surfaces expose PR-AUC-aware evidence, deterministic walk-forward coverage locks the `>= 0.7` PR-AUC guardrail on separable labeled data, and the existing training/inference compatibility suites remain green.
