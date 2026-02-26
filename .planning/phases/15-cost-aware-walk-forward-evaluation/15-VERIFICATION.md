---
phase: 15-cost-aware-walk-forward-evaluation
verified: "2026-02-26T07:27:37Z"
status: passed
score: 3/3 must-haves verified
---

# Phase 15: cost-aware-walk-forward-evaluation — Verification

## Observable Truths
| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Walk-forward evaluation enforces time ordering and supports purge/embargo controls for overlap-safe horizons. | verified | `src/bitbat/dataset/splits.py` now supports `purge_bars`, `embargo_bars`, and horizon-derived purge sizing; CLI `model cv` exposes these controls and routes them into split generation in `src/bitbat/cli.py`; fold metadata persists leakage controls via `src/bitbat/model/walk_forward.py`; covered by `tests/dataset/test_splits.py`, `tests/model/test_walk_forward.py`, and CLI regression tests. |
| 2 | Evaluation metrics are cost-aware and include explicit fee/slippage net-vs-gross attribution. | verified | `src/bitbat/backtest/engine.py` emits `fee_costs`/`slippage_costs`; `src/bitbat/backtest/metrics.py` and `src/bitbat/model/walk_forward.py` aggregate net/gross return and fee/slippage totals; `src/bitbat/cli.py` exposes backtest fee/slippage controls and net-aware report output; covered by `tests/backtest/test_engine.py`, `tests/backtest/test_metrics.py`, `tests/model/test_walk_forward.py`, and CLI cost tests. |
| 3 | Candidate reports and champion selection are explicit, deterministic, and persisted for CLI/autonomous flows. | verified | `src/bitbat/model/evaluate.py` provides deterministic `build_candidate_report` and `select_champion_report`; `src/bitbat/cli.py` persists `candidate_reports` and `champion_decision` in `metrics/cv_summary.json`; `src/bitbat/autonomous/retrainer.py` consumes champion decisions and blocks deployment when `promote_candidate` is false; covered by `tests/model/test_evaluate.py`, `tests/model/test_walk_forward.py`, `tests/autonomous/test_retrainer.py`, and CLI champion-report tests. |

## Required Artifacts
| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/bitbat/dataset/splits.py` | Purge/embargo-capable split builder | verified | Deterministic bar-based purge/embargo controls with horizon-derived purge fallback |
| `src/bitbat/backtest/engine.py` | Fee/slippage-aware trade cost model | verified | Separate fee and slippage columns plus total cost compatibility |
| `src/bitbat/backtest/metrics.py` | Net/gross/cost-attribution summary outputs | verified | Includes `net_sharpe`, `gross_sharpe`, `total_fee_costs`, `total_slippage_costs`, `gross_return` |
| `src/bitbat/model/evaluate.py` | Candidate report and champion rule helpers | verified | Deterministic multi-metric candidate reports and rule-based champion selection |
| `src/bitbat/cli.py` | Persisted candidate/champion outputs in CV flow | verified | Writes `candidate_reports` + `champion_decision` and prints champion outcome |
| `src/bitbat/autonomous/retrainer.py` | Champion-aware deployment gating | verified | Rejects deployment when champion decision disallows promotion |

## Requirements Coverage
| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| EVAL-01 | complete | None |
| EVAL-02 | complete | None |
| EVAL-03 | complete | None |

## Validation Evidence
- `poetry run pytest tests/dataset/test_splits.py -q -k "walk_forward and (embargo or purge or leakage)"` -> 4 passed
- `poetry run pytest tests/model/test_walk_forward.py -q -k "walk_forward or fold or embargo or purge"` -> 14 passed
- `poetry run pytest tests/test_cli.py -q -k "model and cv and (purge or embargo or walk_forward)"` -> 1 passed, 18 deselected
- `poetry run pytest tests/backtest/test_engine.py -q -k "cost or slippage or net or gross"` -> 4 passed, 3 deselected
- `poetry run pytest tests/backtest/test_metrics.py tests/model/test_walk_forward.py -q -k "net or gross or cost or sharpe"` -> 2 passed, 15 deselected
- `poetry run pytest tests/test_cli.py -q -k "backtest and (cost or slippage or net or gross)"` -> 1 passed, 19 deselected
- `poetry run pytest tests/model/test_evaluate.py -q -k "candidate or champion or report or metric"` -> 4 passed, 2 deselected
- `poetry run pytest tests/model/test_walk_forward.py tests/autonomous/test_retrainer.py -q -k "champion or candidate or report or retrain"` -> 5 passed, 15 deselected
- `poetry run pytest tests/test_cli.py -q -k "model and (report or champion or candidate or evaluation)"` -> 1 passed, 20 deselected
- `poetry run pytest tests/dataset/test_splits.py tests/backtest/test_engine.py tests/backtest/test_metrics.py tests/model/test_evaluate.py tests/model/test_walk_forward.py tests/autonomous/test_retrainer.py tests/test_cli.py -q -k "walk_forward or purge or embargo or leakage or cost or slippage or net or gross or candidate or champion or report or metric or retrain"` -> 38 passed, 22 deselected

## Result
Phase 15 goal is achieved. Leakage-safe walk-forward controls, explicit cost-attribution metrics, and deterministic candidate/champion reporting are implemented with CLI and autonomous flow alignment.
