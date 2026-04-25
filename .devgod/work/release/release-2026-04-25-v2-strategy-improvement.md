# Release Summary

`release-2026-04-25-v2-strategy-improvement`

## Scope

Improve BitBat v2 paper-only strategy quality with a deterministic, explainable filter layer and a
reproducible offline comparison against the previous heuristic baseline.

## Completed

- extracted strategy logic from the v2 runtime into `src/bitbat_v2/strategy.py`
- preserved the prior heuristic as `baseline_v1` for offline comparison only
- shipped `filtered_momentum_v2` as the default v2 paper strategy
- exposed expanded deterministic signal reasons and decision explanations through the runtime and
  API
- added repo-local offline evaluation in `src/bitbat_v2/evaluation.py`
- added `scripts/evaluate_bitbat_v2_strategy.py` for deterministic baseline vs improved comparison
- passed the targeted and full `tests/v2` suites

## Evidence

- targeted tests:
  - `29 passed in 3.43s`
- full v2 tests:
  - `34 passed in 3.72s`
- offline comparison on `data/raw/prices/btcusd_yf_5m.parquet`:
  - baseline `final_equity=9467.84`, `trade_count=4099`, `max_drawdown_pct=-0.0904`
  - improved `final_equity=9592.53`, `trade_count=2221`, `max_drawdown_pct=-0.0863`
  - delta `final_equity_delta=124.69`, `trade_count_delta=-1878`, `realized_pnl_delta=121.11`

## Current posture

- v2 remains paper-only
- live-money trading remains disabled
- no cutover has happened
- legacy services remain intact
- the local comparison shows improved paper-simulation proxy metrics, not proof of live
  profitability
