# Task ID

`task-12-deterministic-strategy-improvement`

## Reviewer role

`reviewer + security_reviewer + qa_engineer`

## Review state

`passed`

## Severity

`medium`

## Findings

- extracted deterministic v2 strategy logic into `src/bitbat_v2/strategy.py`
- kept the legacy heuristic as named baseline `baseline_v1` for offline comparison only
- switched the runtime default to deterministic `filtered_momentum_v2` with explainable reasons and
  filter gates
- added config controls for sell threshold, trend lookbacks, max range ratio, and minimum body
  strength
- added reproducible local evaluation in `src/bitbat_v2/evaluation.py` and
  `scripts/evaluate_bitbat_v2_strategy.py`
- runtime, API, duplicate-candle suppression, pause semantics, stale-data guard, and paper-only
  execution remain intact
- no live-money or cutover behavior was introduced

## Verification evidence

- `poetry run pytest tests/v2/test_strategy.py tests/v2/test_evaluation.py tests/v2/test_runtime.py tests/v2/test_api.py -q`
  - passed: `29 passed in 3.43s`
- `poetry run pytest tests/v2 -q`
  - passed: `34 passed in 3.72s`
- `poetry run python scripts/evaluate_bitbat_v2_strategy.py --input data/raw/prices/btcusd_yf_5m.parquet`
  - passed and produced deterministic local comparison output

## Baseline vs improved metrics

- baseline `baseline_v1`
  - `trade_count=4099`
  - `buy_count=2052`
  - `sell_count=2047`
  - `hold_rate=0.7587`
  - `final_equity=9467.84`
  - `realized_pnl=-558.95`
  - `unrealized_pnl=26.79`
  - `max_drawdown_pct=-0.0904`
- improved `filtered_momentum_v2`
  - `trade_count=2221`
  - `buy_count=1113`
  - `sell_count=1108`
  - `hold_rate=0.8693`
  - `final_equity=9592.53`
  - `realized_pnl=-437.84`
  - `unrealized_pnl=30.37`
  - `max_drawdown_pct=-0.0863`
- delta
  - `final_equity_delta=124.69`
  - `trade_count_delta=-1878`
  - `hold_rate_delta=0.1106`
  - `realized_pnl_delta=121.11`
  - `unrealized_pnl_delta=3.58`
  - `max_drawdown_pct_delta=0.0041`

## Waiver reason

- none

## Decision

`approved`
