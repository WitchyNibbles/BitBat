# Task ID

`task-12-deterministic-strategy-improvement`

## Owner role

`backend_engineer`

## Goal

Improve `bitbat_v2` paper-trading decision quality with a deterministic, explainable strategy
upgrade and a reproducible offline comparison against the current baseline heuristic.

## Current gate state

`ready`

## Inputs

- complete `task-04-strategy-risk-and-paper-broker`
- complete `task-11-autonomous-paper-execution`
- current v2 runtime remains paper-only and uses a thin deterministic heuristic

## Outputs

- extracted v2 strategy module with named baseline and improved strategies
- improved default paper-only strategy for runtime and API output
- reproducible offline evaluation path using repo-local 5-minute BTC data
- tests covering strategy logic, runtime behavior, API explanations, and evaluation metrics
- review evidence comparing baseline vs improved behavior with concrete local metrics

## Gate decision

`complete`

## Dependencies

- `task-04-strategy-risk-and-paper-broker`
- `task-11-autonomous-paper-execution`

## Allowed write scope

- `src/bitbat_v2/`
- `tests/v2/`
- `scripts/evaluate_bitbat_v2_strategy.py`
- `.devgod/work/tasks/task-12-deterministic-strategy-improvement.md`
- `.devgod/work/reviews/review-12-task-12.md`
- `.devgod/work/release/release-2026-04-25-v2-strategy-improvement.md`
- `.devgod/work/plans/plan-2026-04-25-bitbat-clean-room.md`
- `.devgod/work/briefs/brief-2026-04-25-bitbat-rebuild.md`

## Out of scope

- live-money trading
- cutover work
- legacy service removal
- non-deterministic or opaque strategy behavior

## Acceptance criteria

- a new deterministic strategy-improvement slice exists after task 11
- `bitbat_v2` uses the improved deterministic strategy by default
- the prior heuristic remains available as a named baseline for offline comparison only
- decisions and API/runtime output remain explainable
- offline evaluation compares baseline vs improved strategy with concrete repo-local metrics
- runtime remains paper-only and legacy behavior remains intact
- touched v2 tests pass

## Verification steps

- run `poetry run pytest tests/v2/test_strategy.py tests/v2/test_evaluation.py tests/v2/test_runtime.py tests/v2/test_api.py -q`
- run `poetry run pytest tests/v2 -q`
- run `poetry run python scripts/evaluate_bitbat_v2_strategy.py --input data/raw/prices/btcusd_yf_5m.parquet`

## Required reviews

- reviewer
- security_reviewer
- qa_engineer

## Security checks

- confirm the runtime remains paper-only
- confirm no new live-trading or cutover flags are introduced
- confirm auth boundaries on the operator API remain unchanged
- confirm evaluation uses repo-local data and does not widen trust boundaries

## Anti-patterns to avoid

- hidden background strategy behavior with no tests
- profitability claims unsupported by repo-local evidence
- coupling strategy improvement to legacy runtime changes

## Rollback notes

- revert the default v2 strategy to the named baseline heuristic while keeping paper-only execution

## Handoff format

- strategy interface and default behavior
- config/env additions
- evaluation command and output
- verification evidence

## Verification evidence

- targeted suite:
  - `poetry run pytest tests/v2/test_strategy.py tests/v2/test_evaluation.py tests/v2/test_runtime.py tests/v2/test_api.py -q`
  - passed: `29 passed in 3.43s`
- full v2 suite:
  - `poetry run pytest tests/v2 -q`
  - passed: `34 passed in 3.72s`
- offline comparison:
  - `poetry run python scripts/evaluate_bitbat_v2_strategy.py --input data/raw/prices/btcusd_yf_5m.parquet`
  - baseline `baseline_v1`:
    - `trade_count=4099`
    - `hold_rate=0.7587`
    - `final_equity=9467.84`
    - `realized_pnl=-558.95`
    - `unrealized_pnl=26.79`
    - `max_drawdown_pct=-0.0904`
  - improved `filtered_momentum_v2`:
    - `trade_count=2221`
    - `hold_rate=0.8693`
    - `final_equity=9592.53`
    - `realized_pnl=-437.84`
    - `unrealized_pnl=30.37`
    - `max_drawdown_pct=-0.0863`
  - deltas:
    - `final_equity_delta=124.69`
    - `trade_count_delta=-1878`
    - `hold_rate_delta=0.1106`
    - `realized_pnl_delta=121.11`
    - `max_drawdown_pct_delta=0.0041`
