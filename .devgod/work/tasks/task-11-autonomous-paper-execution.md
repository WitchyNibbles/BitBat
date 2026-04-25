# Task ID

`task-11-autonomous-paper-execution`

## Owner role

`backend_engineer`

## Goal

Add a safe autonomous paper-mode execution loop to `bitbat_v2` so the runtime can poll Coinbase,
process each new candle once, and keep trading in paper mode without manual button presses.

## Current gate state

`ready`

## Inputs

- approved `task-06-shadow-run-cutover-and-deprecation`
- approved `task-07-immediate-operational-cleanup`
- current v2 runtime and API only process candles through manual control endpoints

## Outputs

- background paper-only market sync loop for v2
- duplicate-candle guard so the same Coinbase candle is not processed repeatedly
- operator-visible runtime status for the autonomous loop
- tests for loop behavior, duplicate prevention, and API visibility

## Gate decision

`complete`

## Config flags

- `BITBAT_V2_AUTORUN_ENABLED=true|false`
- `BITBAT_V2_AUTORUN_INTERVAL_SECONDS=<seconds>`

## Behavior shipped

- autonomous paper trader runs inside the v2 API process when autorun is enabled
- the runtime suppresses repeated processing of the same Coinbase candle start time
- `/v1/health` now reports autonomous status including:
  - `enabled`
  - `interval_seconds`
  - `running`
  - `last_cycle_status`
  - `last_cycle_started_at`
  - `last_cycle_completed_at`
  - `last_error`
  - `last_processed_candle_start`
  - `last_action`
- manual `/v1/control/sync-market` now also benefits from duplicate-candle suppression

## Strategy note

- this slice fixes autonomy and duplicate execution
- it does not claim guaranteed profitability
- the current signal model remains the deterministic heuristic from tasks 03 and 04

## Dependencies

- `task-03-market-data-and-model-thin-slice`
- `task-04-strategy-risk-and-paper-broker`

## Allowed write scope

- `src/bitbat_v2/`
- `tests/v2/`
- `.devgod/work/tasks/task-11-autonomous-paper-execution.md`
- `.devgod/work/reviews/review-11-task-11.md`
- `.devgod/work/plans/plan-2026-04-25-bitbat-clean-room.md`
- `.devgod/work/briefs/brief-2026-04-25-bitbat-rebuild.md`

## Out of scope

- live-money execution
- discretionary agent trading
- guaranteed profitability claims
- legacy runtime changes

## Acceptance criteria

- v2 can run an autonomous paper-only sync loop when explicitly enabled
- the same Coinbase candle is processed at most once by the autonomous path
- operator-visible status shows whether autonomous execution is enabled and whether the last sync
  succeeded, skipped, or failed
- existing manual controls still work
- tests cover autonomous sync, duplicate suppression, and status exposure

## Verification steps

- run `pytest tests/v2/test_autorun.py tests/v2/test_api.py tests/v2/test_runtime.py -q`

## Required reviews

- reviewer
- security_reviewer
- qa_engineer

## Security checks

- confirm autonomous execution remains paper-only
- confirm duplicate suppression prevents repeated action on the same market candle
- confirm auth boundaries on the operator API remain unchanged

## Anti-patterns to avoid

- processing the same exchange candle over and over
- hidden background mutation with no operator visibility
- coupling autonomous paper execution to live-money or cutover work

## Rollback notes

- disable the loop via config and fall back to manual sync endpoints

## Handoff format

- config flags
- loop behavior
- duplicate guard behavior
- verification evidence

## Verification evidence

- targeted suite:
  - `poetry run pytest tests/v2/test_autorun.py tests/v2/test_runtime.py tests/v2/test_api.py -q`
  - passed
- full v2 suite:
  - `poetry run pytest tests/v2 -q`
  - passed
