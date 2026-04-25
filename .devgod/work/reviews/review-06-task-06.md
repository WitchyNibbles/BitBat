# Task ID

`task-06-shadow-run-cutover-and-deprecation`

## Reviewer role

`reviewer + security_reviewer + qa_engineer + infra_engineer`

## Review state

`approved`

## Severity

`medium`

## Findings

- real shadow session executed on `2026-04-25` with `bitbat_v2.api.app:app` on `http://localhost:8100` and the React dashboard on `http://localhost:5173`
- v2 paper state survived a backend restart without manual database repair:
  - `event_count` stayed at `11`
  - portfolio stayed at `cash=8989.0`, `position_qty=0.01`, `avg_entry_price=101100.0`
  - order `ord-24fc9eb0093b` remained available after restart
- control plane behavior matched expectations:
  - live sync succeeded with `stale_data=false` once `BITBAT_V2_STALE_AFTER_SECONDS=300` was set
  - simulated candle created a paper fill
  - pause blocked execution without adding a new order
  - retrain, acknowledge, and reset mutated control and paper state correctly
  - order history cleared after reset
- dashboard evidence is real, not inferred:
  - Oracle page rendered live v2 state
  - UI-triggered live sync moved the event counter from `18` to `23`
  - UI-triggered pause and resume toggled the status line correctly
  - browser run showed no page errors
- safe infra fix applied:
  - `docker-compose.yml` healthcheck for `bitbat-v2-api` now sends the operator token when probing `/v1/health`
- safe runtime fix applied:
  - stale-market detection now evaluates freshness from candle close time instead of candle start time
- default shadow path is now verified:
  - `POST /v1/control/sync-market` succeeded on default settings with `stale_data=false`
  - default-session event count grew from `0` to `5`
  - the Oracle dashboard rendered `5 recorded events` and a UI-triggered live sync moved it to `10`
- promotion conditions for task 06 are satisfied:
  - live shadow evidence exists
  - no live-money path was enabled
  - legacy services remain intact and unchanged

## Waiver reason

- none

## Decision

`approved`
