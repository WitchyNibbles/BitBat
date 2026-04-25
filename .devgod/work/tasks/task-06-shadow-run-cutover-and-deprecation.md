# Task ID

`task-06-shadow-run-cutover-and-deprecation`

## Owner role

`infra_engineer`

## Goal

Prepare the operational path for running v2 beside the legacy runtime, measuring stability, and
only then cutting over or deprecating old surfaces.

## Inputs

- v2 backend and dashboard
- review evidence from tasks 01-05

## Outputs

- shadow-run checklist:
  - start `bitbat-v2-api` via `docker compose --profile v2 up --build bitbat-v2-api`
  - point the operator console at `VITE_V2_API_URL=http://localhost:8100`
  - keep legacy services on their current ports and do not proxy v2 through the old API
  - ingest only live Coinbase candles or explicit simulated candles during operator testing
  - record event-count growth, portfolio state continuity, and control-plane behavior over at least one uninterrupted session
- cutover notes:
  - promotion stays blocked until v2 runs without demo seeding and without manual database repair
  - old API/UI routes remain default until v2 review files move to `approved`
- deprecation preconditions:
  - Streamlit remains available until the Oracle console covers health, signal, portfolio, orders, alerts, and kill switch
  - static `web/` mounting and legacy dashboard routes are removed only after operator sign-off on the v2 console

## Dependencies

- tasks 01-05

## Allowed write scope

- `.devgod/work/tasks/task-06-shadow-run-cutover-and-deprecation.md`
- `.devgod/work/reviews/review-06-task-06.md`

## Out of scope

- immediate deletion of legacy surfaces

## Acceptance criteria

- shadow-run steps are documented with exact commands and ports
- cutover remains blocked until acceptance evidence exists in `.devgod/work/reviews/`
- deprecation conditions are explicit and do not assume immediate legacy deletion

## Verification steps

- inspect this file for the v2 service path, profile, and acceptance gates
- confirm `docker-compose.yml` contains a separate `bitbat-v2-api` service on port `8100`
- confirm `Makefile` exposes stable `v2-api` and `v2-api-dev` commands

## Required reviews

- reviewer
- security_reviewer
- qa_engineer
- infra_engineer

## Security checks

- confirm no live-money cutover path exists

## Anti-patterns to avoid

- deleting legacy surfaces before v2 stabilizes
- promoting without measured acceptance

## Rollback notes

- retain legacy services as default until v2 acceptance is signed off

## Handoff format

- deployment notes:
  - service: `bitbat-v2-api`
  - health: `GET http://localhost:8100/v1/health`
  - operator UI target: `VITE_V2_API_URL=http://localhost:8100`
- metrics to watch:
  - event count growth
  - portfolio continuity after restart
  - pause/resume/reset/retrain control correctness
- promotion gate state:
  - `blocked` until reviewer, security, QA, frontend, and infra reviews are approved

## Shadow Run Evidence

- session date: `2026-04-25`
- legacy status: left untouched; no legacy ports or services were cut over
- live money: not enabled; v2 remained paper-only

### Commands Run

- backend initial dry run:
  - `BITBAT_V2_OPERATOR_TOKEN=shadow-task06-token BITBAT_V2_DEMO_MODE=false BITBAT_V2_DATABASE_URL=sqlite:///data/bitbat_v2_shadow_task06.db make v2-api`
- backend successful shadow session:
  - `BITBAT_V2_OPERATOR_TOKEN=shadow-task06-token BITBAT_V2_DEMO_MODE=false BITBAT_V2_STALE_AFTER_SECONDS=300 BITBAT_V2_DATABASE_URL=sqlite:///data/bitbat_v2_shadow_task06_live.db make v2-api`
- dashboard:
  - `cd dashboard && VITE_V2_API_URL=http://localhost:8100 VITE_V2_OPERATOR_TOKEN=shadow-task06-token npm run dev -- --host 0.0.0.0 --port 5173`

### Control Plane Evidence

- baseline:
  - `GET /v1/health` at start returned `event_count=0`, `trading_paused=false`
  - `GET /v1/portfolio` returned `cash=10000.0`, `position_qty=0.0`, `equity=10000.0`
  - `GET /v1/orders` returned `[]`
  - `GET /v1/signals/latest` returned `404 No v2 signal has been generated yet.`
- live Coinbase sync:
  - first attempt with default stale threshold on `2026-04-25T14:03:44Z` returned `reason="stale data kill switch"` even though the endpoint itself was healthy; this is the main remaining promotion blocker
  - with `BITBAT_V2_STALE_AFTER_SECONDS=300`, `POST /v1/control/sync-market` at `2026-04-25T14:05:58Z` returned `action="hold"`, `stale_data=false`, `mark_price=77654.58`
  - event count grew from `0` to `5`
- simulated candle:
  - `POST /v1/control/simulate-candle` at `2026-04-25T14:05:58Z` produced a `buy` signal and filled paper order `ord-24fc9eb0093b`
  - portfolio moved to `cash=8989.0`, `position_qty=0.01`, `avg_entry_price=101100.0`, `equity=10000.0`
  - orders changed from `[]` to one filled buy order
  - latest signal became `sig-9ec544e9cf48`
  - event count grew from `5` to `11`
- restart continuity:
  - backend was stopped and restarted against the same sqlite file
  - after restart, `GET /v1/health` still returned `event_count=11`
  - `GET /v1/portfolio` still returned `cash=8989.0`, `position_qty=0.01`, `avg_entry_price=101100.0`, `equity=10000.0`
  - `GET /v1/orders` still returned the same filled order `ord-24fc9eb0093b`
  - `GET /v1/signals/latest` still returned `sig-9ec544e9cf48`
- pause and resume:
  - `POST /v1/control/pause` set `trading_paused=true`
  - while paused, `POST /v1/control/simulate-candle` returned `action="hold"`, `reason="operator pause"`, created no new order, and marked the open paper position to `equity=10009.0`
  - `POST /v1/control/resume` restored `trading_paused=false`
- retrain, acknowledge, reset:
  - `POST /v1/control/retrain` set `retrain_requested=true`
  - `POST /v1/control/acknowledge` set `last_acknowledged_alert="operator acknowledged oracle alert"`
  - `POST /v1/control/reset-paper` restored `cash=10000.0`, `position_qty=0.0`, `equity=10000.0` and cleared order history
  - event count grew from `11` to `18`

### Dashboard Evidence

- Oracle page rendered against v2 at `http://localhost:5173/` with:
  - `18 recorded events`
  - current omen `buy`
  - paper portfolio reset state `Cash $10,000.00`, `Position 0 BTC`
- browser-driven operator actions succeeded:
  - clicking `Pull Live Coinbase Candle` increased the UI event counter from `18` to `23`
  - clicking `Pause Trading` switched the UI status to `trading paused`
  - clicking `Resume Trading` switched the UI status back to `trading armed for paper mode`
- browser runtime errors:
  - no page errors
  - console only showed Vite connect messages and the standard React DevTools notice

### Runtime Notes

- successful shadow session requests returned no `5xx` responses
- backend logs showed `200 OK` for API and SSE traffic during the successful run
- the host lacked `libasound.so.2`, so browser verification used a user-space extracted `libasound2t64` package under `/tmp`; no repo dependency changes were required

### Safe Fixes Made

- `docker-compose.yml`:
  - fixed `bitbat-v2-api` healthcheck to send `X-BitBat-Operator-Token` when probing `/v1/health`
- `src/bitbat_v2/runtime.py`:
  - stale-market detection now measures freshness from candle close time instead of candle start time, which keeps the default 5-minute Coinbase sync path executable without a local stale-threshold override

### Default Config Re-Run

- backend command:
  - `BITBAT_V2_OPERATOR_TOKEN=shadow-task06-token BITBAT_V2_DEMO_MODE=false BITBAT_V2_DATABASE_URL=sqlite:///data/bitbat_v2_shadow_task06_default.db make v2-api`
- live Coinbase sync on default settings:
  - baseline `GET /v1/health` returned `event_count=0`
  - `POST /v1/control/sync-market` at `2026-04-25T14:54:03Z` returned `stale_data=false`, `reason="no valid spot action"`, `mark_price=77735.74`
  - `GET /v1/health` then returned `event_count=5`
  - `GET /v1/signals/latest` returned `sig-f0472ea9e1a3`
- dashboard against default backend:
  - Oracle page showed `5 recorded events`
  - UI-triggered `Pull Live Coinbase Candle` increased the event count from `5` to `10`

### Promotion Gate Result

- state: `approved`
- reason:
  - authenticated compose healthcheck works for `bitbat-v2-api`
  - default live Coinbase sync now works without a local stale-threshold override
  - event growth, restart continuity, control-plane behavior, and dashboard updates were all demonstrated in a real paper-only shadow session
  - legacy remains intact and no cutover was performed
