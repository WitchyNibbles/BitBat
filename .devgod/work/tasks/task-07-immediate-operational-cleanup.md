# Task ID

`task-07-immediate-operational-cleanup`

## Owner role

`infra_engineer`

## Goal

Close out the approved task-06 shadow session safely, keep the evidence useful, and remove ad hoc
runtime residue without changing the default legacy posture.

## Current gate state

`complete`

## Inputs

- approved `task-06-shadow-run-cutover-and-deprecation`
- approved `review-06-task-06`
- currently running local shadow-session processes on ports `8100` and `5173`
- observed on `2026-04-25`:
  - `GET /v1/health` on `http://localhost:8100` returned `status=ok`, `trading_paused=false`,
    `event_count=68`

## Outputs

- operator runbook for starting, checking, and stopping the paper-only shadow session
- shadow watchlist covering health, event growth, restart continuity, control-plane behavior, and
  dashboard freshness
- explicit decision and evidence for the currently running shadow-session processes
- artifact hygiene note identifying which shadow databases and evidence files are canonical

## Gate decision

`complete`

The residual shadow session was shut down on `2026-04-25`. No named soak owner was recorded.

## Dependencies

- `task-06-shadow-run-cutover-and-deprecation`
- `review-06-task-06`

## Allowed write scope

- `.devgod/work/tasks/task-07-immediate-operational-cleanup.md`
- `.devgod/work/reviews/review-07-task-07.md`
- `.devgod/work/plans/plan-2026-04-25-bitbat-clean-room.md`
- `.devgod/work/briefs/brief-2026-04-25-bitbat-rebuild.md`

## Out of scope

- cutover execution
- live-money enablement
- legacy service removal
- product or strategy changes

## Acceptance criteria

- a start, health-check, smoke-test, and stop sequence is documented with exact commands and ports
- the shadow watchlist is concrete and tied to task-06 evidence
- current shadow-session processes are explicitly classified as either:
  - a named soak owned by an operator
  - residual task-06 state to be shut down
- if classified as residual state, the stop command and expected clean state are documented
- legacy services remain unchanged

## Verification steps

- inspect `task-06` and `review-06` for the baseline commands and evidence
- verify whether listeners still exist on ports `8100` and `5173`
- verify the stop path leaves no v2 shadow listeners behind if cleanup is performed

## Runbook

### Start

- API:
  - `BITBAT_V2_OPERATOR_TOKEN=<token> BITBAT_V2_DEMO_MODE=false BITBAT_V2_DATABASE_URL=sqlite:///data/bitbat_v2_shadow_task06_default.db make v2-api`
- dashboard:
  - `cd dashboard && VITE_V2_API_URL=http://localhost:8100 VITE_V2_OPERATOR_TOKEN=<token> npm run dev -- --host 0.0.0.0 --port 5173`

### Health checks

- API health:
  - `curl -H 'X-BitBat-Operator-Token: <token>' http://localhost:8100/v1/health`
- portfolio baseline:
  - `curl -H 'X-BitBat-Operator-Token: <token>' http://localhost:8100/v1/portfolio`
- latest signal:
  - `curl -H 'X-BitBat-Operator-Token: <token>' http://localhost:8100/v1/signals/latest`

### Smoke checks

- confirm `/v1/health` returns `status=ok`
- confirm event count grows after `POST /v1/control/sync-market`
- confirm dashboard loads at `http://localhost:5173/`
- confirm pause and resume work from the API or Oracle console

### Stop

- stop the API:
  - `pkill -f 'bitbat_v2.api.app:app --host 0.0.0.0 --port 8100'`
- stop the dashboard:
  - `pkill -f 'dashboard/node_modules/.bin/vite --host 0.0.0.0 --port 5173'`
- verify listeners are gone:
  - `ss -ltnp '( sport = :5173 or sport = :8100 )'`

## Shadow watchlist

- health stays `status=ok`
- `event_count` grows when sync or simulate actions run
- restart preserves `event_count`, portfolio, orders, and latest signal
- pause blocks execution without creating new orders
- resume restores paper-trading readiness
- dashboard event counter and status line update without page errors
- legacy services remain on their existing ports and are not proxied through v2

## Artifact hygiene

- canonical shadow evidence source:
  - `.devgod/work/tasks/task-06-shadow-run-cutover-and-deprecation.md`
  - `.devgod/work/reviews/review-06-task-06.md`
- preferred baseline sqlite file for future paper-only reruns:
  - `data/bitbat_v2_shadow_task06_default.db`
- keep as historical evidence only:
  - `data/bitbat_v2_shadow_task06.db`
  - `data/bitbat_v2_shadow_task06_live.db`

## Execution evidence

- observed before shutdown on `2026-04-25`:
  - listeners existed on `8100` and `5173`
  - `GET /v1/health` returned `status=ok`, `trading_paused=false`, `event_count=68`
- shutdown actions executed:
  - terminated the parent `make v2-api` process and remaining `uvicorn` worker
  - terminated the `vite` dashboard process
- observed after shutdown:
  - `curl` to `http://localhost:8100/v1/health` failed to connect
  - `ss -ltnp '( sport = :5173 or sport = :8100 )'` returned no listeners

## Required reviews

- reviewer
- infra_engineer

## Security checks

- confirm the paper-only warning remains explicit in the runbook
- confirm no cutover or live-money instruction is introduced

## Anti-patterns to avoid

- leaving orphaned shadow services running with no named owner
- treating a leftover local session as a formal soak window
- overwriting legacy defaults while doing cleanup

## Rollback notes

- shadow services can be restarted with the documented paper-only commands from task 06

## Handoff format

- runbook:
  - start commands
  - health checks
  - smoke steps
  - stop commands
- observed residual state:
  - listeners on `8100` and `5173`
  - v2 health `status=ok`, `event_count=68`
- process decision:
  - `shutdown now` executed because no named operator recorded a time-boxed soak owner or window
- artifact baseline:
  - canonical shadow database and evidence locations
