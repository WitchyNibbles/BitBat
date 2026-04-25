# Task ID

`task-10-legacy-deprecation-prerequisites`

## Owner role

`planner`

## Goal

Define the explicit prerequisites for any future legacy deprecation while keeping all current legacy
services and routes intact.

## Current gate state

`hold`

## Inputs

- approved `task-06-shadow-run-cutover-and-deprecation`
- future cutover-preparation outputs from `task-09`

## Outputs

- parity matrix between the Oracle console and legacy operator surfaces
- inventory of legacy services, routes, and consumers that would be affected later
- deprecation approval prerequisites and rollback expectations for a future RFC

## Legacy parity requirements

- Oracle console must cover:
  - health
  - latest signal
  - portfolio state
  - order history
  - alerts
  - kill-switch or pause controls
- affected legacy surfaces to keep intact until later approval:
  - `streamlit/`
  - legacy API on port `8000`
  - legacy dashboard and proxy paths already in service

## Future deprecation prerequisites

1. a future approved cutover phase exists and has completed successfully
2. operator sign-off confirms the Oracle console is acceptable as the default surface
3. legacy consumers, scripts, dashboards, and routes are inventoried
4. a deprecation RFC names the exact services or routes to retire
5. rollback and archive ownership are assigned before any shutdown window starts

## Dependencies

- `task-09-future-cutover-preparation`
- future approved cutover phase not yet created

## Allowed write scope

- `.devgod/work/tasks/task-10-legacy-deprecation-prerequisites.md`
- `.devgod/work/reviews/review-10-task-10.md`
- `.devgod/work/plans/plan-2026-04-25-bitbat-clean-room.md`
- `.devgod/work/briefs/brief-2026-04-25-bitbat-rebuild.md`

## Out of scope

- removing Streamlit
- removing legacy API or dashboard routes
- switching default operators to v2

## Acceptance criteria

- the parity matrix covers health, signals, portfolio, orders, alerts, and kill-switch behavior
- legacy consumers and fallback paths are identified
- deprecation prerequisites require future operator sign-off and a separate approved deprecation RFC
- the task does not schedule or imply legacy shutdown

## Verification steps

- inspect the task for an explicit hold state
- verify no removal or redirect step is included
- verify legacy is described as intact until a future approved task changes that

## Required reviews

- reviewer
- infra_engineer

## Security checks

- confirm no trust boundary is widened by future parity assumptions

## Anti-patterns to avoid

- treating feature parity guesses as evidence
- coupling legacy shutdown to cutover preparation in one packet
- deleting fallback surfaces before operator sign-off

## Rollback notes

- no runtime rollback is needed because this task stays on hold

## Handoff format

- parity matrix
- consumer inventory
- future deprecation RFC prerequisites
