# Task ID

`task-09-future-cutover-preparation`

## Owner role

`planner + infra_engineer`

## Goal

Define the future paper-only cutover readiness bar without performing a cutover, changing the
default surface, or enabling live-money trading.

## Current gate state

`blocked`

Planning-only work is allowed. Any execution that changes defaults remains blocked.

## Inputs

- approved `task-06-shadow-run-cutover-and-deprecation`
- approved `review-06-task-06`
- task-07 cleanup outputs if available

## Outputs

- explicit cutover go/no-go criteria
- repeat-shadow expectations for a later extended paper soak
- written rollback and smoke-check requirements for a future approved rehearsal

## Go/no-go checklist

- `default config only`
  - no local stale-threshold overrides
  - no ad hoc browser-library workarounds
- `paper only`
  - live-money execution remains disabled
- `repeatability`
  - at least one fresh rerun succeeds from a clean process start using documented commands
- `state continuity`
  - restart preserves portfolio, orders, event count, and latest signal
- `operator controls`
  - pause, resume, reset, retrain, acknowledge, and live sync behave as documented
- `rollback`
  - same-day rollback path is written and tested at the process-routing level before any default change
- `operator sign-off`
  - explicit approval is recorded before any default-surface switch

## Future task gates

- `extended-paper-soak`
  - state: `blocked`
  - opens only after cleanup and any chosen environment hardening are complete
- `cutover-rehearsal-plan`
  - state: `hold`
  - opens only after the extended paper soak is approved
- `actual cutover`
  - state: `blocked`
  - out of scope for this packet

## Dependencies

- `task-06-shadow-run-cutover-and-deprecation`
- `review-06-task-06`
- `task-07-immediate-operational-cleanup`

## Allowed write scope

- `.devgod/work/tasks/task-09-future-cutover-preparation.md`
- `.devgod/work/reviews/review-09-task-09.md`
- `.devgod/work/plans/plan-2026-04-25-bitbat-clean-room.md`
- `.devgod/work/briefs/brief-2026-04-25-bitbat-rebuild.md`

## Out of scope

- executing cutover
- changing default ports or routes
- enabling live-money execution
- shutting down legacy services

## Acceptance criteria

- go/no-go criteria are written down and require:
  - default config operation
  - no demo seeding
  - no manual database repair
  - repeatable paper-only shadow success
  - operator sign-off
  - same-day rollback path
- a later extended paper soak is defined as a prerequisite, not implied complete
- any rehearsal is documented as future planning only

## Verification steps

- inspect the task for an explicit cutover block
- verify live-money remains explicitly stopped
- verify legacy remains the default until a future approved phase says otherwise

## Required reviews

- reviewer
- security_reviewer
- infra_engineer

## Security checks

- confirm no live-money path is introduced
- confirm auth and rollback expectations remain explicit

## Anti-patterns to avoid

- turning planning criteria into an implied approval
- using task-06 approval as permission to switch defaults
- writing a cutover packet with no rollback bar

## Rollback notes

- no runtime rollback is needed because this task is planning-only

## Handoff format

- go/no-go checklist
- soak prerequisites
- future rehearsal prerequisites
