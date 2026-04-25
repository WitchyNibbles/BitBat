# Task ID

`task-04-strategy-risk-and-paper-broker`

## Owner role

`backend_engineer`

## Goal

Implement deterministic buy/sell/hold decisions, hard risk caps, stale-data protection, operator
pause, and paper order/accounting flows.

## Inputs

- v2 runtime and signal generation

## Outputs

- strategy and risk services
- paper broker and portfolio projection
- control endpoints for pause/resume/reset/retrain/ack

## Dependencies

- task 03

## Allowed write scope

- `src/bitbat_v2/`
- `tests/v2/`
- `.devgod/work/tasks/task-04-strategy-risk-and-paper-broker.md`
- `.devgod/work/reviews/review-04-task-04.md`

## Out of scope

- live exchange orders
- discretionary agent trading

## Acceptance criteria

- decisions are deterministic and fully explained
- paused or stale systems do not place paper orders
- paper fills update portfolio state and order history

## Verification steps

- run runtime, storage, and API tests

## Required reviews

- reviewer
- security_reviewer
- qa_engineer

## Security checks

- verify pause semantics block execution
- verify reset and retrain controls are explicit POST actions

## Anti-patterns to avoid

- side-effectful trading without risk checks
- silent state mutation

## Rollback notes

- remove strategy/broker services if the risk model is redesigned

## Handoff format

- controls added, risk rules, verification evidence
