# Task ID

`task-05-live-dashboard-and-operator-controls`

## Owner role

`frontend_designer`

## Goal

Expose the v2 runtime through a witchy operator console with real-time event flow, signal state,
portfolio state, order history, alerts, and kill-switch controls.

## Inputs

- v2 API surface
- existing React dashboard shell

## Outputs

- operator console page
- v2 dashboard API client and SSE hook
- control actions wired to the v2 backend

## Dependencies

- task 04

## Allowed write scope

- `dashboard/src/`
- `.devgod/work/tasks/task-05-live-dashboard-and-operator-controls.md`
- `.devgod/work/reviews/review-05-task-05.md`

## Out of scope

- full replacement of every legacy page
- multi-user UX

## Acceptance criteria

- operator can see live events, latest signal, portfolio summary, and orders
- operator can pause, resume, reset paper account, and request retrain
- UI is visibly witchy and mobile-safe

## Verification steps

- build the dashboard
- verify the console renders against the v2 API

## Required reviews

- reviewer
- security_reviewer
- qa_engineer
- frontend_designer

## Security checks

- control actions use explicit POST requests
- error states are visible to the operator

## Anti-patterns to avoid

- hiding runtime status
- burying kill switch controls

## Rollback notes

- remove v2 operator console page if UI direction changes

## Handoff format

- visual intent, state flows, build evidence
