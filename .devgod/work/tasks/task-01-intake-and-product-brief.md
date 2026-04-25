# Task ID

`task-01-intake-and-product-brief`

## Owner role

`product_strategist`

## Goal

Capture the customer request, operator journey, risks, trust boundaries, and the paper-trading-only
 stop/go line for BitBat v2.

## Inputs

- user request
- rebuild brief
- repository runtime survey

## Outputs

- intake brief
- thin-slice operator journey
- success criteria and risk framing

## Dependencies

- none

## Allowed write scope

- `.devgod/work/briefs/brief-2026-04-25-bitbat-rebuild.md`
- `.devgod/work/plans/plan-2026-04-25-bitbat-clean-room.md`
- `.devgod/work/tasks/task-01-intake-and-product-brief.md`
- `.devgod/work/reviews/review-01-task-01.md`

## Out of scope

- implementation changes

## Acceptance criteria

- brief defines goal, audience, constraints, risks, trust boundaries, and success criteria
- stop/go explicitly permits paper trading and blocks live capital
- thin-slice operator journey is stated

## Verification steps

- read artifact and confirm every brief section is present

## Required reviews

- reviewer

## Security checks

- confirm live execution remains out of scope

## Anti-patterns to avoid

- vague product language
- hidden live-money assumptions

## Rollback notes

- remove brief artifacts if customer direction changes materially

## Handoff format

- concise summary plus exact artifact paths
