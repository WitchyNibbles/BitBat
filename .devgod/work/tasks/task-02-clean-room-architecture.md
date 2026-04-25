# Task ID

`task-02-clean-room-architecture`

## Owner role

`solution_architect`

## Goal

Define the clean-room `bitbat_v2` namespace, data model, runtime boundaries, event schema, storage
shape, and cutover posture.

## Inputs

- rebuild brief
- current repo topology

## Outputs

- `src/bitbat_v2/` package scaffold
- config, domain, storage, runtime, and API modules
- architecture notes embedded in code and plan artifacts

## Dependencies

- task 01

## Allowed write scope

- `src/bitbat_v2/`
- `tests/v2/`
- `pyproject.toml`
- `.devgod/work/tasks/task-02-clean-room-architecture.md`
- `.devgod/work/reviews/review-02-task-02.md`

## Out of scope

- live-money execution
- legacy runtime removal

## Acceptance criteria

- v2 code lives in a separate package
- storage supports append-only runtime events and read models
- public v2 API exists behind `/v1`
- legacy surfaces are treated as reference-only

## Verification steps

- run v2 backend tests
- inspect route list and storage schema behavior

## Required reviews

- reviewer
- security_reviewer
- infra_engineer

## Security checks

- no secret assumptions
- operator controls validated
- runtime storage isolated from devgod orchestration state

## Anti-patterns to avoid

- importing legacy modules into the v2 core
- broad write scopes

## Rollback notes

- remove `src/bitbat_v2/` and v2 tests if architecture is replaced

## Handoff format

- changed paths, API summary, verification evidence
