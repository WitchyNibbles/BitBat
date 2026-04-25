# Task ID

`task-02-clean-room-architecture`

## Reviewer role

`reviewer + security_reviewer + infra_engineer`

## Review state

`passed`

## Severity

`low`

## Findings

- `src/bitbat_v2/` is isolated from the legacy runtime and packaged in `pyproject.toml`
- v2 storage uses dedicated `v2_*` tables and a separate default database path
- `/v1` routes are exposed through the clean-room FastAPI app with operator auth and bounded SSE
- compose and Make targets now provide a distinct v2 startup path for architecture validation

## Waiver reason

- none

## Decision

`approved`
