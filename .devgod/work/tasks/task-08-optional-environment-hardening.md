# Task ID

`task-08-optional-environment-hardening`

## Owner role

`infra_engineer`

## Goal

Replace temporary local verification hacks with repeatable environment setup while keeping this work
explicitly optional and separate from cutover.

## Current gate state

`optional`

## Inputs

- approved `task-06-shadow-run-cutover-and-deprecation`
- host note that browser verification required a temporary `/tmp` `libasound.so.2` workaround
- current local startup contract for v2 API and dashboard

## Outputs

- hardening follow-up for browser runtime dependencies
- reproducible local environment notes for API, dashboard, and browser verification
- optional startup-validation follow-up for required v2 environment variables

## Recommended host fix

- host OS observed: `Ubuntu 24.04.3 LTS`
- missing runtime library: `libasound.so.2`
- preferred package on this host family:
  - `libasound2t64`
- machine-level install path:
  - `sudo apt-get update && sudo apt-get install -y libasound2t64`

If browser verification moves into a container or image later, the same dependency must exist in
that runtime layer instead of relying on `/tmp` extraction.

## Concrete next actions

1. install `libasound2t64` on the verification host or bake it into the browser runtime image
2. rerun the headless UI verification path without extracting any user-space package into `/tmp`
3. record the clean verification path in the relevant operator or environment docs
4. optionally add a preflight check that fails fast when `libasound.so.2` is missing

## Dependencies

- `task-06-shadow-run-cutover-and-deprecation`

## Allowed write scope

- `.devgod/work/tasks/task-08-optional-environment-hardening.md`
- `.devgod/work/reviews/review-08-task-08.md`
- `.devgod/work/plans/plan-2026-04-25-bitbat-clean-room.md`
- `.devgod/work/briefs/brief-2026-04-25-bitbat-rebuild.md`

## Out of scope

- cutover approval
- legacy shutdown
- live-money enablement

## Acceptance criteria

- the browser workaround is explicitly marked temporary and non-baseline
- a machine-level dependency fix or equivalent reproducible browser runtime layer is identified
- follow-up work states how to rerun headless verification without `/tmp` library extraction
- any env-validation hardening stays paper-only and does not add secrets to the repo
- the Ubuntu host-family package name is captured explicitly for this machine class

## Verification steps

- inspect `task-06` for the current workaround note
- verify the hardening proposal does not depend on manual `/tmp` extraction
- verify the proposal is framed as environment hygiene, not product completion or cutover readiness

## Required reviews

- reviewer
- infra_engineer

## Security checks

- confirm no secrets are introduced into docs or example commands
- confirm hardening work does not widen runtime exposure

## Anti-patterns to avoid

- treating a one-off local package extraction as the supported path
- bundling host hardening together with cutover approval
- implying the repo itself should vendor machine libraries

## Rollback notes

- if optional hardening is deferred, retain task-06 evidence and keep the workaround note visible

## Handoff format

- dependency gap:
  - missing host browser dependency `libasound.so.2`
- recommended fix:
  - `sudo apt-get install -y libasound2t64` on Ubuntu 24.04-class hosts
  - equivalent browser-runtime provisioning for containerized verification
- follow-up check:
  - rerun headless UI verification without `/tmp` hacks
