# Task Packet

## Task ID

`2026-05-09-witchy-tech-frontend-plan`

## Owner role

`frontend_designer`

## Completion standard

`specialist_verified`

## Required specialist roles

- `frontend_designer`
- `backend_engineer`
- `reviewer`
- `qa_engineer`
- `security_reviewer`

## Quality gates

- `frontend_acceptance`
- `accessibility_acceptance`
- `responsive_acceptance`
- `e2e_required`
- `performance_check_required`

## Goal

Execute the witchy-tech frontend refactor plan on the live React dashboard, make `/v1/paper` and
`/v1/performance` the paper-trading source of truth, ship a dedicated paper cockpit, modernize
navigation semantics, and verify the result with unit tests, backend tests, Playwright E2E, and a
live browser pass.

## Inputs

- `.devgod/work/plans/plan-2026-05-09-witchy-tech-frontend-plan.md`
- `.devgod/work/briefs/brief-2026-05-09-witchy-tech-frontend-plan.md`
- `dashboard/src/**`
- `src/bitbat_v2/api/app.py`
- `tests/v2/test_api.py`
- `.devgod/rules/frontend-acceptance.md`

## Dependencies

- v2 operator API endpoints `/v1/health`, `/v1/paper`, `/v1/performance`, `/v1/signals/latest`
- local Vite dashboard toolchain
- Playwright browser runtime plus extracted local `libasound` shim for this environment

## Outputs

- routed React operator console with `Paper Trade` slice
- shared paper view-model formatting and freshness logic
- refreshed `Command Center`, `Oracle`, `Performance`, and `System` routes
- Vitest unit coverage, Playwright E2E coverage, and v2 CORS regression coverage
- legacy `web/` marked as non-authoritative

## Allowed write scope

- `dashboard/**`
- `src/bitbat_v2/api/app.py`
- `tests/v2/test_api.py`
- `README.md`
- `web/**`
- `.devgod/work/tasks/**`
- `.devgod/work/reviews/**`

## Out of scope

- redesigning the legacy Python API surface
- deleting the legacy `web/` shell
- changing production deployment topology

## Assumptions

### Approved assumptions

- `dashboard/` is the authoritative frontend surface
- the v2 paper endpoints are stable enough to drive the first full cockpit slice
- local-only CORS expansion for common dashboard origins is acceptable

### Blocked assumptions

- do not assume the static `web/` shell remains authoritative
- do not assume mocked E2E alone proves live runtime behavior
- do not assume Vite dev-server HTML responses are valid v2 API probe responses

## Acceptance criteria

- real route semantics replace hash navigation
- `Paper Trade` renders portfolio, PnL, orders, closed trades, alerts, and equity curve from v2
- `Command Center`, `Oracle`, `Performance`, and `System` use clearer truthful language
- local dashboard origins can reach the v2 API in live browser use
- build, lint, unit tests, backend tests, and Playwright E2E all pass

## Verification steps

- `npm run build` in `dashboard/`
- `npm run lint` in `dashboard/`
- `npm run test` in `dashboard/`
- `npm audit --json` in `dashboard/`
- `LD_LIBRARY_PATH="$PWD/.local-libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" npm run e2e` in `dashboard/`
- `poetry run pytest tests/v2/test_api.py`
- live Playwright browser pass on:
  - `http://localhost:5173/paper-trade`
  - `http://localhost:5173/oracle`

## Required reviews

- `reviewer`
- `security_reviewer`
- `qa_engineer`

## Security checks

- operator token remains header-based and is never rendered into UI copy
- destructive pause action requires explicit browser confirmation
- v2 base-url probing rejects HTML fallback responses
- local CORS allowlist is explicit to known dashboard origins only
- `npm audit` reports zero vulnerabilities after lockfile remediation

## Retrieval guidance

- trust `src/bitbat_v2/api/app.py` for paper-trading contracts and CORS policy
- trust `dashboard/src/api/v2Client.ts` and `dashboard/src/api/paperViewModel.ts` for frontend read models
- treat mocked Playwright routes as route-behavior coverage, not live API proof

## Anti-patterns to avoid

- reconstructing paper state from unrelated legacy endpoints
- letting the dev server HTML fallback masquerade as a valid API probe
- letting mocked E2E replace live browser verification
- widening CORS to `*`

## Rollback notes

- revert `dashboard/src/**` and `dashboard/tests/**` for UI rollback
- revert `src/bitbat_v2/api/app.py` and `tests/v2/test_api.py` together if local-origin CORS needs to back out
- legacy `web/` archive notes can be reverted independently

## Handoff format

- owner role: `frontend_designer`
- completion standard: `specialist_verified`
- specialist execution evidence: routed React refactor, v2 probe hardening, local-origin CORS fix
- quality gate evidence: build, lint, vitest, pytest, Playwright E2E, live browser proof
