# Review Gate

## Task ID

`2026-05-09-witchy-tech-frontend-plan`

## Reviewer role

`security_reviewer`

## Actor

`codex-security-gate`

## Actor role

`security_reviewer`

## Provenance status

`runtime_verified`

## Review state

`passed`

## Severity

`low`

## Specialist execution evidence

- `dashboard/src/api/v2Client.ts` keeps the operator token in headers only and now rejects HTML
  fallback responses during base-url probing.
- `src/bitbat_v2/api/app.py` widens CORS only to explicit local dashboard origins instead of using
  a wildcard.
- `dashboard/src/pages/Oracle.tsx` places confirmation in front of the pause action.
- `README.md` and `web/README.md` demote the legacy static shell to archive/demo status.

## Quality gate evidence

- `npm audit --json`
- `npm run build`
- `npm run test`
- `poetry run pytest tests/v2/test_api.py`
- live Playwright MCP browser verification against the patched v2 API

## Findings

- No unresolved high or critical security issue remains in the delivered slice.
- The direct `vite` advisory and transitive frontend audit findings were remediated via `npm audit fix`
  before completion.

## Residual risk

- Local-origin CORS policy intentionally supports multiple common dev origins; future production
  deploys should keep environment-specific origin control explicit.
- Legacy API-backed diagnostics pages were not redesigned in this slice and retain their prior
  trust boundaries.

## Verification evidence

- `npm audit --json` reported zero vulnerabilities after remediation.
- `poetry run pytest tests/v2/test_api.py` passed with the local-dashboard CORS regression test.
- `LD_LIBRARY_PATH="$PWD/.local-libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" npm run e2e`
- Runtime proof: Playwright MCP live browser session at `http://localhost:5173/paper-trade` loaded
  successfully against the live v2 backend after the CORS allowlist patch.
- Runtime proof: Playwright MCP live browser session at `http://localhost:5173/oracle` executed a
  paper-only control action without exposing the operator token in visible UI copy.

## Waiver authority

`none`

## Waiver reason

None.

## Decision

`approved`

## Source handoff

Manager summary of security review: probe hardening, local-origin CORS narrowing, action
confirmation, and a zero-vulnerability frontend lockfile leave no blocking security finding for
this slice.
Runtime proof: same `npm audit`, pytest, Playwright E2E, and MCP live-browser evidence cited above.
