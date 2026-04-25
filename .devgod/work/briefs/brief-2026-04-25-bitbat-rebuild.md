# Brief ID

`brief-2026-04-25-bitbat-rebuild`

## Request

Original user ask:

Rebuild BitBat from scratch as a witchy-themed application that predicts Bitcoin prices in real
time so autonomous agents can make buy/sell transactions that generate income, while the operator
can watch the system in real time on a dashboard.

## Goal

Ship a clean-room `bitbat_v2` runtime that supports a single-operator BTC spot trading cockpit with
real-time prediction, deterministic strategy execution, paper trading, and live dashboard updates.

## Audience

- primary: solo operator monitoring and controlling BitBat
- secondary: future Codex/devgod implementation threads extending the v2 slice

## Constraints

- keep legacy `src/bitbat/`, `streamlit/`, and the existing dashboard surface available during the
  rebuild
- new runtime must live under `src/bitbat_v2/`
- v1 execution is paper trading only
- first market scope is BTC spot on one venue
- agents orchestrate work; strategy decisions stay deterministic and rule-bound
- devgod artifacts under `.devgod/work/` are the operational source of truth

## Risks

- legacy runtime overlap can create accidental scope bleed into v2
- real-time trading language creates safety pressure toward live-money behavior before controls are
  mature
- current dashboard/API split makes operator trust fragile unless a single v2 control surface is
  clear
- exchange adapters and event streaming introduce runtime and persistence failure modes

## Unknowns

- final production venue credentials and live order flow are intentionally deferred
- full model selection strategy beyond the thin-slice deterministic signal generator remains open
- shadow-run acceptance thresholds still need refinement during later hardening

## Success criteria

- `bitbat_v2` exposes `/v1/health`, `/v1/portfolio`, `/v1/signals/latest`, `/v1/orders`,
  `/v1/control/*`, and `/v1/stream/events`
- v2 persists append-only events plus read models in its own runtime database
- the operator can watch predictions, decisions, fills, PnL, and alerts from a witchy React console
- trading pause, paper-account reset, retrain request, and alert acknowledgement are operator
  actions with tests
- review gates exist for every task packet and block unsafe promotion

## Out of scope

- live-money exchange execution
- multi-tenant auth, billing, or SaaS account management
- derivatives, leverage, or multi-venue routing
- decommissioning every legacy surface in this implementation pass

## Trust boundaries

- external market data from Coinbase public APIs is untrusted input
- model output is advisory input to deterministic strategy rules, not direct execution authority
- operator control commands are trusted only after validation at the v2 API boundary
- devgod shared core stores planning/orchestration state only, not trading runtime state

## Stop/go

`go`

Live-money execution remains `stop` pending a separate approval plan.

## Current status

- tasks `01` through `06` are complete
- `task-06` and `review-06` are approved
- BitBat v2 shadow-run evidence exists and is documented
- `task-07` immediate operational cleanup is complete and the residual shadow session was shut down
- `task-08` optional environment hardening is planned with a concrete Ubuntu host fix
- `task-09` future cutover preparation is captured as planning-only and remains execution-blocked
- `task-10` legacy deprecation prerequisites are captured and remain on hold
- `task-11` autonomous paper execution is complete for v2
- `task-12` deterministic strategy improvement and offline evaluation is complete for v2
- v2 remains paper-only
- legacy services remain the default runtime and are intentionally unchanged
- no cutover or legacy deprecation work is approved for execution in this phase

## Next step

Planner action required:

No required execution remains in the post-task-06 phase.

Optional follow-up:

- `task-08-optional-environment-hardening`

Future gated work:

- `task-09-future-cutover-preparation`
- `task-10-legacy-deprecation-prerequisites`

These remain blocked or on hold until later approved phases.
