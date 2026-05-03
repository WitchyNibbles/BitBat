# Intake Brief

## Brief ID

`brief-2026-05-03-profit-first-paper-trading`

## Request

Original user ask:

Create a new sprint to change the system so it seeks profit, and make a paper trading view so the
user can see how it is doing.

## Goal

Shift BitBat's active operator path from label-centric prediction diagnostics toward profit-seeking,
paper-only BTC trading in `bitbat_v2`, with an operator view that exposes portfolio state,
execution, and performance clearly.

## Audience

- primary: single operator using the local BitBat cockpit
- secondary: engineering maintaining the clean-room `bitbat_v2` runtime

## Constraints

- keep scope inside `bitbat_v2`
- keep live-money trading disabled
- treat legacy `data/autonomous.db` and older prediction surfaces as diagnostic-only
- preserve explicit paper-only labeling across API and UI
- include transaction-cost realism: fees, slippage, and benchmark comparison
- no new broker auth or exchange order APIs in this sprint

## Risks

- backtest and paper-profit overfit can create false confidence
- UI can look healthy while ledger math or idempotency is wrong
- live/paper confusion can create operator trust and safety problems
- KPI drift between legacy prediction metrics and v2 profit metrics can confuse users
- hidden writes outside the v2 store can break boundary assumptions

## Unknowns

- whether the current deterministic v2 strategy is strong enough once fees and slippage are applied
- whether sizing should be fixed, score-proportional, or risk-budgeted in the first sprint
- whether the paper view should live in Streamlit only or also be exposed through the v2 API for a
  later React cockpit
- what exact historical ledger backfill, if any, is needed for existing v2 sessions

## Success criteria

- v2 emits trade actions and sizing using profit-oriented evaluation rules instead of accuracy-only
  direction labels
- v2 stores an auditable paper ledger with fills, trades, portfolio state, equity, and performance
  read models
- operators can see net paper PnL after costs, exposure, drawdown, order history, and runtime
  freshness in a dedicated paper trading view
- evaluation compares strategy outcomes against cash and buy-and-hold, not only hit rate
- legacy diagnostic paths remain read-only and isolated from v2 paper execution

## Out of scope

- live-money execution
- real broker connectivity or order placement
- multi-asset expansion
- tax, accounting, or reporting workflows
- optimizing for raw classification accuracy as the primary product goal

## Trust boundaries

- `bitbat_v2` database is the only mutable source of truth for paper trading
- legacy autonomous prediction data is read-only diagnostic input
- market data remains external and untrusted until validated at ingestion boundaries
- operator controls remain authenticated and must preserve pause/kill-switch semantics

## Stop/go

`go`

## Next step

Planner action required:

Create a new v2-only sprint packet covering metric contract, append-only paper ledger, profit
evaluator/API, paper trading view, and QA/security gates before implementation begins.
