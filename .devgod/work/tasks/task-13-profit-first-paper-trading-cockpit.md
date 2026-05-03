# Task ID

`task-13-profit-first-paper-trading-cockpit`

## Owner role

`backend_engineer`

## Goal

Turn `bitbat_v2` into a profit-seeking, paper-only BTC trading cockpit by replacing label-centric
success criteria with net trading performance and by adding a paper trading view that shows
portfolio state, fills, equity, drawdown, and runtime freshness to the operator.

## Current gate state

`complete`

## Inputs

- approved `brief-2026-05-03-profit-first-paper-trading`
- completed `task-11-autonomous-paper-execution`
- completed `task-12-deterministic-strategy-improvement`
- current v2 runtime is paper-only but strategy/evaluation framing is still conservative and not
  productized around net profitability
- legacy autonomous prediction surfaces remain available and should be treated as diagnostic-only

## Outputs

- profit-oriented evaluation contract for v2 including fees, slippage, drawdown, and benchmark
  comparison
- append-only paper ledger and read models for fills, trades, equity curve, and performance
- API endpoints for paper portfolio and performance inspection
- dedicated operator paper trading view with portfolio, PnL, exposure, order history, and runtime
  freshness
- tests proving PnL math, idempotency, read-model accuracy, and UI visibility

## Gate decision

`complete`

## Scope boundaries

- v2 only
- paper only
- legacy `data/autonomous.db` read-only
- no live exchange order APIs

## Planned workstreams

1. metric contract
   - define profit-first KPIs:
     - net paper PnL after fees/slippage
     - equity curve
     - max drawdown
     - exposure
     - expectancy per trade
     - buy-and-hold delta
2. execution and ledger model
   - formalize signal -> sizing -> fill -> trade -> portfolio -> performance flow
   - add append-only paper ledger and deterministic projections
   - preserve duplicate-candle and pause protections
3. API surface
   - add `/v1/paper` and `/v1/performance` style endpoints in the v2 operator API
   - expose fills, trades, equity, benchmarks, and freshness
4. operator view
   - add a desktop-first paper trading view
   - include command/status header, PnL/exposure tiles, signal/portfolio workbench, orders
     timeline, live event ledger, and alert log
   - make mobile watch-only, not full-operator parity
5. verification and gates
   - ledger economics tests
   - integration tests for idempotency and DB boundaries
   - UI smoke coverage for paper dashboard
   - QA and security review gates before implementation is called complete

## Dependencies

- `task-11-autonomous-paper-execution`
- `task-12-deterministic-strategy-improvement`

## Allowed write scope

- `src/bitbat_v2/`
- `tests/v2/`
- `streamlit/`
- `.devgod/work/tasks/task-13-profit-first-paper-trading-cockpit.md`
- `.devgod/work/briefs/brief-2026-05-03-profit-first-paper-trading.md`
- `.devgod/work/plans/plan-2026-04-25-bitbat-clean-room.md`

## Out of scope

- live-money execution
- exchange trading credentials
- broker auth flows
- discretionary agent trading
- multi-venue support
- tax/report exports

## Acceptance criteria

- v2 evaluates trading quality using realized paper performance after costs, not only directional
  correctness
- v2 persists fills, trades, equity, and portfolio state as auditable paper-trading records
- operators can inspect current paper balance, open position, net PnL, drawdown, order count,
  recent fills, and runtime freshness in one view
- strategy comparisons include buy-and-hold and cash baselines
- legacy diagnostic prediction paths remain isolated and produce no side effects on v2 execution

## Verification steps

- run targeted unit tests for signal sizing, fill math, realized/unrealized PnL, and projections
- run targeted integration tests for duplicate handling, idempotent ledger writes, and v2-only DB
  mutation
- run dashboard smoke tests for the paper trading view and operator status presentation
- confirm no writes occur outside the v2 paper store during the new flow

## Required reviews

- reviewer
- security_reviewer
- qa_engineer
- frontend_designer

## Security checks

- keep the system visibly and technically paper-only
- reject any live-trading keys or real order endpoints in this sprint
- preserve kill switch and operator pause behavior
- enforce append-only audit semantics for fills and performance history
- validate inputs on any new API or control surface

## QA notes

- test economics, not just rendering
- verify fees and slippage are included in reported net PnL
- verify stale or duplicate runtime states cannot create false-positive performance displays
- verify legacy diagnostics remain read-only

## Anti-patterns to avoid

- using label accuracy as the primary shipped KPI
- showing a green dashboard backed by incorrect ledger math
- hiding paper-only status behind trading-themed theatrics
- mixing legacy autonomous DB writes into the v2 execution path

## Rollback notes

- keep the current v2 operator path available behind the existing paper-only controls
- disable any new evaluator or view routes if ledger or projection correctness is not proven

## Handoff format

- metric contract
- ledger schema
- API changes
- UI panels
- verification evidence
- remaining gaps

## Verification evidence

- focused regression:
  - `poetry run pytest tests/v2/test_api.py tests/v2/test_paper.py tests/v2/test_runtime.py tests/v2/test_storage.py tests/v2/test_streamlit_paper_view.py -q`
  - passed
- full v2 suite:
  - `poetry run pytest tests/v2 -q`
  - `41 passed`
- supported Streamlit smoke:
  - `poetry run pytest tests/gui/test_phase12_supported_views_smoke.py -q`
  - `5 passed`
- lint:
  - `poetry run ruff check src/bitbat_v2 streamlit/pages/2_📈_Performance.py tests/v2`
  - passed
