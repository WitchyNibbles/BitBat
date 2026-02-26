---
phase: 15-cost-aware-walk-forward-evaluation
plan: "02"
subsystem: evaluation
tags: [backtest, fee-model, slippage, net-metrics, walk-forward]
requires:
  - phase: 15-cost-aware-walk-forward-evaluation
    provides: Purge/embargo-aware fold scheduling and leakage-safe split controls from Plan 15-01
provides:
  - Backtest engine fee/slippage attribution with explicit cost components
  - Fold-level and aggregate net/gross metrics with machine-readable cost breakdowns
  - CLI cost-control flags and operator-facing net-aware backtest summaries
affects: [backtest-engine, walk-forward-evaluation, cli-reporting, config, v1.2-phase15]
tech-stack:
  added: []
  patterns:
    - Net and gross metrics are emitted together with explicit fee/slippage attribution
    - CLI backtest output mirrors persisted cost-attribution schema for auditability
key-files:
  created: []
  modified:
    - src/bitbat/backtest/engine.py
    - src/bitbat/backtest/metrics.py
    - src/bitbat/model/walk_forward.py
    - src/bitbat/cli.py
    - src/bitbat/config/default.yaml
    - tests/backtest/test_engine.py
    - tests/backtest/test_metrics.py
    - tests/model/test_walk_forward.py
    - tests/test_cli.py
key-decisions:
  - "Kept backward compatibility by treating legacy cost_bps as fee-only when explicit fee/slippage controls are not provided."
  - "Standardized cost-aware reporting on net_sharpe/gross_sharpe with fee and slippage totals in the same payload."
patterns-established:
  - "Backtest outputs must expose fee_costs and slippage_costs columns alongside total costs."
  - "Walk-forward summaries must include total_fee_costs and total_slippage_costs when cost metrics are present."
requirements-completed: [EVAL-02]
duration: 4 min
completed: 2026-02-26
---

# Phase 15 Plan 02: Cost-Aware Evaluation Metrics Summary

**Backtest and walk-forward evaluation now report realistic net metrics with explicit fee/slippage attribution and CLI-visible cost assumptions.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-26T07:14:50Z
- **Completed:** 2026-02-26T07:18:55Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Added explicit fee/slippage cost decomposition in the backtest engine while preserving legacy cost-only behavior.
- Propagated net/gross return and fee/slippage attribution fields into backtest summaries and walk-forward aggregates.
- Exposed fee/slippage controls in CLI backtest runs and printed net-aware operator output with cost breakdowns.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend backtest engine with explicit fee and slippage cost modeling** - `6f66057` (feat)
2. **Task 2: Propagate cost-aware metrics into summary and fold-level outputs** - `e0200f5` (feat)
3. **Task 3: Surface cost-aware controls and reporting through CLI/config defaults** - `89d71c3` (feat)

## Files Created/Modified

- `src/bitbat/backtest/engine.py` - fee/slippage controls and attributed cost columns.
- `src/bitbat/backtest/metrics.py` - net/gross return plus total fee/slippage summary fields.
- `src/bitbat/model/walk_forward.py` - fold-level fee/slippage totals and gross-return aggregation.
- `src/bitbat/cli.py` - backtest fee/slippage flags and net-aware report output.
- `src/bitbat/config/default.yaml` - default fee/slippage settings for reproducible runs.
- `tests/backtest/test_engine.py` - fee/slippage attribution regression test.
- `tests/backtest/test_metrics.py` - summary coverage for total fee/slippage and gross return fields.
- `tests/model/test_walk_forward.py` - walk-forward cost-attribution summary regression test.
- `tests/test_cli.py` - CLI backtest output/arg passthrough regression test.

## Decisions Made

- Preserved existing defaults by mapping legacy `cost_bps` to fee cost when explicit components are not supplied.
- Emitted net and gross metrics together to keep optimistic and realistic performance views comparable.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- EVAL-02 cost-aware evaluation requirements are now implemented with deterministic attribution outputs.
- Phase 15 Plan 03 can build champion selection/reporting directly on these cost-aware aggregate metrics.

## Self-Check: PASSED

- `poetry run pytest tests/backtest/test_engine.py -q -k "cost or slippage or net or gross"` -> 4 passed, 3 deselected
- `poetry run pytest tests/backtest/test_metrics.py tests/model/test_walk_forward.py -q -k "net or gross or cost or sharpe"` -> 2 passed, 15 deselected
- `poetry run pytest tests/test_cli.py -q -k "backtest and (cost or slippage or net or gross)"` -> 1 passed, 19 deselected

---
*Phase: 15-cost-aware-walk-forward-evaluation*
*Completed: 2026-02-26*
