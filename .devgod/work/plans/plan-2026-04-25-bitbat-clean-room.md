# BitBat Clean-Room Rebuild

## Summary

Build a separate `bitbat_v2` runtime for a witchy, single-operator BTC spot paper-trading cockpit.
Keep legacy surfaces available while the new API, runtime, and dashboard mature behind explicit
review gates.

## Execution order

1. intake and product brief
2. clean-room architecture and data model
3. market-data plus model thin slice
4. deterministic strategy, risk, and paper broker
5. operator dashboard and control plane
6. shadow run, cutover, and deprecation planning
7. immediate operational cleanup
8. optional environment hardening
9. future cutover preparation
10. legacy deprecation prerequisites
11. autonomous paper execution
12. deterministic strategy improvement and offline evaluation

## Bounded architecture

- `domain`: candles, features, signals, decisions, orders, portfolio, alerts, controls
- `services`: ingestion, signal generation, strategy, risk, paper execution, event projection
- `adapters`: Coinbase market data, SQL persistence, REST/SSE API, React client
- `control`: pause, resume, retrain, reset paper account, acknowledge alerts, simulate candle

## Runtime defaults

- venue: Coinbase BTC-USD spot
- execution: paper only
- operator model: solo owner
- storage: dedicated v2 runtime database with append-only events and read models
- UI: React operator console as the long-term surface

## Review policy

- every task must have a review artifact
- any `blocked` or `critical` review stops promotion
- required reviewers by task:
  - all tasks: reviewer
  - tasks 02, 04, 05, 06: security_reviewer
  - tasks 03, 04, 05, 06: qa_engineer
  - task 05: frontend_designer
  - tasks 02, 06: infra_engineer

## Done bar

- v2 tests pass for touched areas
- API and dashboard work end to end for the thin slice
- review gates are recorded
- remaining gaps are documented in task 06 handoff material

## Post-task-06 operational phase

### 7. immediate operational cleanup

- gate state: `ready`
- purpose:
  - close out the approved shadow session cleanly
  - convert task-06 evidence into a stable operator runbook and watchlist
  - decide the fate of any still-running shadow processes
- current decision:
  - ad hoc shadow-session processes on ports `8100` and `5173` should be shut down after evidence
    capture unless a named operator is actively running a time-boxed soak

### 8. optional environment hardening

- gate state: `optional`
- purpose:
  - replace temporary machine-specific verification workarounds with repeatable environment setup
  - make browser verification reproducible on a clean host
- current decision:
  - the temporary `/tmp` `libasound.so.2` workaround is not the long-term baseline and should be
    replaced by a machine-level dependency fix or equivalent reproducible browser runtime layer

### 9. future cutover preparation

- gate state: `blocked`
- purpose:
  - define cutover go/no-go criteria, soak expectations, rollback requirements, and operator sign-off
- current decision:
  - planning is allowed
  - actual cutover remains blocked
  - live-money trading remains blocked

### 10. legacy deprecation prerequisites

- gate state: `hold`
- purpose:
  - define the evidence required before any legacy route or service is retired
- current decision:
  - legacy services stay intact
  - no deprecation or shutdown work begins until a future approved cutover phase is complete

### 11. autonomous paper execution

- gate state: `ready`
- purpose:
  - make v2 capable of autonomous paper-only market polling and decision execution
  - prevent duplicate processing of the same Coinbase candle
  - expose enough runtime status for the operator to verify the loop is active
- current decision:
  - paper-only autonomous execution is allowed
  - live-money trading remains blocked
  - legacy services remain untouched

### 12. deterministic strategy improvement and offline evaluation

- gate state: `complete`
- purpose:
  - improve paper-trading decision quality without changing trust boundaries
  - keep strategy decisions deterministic and explainable
  - provide a reproducible local comparison between the old heuristic and the improved strategy
- current decision:
  - `filtered_momentum_v2` is the default v2 paper strategy
  - `baseline_v1` remains available for offline comparison only
  - local evaluation evidence exists for `data/raw/prices/btcusd_yf_5m.parquet`
  - live-money trading remains blocked
  - legacy services remain untouched
