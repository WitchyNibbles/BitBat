# Release Summary

`release-2026-04-25-autonomous-paper-execution`

## Scope

Add autonomous paper-mode execution to BitBat v2 without enabling live money, cutting over, or
changing legacy defaults.

## Completed

- added an autonomous paper trader loop behind config flags
- added duplicate-candle suppression for repeated Coinbase syncs
- exposed autorun status in `/v1/health`
- kept manual controls intact
- passed `tests/v2`

## Current posture

- v2 can now auto-poll and auto-trade in paper mode when explicitly enabled
- live-money trading remains disabled
- legacy services remain intact
- profitability is still limited by the current heuristic model and was not solved by this slice

## Operator note

- to review autonomous behavior, start the v2 API with:
  - `BITBAT_V2_AUTORUN_ENABLED=true`
  - optionally set `BITBAT_V2_AUTORUN_INTERVAL_SECONDS`
