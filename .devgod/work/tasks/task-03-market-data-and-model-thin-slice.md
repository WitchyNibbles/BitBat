# Task ID

`task-03-market-data-and-model-thin-slice`

## Owner role

`backend_engineer`

## Goal

Implement one production-shaped prediction stream using Coinbase BTC-USD candles and a deterministic
signal generator suitable for paper trading.

## Inputs

- v2 architecture
- Coinbase public candle docs

## Outputs

- Coinbase adapter
- candle ingest and signal generation path
- latest signal read model

## Dependencies

- task 02

## Allowed write scope

- `src/bitbat_v2/`
- `tests/v2/`
- `.devgod/work/tasks/task-03-market-data-and-model-thin-slice.md`
- `.devgod/work/reviews/review-03-task-03.md`

## Out of scope

- complex model training pipelines
- multi-venue market aggregation

## Acceptance criteria

- v2 can ingest a candle and emit `candle.closed`, `features.computed`, and `signal.generated`
- latest signal is queryable via API
- market data adapter is tested with mocked transport

## Verification steps

- run v2 runtime and API tests

## Required reviews

- reviewer
- qa_engineer

## Security checks

- validate external API inputs before persistence

## Anti-patterns to avoid

- hidden network dependency in tests
- non-deterministic signal output

## Rollback notes

- remove adapter and signal modules if venue choice changes

## Handoff format

- behavior summary, tests, follow-up risks
