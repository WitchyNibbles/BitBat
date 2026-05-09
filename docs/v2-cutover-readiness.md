# BitBat v2 Cutover Readiness

## Purpose

Define the release bar for making `bitbat_v2` the only default operator runtime and for keeping
legacy execution off by default.

## Canonical Runtime

- Primary runtime: `bitbat_v2`
- Primary paper ledger: `data/bitbat_v2.db`
- Legacy runtime role: diagnostic-only API and historical monitoring data
- Legacy active execution role: opt-in only via `legacy` compose profile or
  `BITBAT_LEGACY_SERVICES_ENABLED=true`

## Cutover Criteria

All items below must be true before calling the cutover complete:

1. `bitbat_v2` starts as the default runtime from the container entrypoint and Docker Compose.
2. Legacy ingestion and monitoring are disabled by default.
3. The React dashboard default operator path points at the v2 API.
4. `make lint` passes.
5. Targeted v2 API, runtime, evaluation, and contract tests pass.
6. Replay proof exists for the legacy-ML bridge inside `bitbat_v2`.
7. Replay proof shows one of:
   - real buy/sell paper activity with correct ledger updates, or
   - a justified no-trade explanation from the active signal model.
8. Duplicate-candle protection is verified.
9. The operator health surface reports signal source and model provenance.
10. Workflow review artifacts exist for `reviewer`, `qa_engineer`, and `security_reviewer`.

## Current Status On 2026-05-09

- `bitbat_v2` is the default entrypoint runtime.
- Legacy ingestion and monitoring are now opt-in through the `legacy` compose profile.
- Lint and targeted v2 tests pass on the remediation branch.
- The legacy-ML bridge runs inside `bitbat_v2`, persists `legacy_xgb_5m_30m` provenance, and
  rejects duplicate candles correctly.
- The recent local replay window currently produces a justified no-trade outcome:
  the model predicts `class=flat` with high `p_flat`, so the runtime records signals and
  portfolio events but does not place orders.
- Cold-start latency for the first legacy-ML replay remains high and must be watched as a
  readiness risk.

## Rollback

If the v2 default path is unhealthy:

1. Keep `bitbat_v2` stopped.
2. Run the legacy API directly:
   `BITBAT_PRIMARY_API=legacy ./scripts/start.sh`
3. If legacy execution must also resume, opt in explicitly:
   `BITBAT_LEGACY_SERVICES_ENABLED=true BITBAT_PRIMARY_API=legacy ./scripts/start.sh`
4. For Docker Compose, enable the legacy profile:
   `docker compose --profile legacy up --build`
5. Preserve `data/bitbat_v2.db` for postmortem analysis; do not overwrite it during rollback.

## Operator Notes

- A no-trade replay is not a runtime failure by itself if the model is explicitly classifying the
  market as flat.
- It is a cutover blocker if the project goal requires active short-span opportunity capture and
  the current model never produces actionable signals on representative local data.
