# Monitor Operations Runbook

This runbook defines the supported monitor startup wiring and the operator diagnostics flow for v1.3.

## Supported Config Wiring

Use one of these supported config paths so monitor runtime pair resolution is explicit:

1. CLI override with `--config`:
   ```bash
   poetry run bitbat --config config/monitor.yaml monitor run-once --freq 1h --horizon 4h
   poetry run bitbat --config config/monitor.yaml monitor start --freq 1h --horizon 4h --interval 300
   ```
2. Environment override with `BITBAT_CONFIG`:
   ```bash
   export BITBAT_CONFIG=config/monitor.yaml
   poetry run bitbat monitor run-once --freq 1h --horizon 4h
   poetry run bitbat monitor status --freq 1h --horizon 4h
   ```

`--config` takes precedence when both are set.

## Service Mode Contract

Service deployments should keep `BITBAT_CONFIG` explicit and pass the same value through `--config`.
The canonical template is [`deployment/bitbat-monitor.service`](../deployment/bitbat-monitor.service).

## Startup Guardrail Triage

If monitor startup is blocked because model artifacts are missing for the resolved runtime pair:

1. Read the startup error. It includes the missing `models/{freq}_{horizon}/xgb.json` path.
2. Confirm config wiring (`--config` or `BITBAT_CONFIG`) points to the intended `freq/horizon`.
3. Bootstrap or train/copy the missing artifact for that pair.
   ```bash
   poetry run python scripts/bootstrap_monitor_model.py --config config/monitor.yaml --start 2026-01-01
   # or
   make bootstrap-monitor-model CONFIG=config/monitor.yaml START=2026-01-01
   ```
4. Re-run `monitor run-once` before returning to long-running mode.

## Cycle-State Diagnostic Interpretation

For each cycle, treat these fields as the source of truth:

- `prediction_state`
- `prediction_reason`
- `realization_state`
- `cycle diagnostic`

Typical operator interpretation:

- `prediction_state=generated` and `realization_state=pending`: prediction created, waiting for horizon realization.
- `prediction_state=none` with `prediction_reason=insufficient_data`: no prediction this cycle due to input insufficiency.
- `prediction_state=none` with `prediction_reason=missing_model`: startup/config/runtime pair mismatch or missing artifact.

## Status and Snapshot Diagnostics

Use `monitor status` for pair-scoped lifecycle counts:

```bash
poetry run bitbat --config config/monitor.yaml monitor status --freq 1h --horizon 4h
```

Expected fields include total, unrealized, and realized predictions for the active pair.

## Schema Remediation Flow

If monitor commands report schema incompatibility, run schema remediation before restarting services:

```bash
poetry run python scripts/init_autonomous_db.py --database-url "sqlite:///data/autonomous.db" --audit
poetry run python scripts/init_autonomous_db.py --database-url "sqlite:///data/autonomous.db" --upgrade
```

Re-run `monitor run-once` and `monitor status` after remediation.

## Release Verification

Before release or deploy promotion, run:

```bash
make test-release
```

This includes the canonical D1 monitor guard suites plus release contract checks.
