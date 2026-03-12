---
slug: monitor-runtime-pair-mismatch-v1-3
created: 2026-02-26
status: root_cause_found
owner: gsd-debug
---

# Debug Session: Monitor runtime pair mismatch after v1.3

## Symptoms
- Expected: `scripts/run_monitoring_agent.py` starts monitoring loop.
- Actual: startup aborts during `MonitoringAgent` initialization.
- Error:
  - `FileNotFoundError: Missing monitor model artifact for resolved runtime pair 5m/30m: models/5m_30m/xgb.json`
- Timeline: observed after v1.3 milestone completion.
- Reproduction: run monitor agent without explicit config/env override in environment where only `models/1h_4h` or other non-`5m_30m` artifacts exist.

## Hypothesis
The runtime config source resolves `freq/horizon` to `5m/30m`, but deployed model artifacts are for a different pair.

## Evidence
1. `scripts/run_monitoring_agent.py` loads runtime config via `set_runtime_config(args.config)`, then uses config `freq/horizon` for `MonitoringAgent`.
2. `src/bitbat/autonomous/agent.py` now fail-fast validates model path: `models/{freq}_{horizon}/xgb.json`.
3. Default config (`src/bitbat/config/default.yaml`) sets `freq: "5m"`, `horizon: "30m"`.
4. Local model inventory includes `models/1h_4h/xgb.json` and `models/1h_1h/xgb.json`, but not `models/5m_30m/xgb.json`.
5. Reproduction check:
   - default config -> `5m/30m` -> model missing
   - `config/user_config.yaml` -> `1h/1h` -> model exists

## Root Cause
This is a runtime configuration and model-artifact pair mismatch, now surfaced intentionally by v1.3 startup guardrails.

## Fix Options
1. Set monitor config to a pair that has an artifact (recommended):
   - run with `--config <path>` or set `BITBAT_CONFIG=<path>`.
2. Train/copy a model artifact for `5m/30m` at `models/5m_30m/xgb.json`.

## Suggested Immediate Remediation
- Ensure monitor startup passes a config with matching `freq/horizon` and existing artifact.
- For service/container startup, set `BITBAT_CONFIG` explicitly and/or pass `--config` in ExecStart/CMD.

