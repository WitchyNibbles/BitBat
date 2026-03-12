# Phase 36: Live Recovery Evidence Closure - Research

**Researched:** 2026-03-12
**Domain:** fresh post-reset recovery evidence for FIXR-03
**Confidence:** HIGH

## Summary

Phase 30 fixed the code-path defects behind the live accuracy collapse, but its saved verification artifact stayed `gaps_found` because the repo still contained stale pre-fix `data/autonomous.db` rows and no fresh post-reset evidence was captured. The milestone audit reopened this as Phase 36.

The smallest credible closure path is:

1. Keep the existing operator-facing reset command (`bitbat system reset --yes`).
2. Add a deterministic recovery-evidence harness that can operate in a sandboxed config without mutating the repo's current `data/`.
3. Train against a training split only, then realize predictions on a held-out split so the evidence is not a same-data self-check.
4. Re-run the Phase 30 diagnosis assertions against the fresh sandbox artifacts and save the passing evidence in planning docs.

## What the codebase supports already

- `src/bitbat/cli/commands/system.py` already provides `bitbat system reset --yes`.
- `src/bitbat/cli/commands/model.py` already provides `bitbat model train`.
- `src/bitbat/autonomous/validator.py` can realize predictions from parquet price history into `prediction_outcomes`.
- `tests/diagnosis/test_pipeline_stage_trace.py` already encodes the four Phase 30 recovery assertions, but it is hardcoded to `models/5m_30m/xgb.json` and `data/autonomous.db`.

## Key gap discovered during research

The reset/retrain flow is not fully config-consistent today:

- `system reset` resolves `models_dir` from config.
- `model train` persists via `src/bitbat/model/persist.py`, which still defaults to the literal `"models"` root unless an explicit root is passed.

That mismatch breaks the sandboxed recovery path because reset can delete one models directory while training writes into another. Phase 36 must close that gap before the documented flow is real.

## Data choice

`data/features/1h_1h/dataset.parquet` is the strongest deterministic recovery-evidence source:

- it contains `up`, `down`, and `flat` labels
- a quick probe showed a fresh retrain on a train/holdout split yields accuracy above the 33% random baseline
- the `1h_1h` horizon lets us reconstruct a synthetic close series directly from `r_forward`

`5m_30m` is a poor candidate because the current dataset has no `flat` labels, which makes the original Phase 30 direction-balance diagnosis contract unreliable for closure evidence.

## Recommended architecture

### 1. Recovery evidence module

Add a reusable module that can:

- split a source dataset into train/eval partitions
- write the train partition into the configured `data_dir`
- reconstruct synthetic price history for the eval partition
- load the trained model artifact
- store eval predictions in a fresh autonomous DB
- run `PredictionValidator.validate_all()`
- emit a machine-readable evidence summary with counts and realized accuracy

### 2. Scripted operator harness

Add a script under `scripts/` so the documented flow is reproducible without ad hoc manual file editing:

- `stage` step: write train/eval datasets into the sandbox config layout
- `realize` step: generate fresh realized outcomes and summary evidence after `bitbat model train`

### 3. Diagnosis test path resolution

Update `tests/diagnosis/test_pipeline_stage_trace.py` to resolve:

- freq/horizon from runtime config
- model path from the configured models root
- DB path from `autonomous.database_url`

This lets the same Phase 30 diagnosis assertions validate fresh Phase 36 evidence under `BITBAT_CONFIG=<sandbox-config>`.

## Validation strategy

- Add a behavioral test for the recovery-evidence builder on a temporary sandbox.
- Add a regression test proving model persistence honors configured `models_dir`.
- Add a docs contract test for the operator recovery runbook path.
- Execute the real flow:
  - `bitbat system reset --yes`
  - recovery-evidence `stage`
  - `bitbat model train`
  - recovery-evidence `realize`
  - `pytest tests/diagnosis/test_pipeline_stage_trace.py -v` under the sandbox config

## Risks

- Diagnosis assertions remain pair-sensitive; the evidence flow must explicitly use `1h_1h`.
- Synthetic price reconstruction is only safe if timestamps are monotonic and the evaluation horizon is compatible with the chosen pair.
- Planning docs are gitignored, so new Phase 36 artifacts must be force-added when committed.
