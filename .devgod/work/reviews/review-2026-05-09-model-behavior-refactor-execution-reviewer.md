# Review Gate

## Task ID

`2026-05-09-model-behavior-refactor-execution`

## Reviewer role

`reviewer`

## Actor

`019e0be9-c5db-7851-acbf-7c898d507874`

## Actor role

`reviewer`

## Provenance status

`runtime_verified`

## Review state

`passed`

## Severity

`low`

## Specialist execution evidence

- Backend slice implemented in:
  - [signals.py](/home/eimi/projects/ai-btc-predictor/src/bitbat_v2/signals.py:1)
  - [runtime.py](/home/eimi/projects/ai-btc-predictor/src/bitbat_v2/runtime.py:1)
  - [domain.py](/home/eimi/projects/ai-btc-predictor/src/bitbat_v2/domain.py:1)
  - [storage.py](/home/eimi/projects/ai-btc-predictor/src/bitbat_v2/storage.py:1)
  - [schemas.py](/home/eimi/projects/ai-btc-predictor/src/bitbat_v2/api/schemas.py:1)
- Focused regression tests were added for provider evidence, buy-hold EV gating, sell-hold EV
  gating, storage round-trip, and legacy-schema migration.

## Quality gate evidence

- `poetry run pytest tests/v2 -q`
- `make lint`
- `poetry run pytest tests/contracts/test_runtime_unification_contract.py tests/gui/test_v2_truthfulness_copy.py -q`

## Findings

- No blocking correctness or regression findings remained after the final patch set.

## Residual risk

- `legacy_ml` remains a trusted model-input path selected by config; that trust boundary was not
  reduced in this slice.
- This slice changes decision gating, not the underlying training objective.

## Verification evidence

- Reviewer handoff reported no blocking findings.
- Final branch-state evidence:
- Runtime proof: `poetry run pytest tests/v2 -q` -> `55 passed`
- Runtime proof: `make lint` -> passed
- Runtime proof: `poetry run pytest tests/contracts/test_runtime_unification_contract.py tests/gui/test_v2_truthfulness_copy.py -q` -> `10 passed`

## Waiver authority

`none`

## Waiver reason

None.

## Decision

`approved`

## Source handoff

Manager summary of reviewer output plus final branch-state command evidence. The scoped signal
evidence/runtime gating slice has no blocking correctness findings.
Runtime proof: final branch state passed `tests/v2`, lint/type-check, and the runtime-unification
contract/UI truthfulness checks.

## 2026-05-09 Addendum

- Mode-specific training profiles now select family and label contract per preset instead of
  routing every preset through one implicit `xgb` classifier.
- Artifact metadata and runtime inference now agree on family and label mode, including
  `triple_barrier` classifier outputs and `random_forest` return regressors.
- Reviewer re-check evidence for this addendum:
  - `poetry run pytest tests/autonomous/test_predictor_operability.py tests/autonomous/test_agent_integration.py tests/model/test_persist.py tests/model/test_train.py tests/model/test_infer.py tests/autonomous/test_orchestrator.py tests/api/test_predictions.py tests/test_config_loader.py tests/gui/test_performance_helpers.py tests/gui/test_timeline.py -q` -> `118 passed`
  - `poetry run ruff check src/bitbat/model/mode_profiles.py src/bitbat/model/persist.py src/bitbat/model/train.py src/bitbat/model/infer.py src/bitbat/autonomous/orchestrator.py src/bitbat/autonomous/predictor.py tests/model/test_persist.py tests/model/test_train.py tests/model/test_infer.py tests/autonomous/test_orchestrator.py` -> passed

## 2026-05-09 Selection Addendum

- Added walk-forward candidate selection per preset in `src/bitbat/model/mode_selection.py`.
- `one_click_train` now records winner-family, candidate reports, and champion decision in active
  model metadata instead of assuming the profile default family won.
- Runtime predictor now resolves artifact family from active model metadata before falling back to
  preset defaults.
- Reviewer re-check evidence for this addendum:
  - `poetry run pytest tests/model/test_mode_selection.py tests/model/test_persist.py tests/model/test_train.py tests/model/test_infer.py tests/autonomous/test_orchestrator.py tests/autonomous/test_predictor_operability.py tests/autonomous/test_agent_integration.py tests/api/test_predictions.py tests/test_config_loader.py tests/gui/test_performance_helpers.py tests/gui/test_timeline.py -q` -> `122 passed`
  - `poetry run ruff check src/bitbat/model/mode_profiles.py src/bitbat/model/mode_selection.py src/bitbat/model/persist.py src/bitbat/model/train.py src/bitbat/model/infer.py src/bitbat/autonomous/orchestrator.py src/bitbat/autonomous/predictor.py tests/model/test_mode_selection.py tests/model/test_persist.py tests/model/test_train.py tests/model/test_infer.py tests/autonomous/test_orchestrator.py tests/autonomous/test_predictor_operability.py` -> passed

## 2026-05-09 Runtime Policy Addendum

- Repaired the `bitbat_v2` runtime contract so `legacy_ml` can import and execute again:
  restored `signal_source` / `legacy_signal_freq` / `legacy_signal_horizon`, added the
  classification compatibility helper, and mapped the legacy heuristic strategy alias.
- `LegacyModelSignalProvider` now uses metadata-aware inference, accepts `triple_barrier`
  side models for tradable modes, resolves `random_forest` winner artifacts for regression-first
  presets, and holds signals when artifact `action_policy` thresholds reject confidence or expected
  value.
- Reviewer re-check evidence for this addendum:
  - `poetry run pytest tests/v2 tests/model/test_infer.py tests/autonomous/test_predictor_operability.py tests/autonomous/test_orchestrator.py -q` -> `73 passed`
  - `poetry run ruff check src/bitbat/model/infer.py src/bitbat_v2/config.py src/bitbat_v2/signals.py src/bitbat_v2/strategy.py tests/model/test_infer.py tests/v2/test_signal_providers.py` -> passed
