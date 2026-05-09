# Review Gate

## Task ID

`2026-05-09-model-behavior-refactor-execution`

## Reviewer role

`security_reviewer`

## Actor

`019e0be9-c94d-7ae0-933e-171e4825795f`

## Actor role

`security_reviewer`

## Provenance status

`runtime_verified`

## Review state

`passed`

## Severity

`low`

## Specialist execution evidence

- Security review covered:
  - [runtime.py](/home/eimi/projects/ai-btc-predictor/src/bitbat_v2/runtime.py:1)
  - [signals.py](/home/eimi/projects/ai-btc-predictor/src/bitbat_v2/signals.py:1)
  - [storage.py](/home/eimi/projects/ai-btc-predictor/src/bitbat_v2/storage.py:1)
  - [schemas.py](/home/eimi/projects/ai-btc-predictor/src/bitbat_v2/api/schemas.py:1)
- The slice preserved paper-only execution and kept order placement inside runtime, not model code.

## Quality gate evidence

- `poetry run pytest tests/v2 -q`
- `make lint`

## Findings

- No blocking security findings were reported.

## Residual risk

- `legacy_ml` remains an env-selectable trusted signal source and therefore continues to widen the
  model trust boundary.
- Future work on model promotion or microstructure ingestion will need fresh security review.

## Verification evidence

- Security handoff reported no blocking findings.
- Final branch-state evidence:
- Runtime proof: `poetry run pytest tests/v2 -q` -> `55 passed`
- Runtime proof: `make lint` -> passed

## Waiver authority

`none`

## Waiver reason

None.

## Decision

`approved`

## Source handoff

Manager summary of security reviewer output plus final branch-state evidence. The slice does not
introduce a new control surface and preserves the existing paper-only boundary.
Runtime proof: final branch state passed the full v2 suite and lint/type-check after the additive
schema migration and API-surface widening.

## 2026-05-09 Addendum

- The mode-specific refactor does not add live-order connectivity, secrets, or a new execution
  surface; it only changes training-selection and local inference selection.
- Residual security risk remains the same trusted-model boundary: artifact metadata is now read at
  runtime, so future work should keep metadata local and immutable once promotion rules exist.
- Addendum verification evidence:
  - `poetry run pytest tests/autonomous/test_predictor_operability.py tests/autonomous/test_agent_integration.py tests/model/test_persist.py tests/model/test_train.py tests/model/test_infer.py tests/autonomous/test_orchestrator.py tests/api/test_predictions.py tests/test_config_loader.py tests/gui/test_performance_helpers.py tests/gui/test_timeline.py -q` -> `118 passed`
  - `poetry run ruff check src/bitbat/model/mode_profiles.py src/bitbat/model/persist.py src/bitbat/model/train.py src/bitbat/model/infer.py src/bitbat/autonomous/orchestrator.py src/bitbat/autonomous/predictor.py tests/model/test_persist.py tests/model/test_train.py tests/model/test_infer.py tests/autonomous/test_orchestrator.py` -> passed

## 2026-05-09 Selection Addendum

- Security surface remains unchanged: the new candidate-selection path is local evaluation only.
- Runtime now trusts active model metadata for family resolution, so metadata integrity remains part
  of the trusted local artifact boundary.
- Addendum verification evidence:
  - `poetry run pytest tests/model/test_mode_selection.py tests/model/test_persist.py tests/model/test_train.py tests/model/test_infer.py tests/autonomous/test_orchestrator.py tests/autonomous/test_predictor_operability.py tests/autonomous/test_agent_integration.py tests/api/test_predictions.py tests/test_config_loader.py tests/gui/test_performance_helpers.py tests/gui/test_timeline.py -q` -> `122 passed`
  - `poetry run ruff check src/bitbat/model/mode_profiles.py src/bitbat/model/mode_selection.py src/bitbat/model/persist.py src/bitbat/model/train.py src/bitbat/model/infer.py src/bitbat/autonomous/orchestrator.py src/bitbat/autonomous/predictor.py tests/model/test_mode_selection.py tests/model/test_persist.py tests/model/test_train.py tests/model/test_infer.py tests/autonomous/test_orchestrator.py tests/autonomous/test_predictor_operability.py` -> passed

## 2026-05-09 Runtime Policy Addendum

- This slice keeps execution scope unchanged and paper-only. The new runtime policy reads trusted
  local artifact metadata and uses it to hold weak signals instead of widening execution behavior.
- Security-sensitive compatibility fixes were additive only:
  - explicit signal-source validation
  - legacy strategy alias mapping
  - local winner-artifact resolution between `xgb` and `random_forest`
  - no new network, auth, or order-placement surface
- Addendum verification evidence:
  - `poetry run pytest tests/v2 tests/model/test_infer.py tests/autonomous/test_predictor_operability.py tests/autonomous/test_orchestrator.py -q` -> `73 passed`
  - `poetry run ruff check src/bitbat/model/infer.py src/bitbat_v2/config.py src/bitbat_v2/signals.py src/bitbat_v2/strategy.py tests/model/test_infer.py tests/v2/test_signal_providers.py` -> passed
