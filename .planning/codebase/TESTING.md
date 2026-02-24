# Testing

**Analysis Date:** 2026-02-24

## Framework and Configuration

- Test runner: Pytest.
- Config in `pyproject.toml`:
  - `testpaths = ["tests"]`
  - `addopts = "-ra"`
  - `pythonpath = ["src"]`
  - slow marker defined: `slow: marks tests as slow and may require network access`.

## Test Suite Layout

- Tests are organized by subsystem and largely mirror runtime modules:
  - `tests/api/`
  - `tests/autonomous/`
  - `tests/model/`
  - `tests/ingest/`
  - `tests/features/`
  - `tests/dataset/`
  - `tests/backtest/`
  - `tests/analytics/`
  - `tests/gui/`
  - `tests/io/`
  - plus contract/timealign/config/CLI checks.

## Common Test Patterns

- Heavy use of `tmp_path` for isolated local artifacts.
- `monkeypatch` used to:
  - isolate cwd-dependent behavior,
  - replace network/system calls,
  - patch config loader behavior.
- API tests use FastAPI `TestClient` and temporary DB/model fixtures.
- Model tests generate temporary XGBoost artifacts for inference/persistence checks.

## Subsystem Coverage Highlights

### Ingestion and Contracts

- Price/news/macro/on-chain ingestion modules tested for shape, dedupe, and contract adherence.
- Contract validation behavior covered in `tests/contracts/test_contracts.py`.

### Dataset and Leakage Controls

- Dataset assembly and split logic tested in `tests/dataset/` and `tests/timealign/`.
- Focus includes forward-return handling and purging/bucket calendar behavior.

### Model Lifecycle

- Training/inference/persistence/evaluation covered in `tests/model/`.
- Includes walk-forward, ensemble, and optimization checks.

### Autonomous Monitoring

- DB models/repository, drift logic, retrainer behavior, ingestion services, and agent integration tested in `tests/autonomous/`.
- Session/phase completion tests exist for end-to-end milestones.

### API and UI

- API route behavior tested in `tests/api/`.
- Streamlit support utilities tested in `tests/gui/`.

## Execution Commands

- Full suite: `make test` or `poetry run pytest`.
- Lint + typing: `make lint`.
- Format: `make fmt`.

## Testing Risks and Gaps

- No shared `conftest.py`; fixtures are distributed, which can increase duplication.
- Some operational flows rely on broad exception fallbacks, so tests may assert degraded behavior rather than strict failure semantics.
- External API integrations are mostly mocked; real integration confidence depends on runtime smoke checks.

