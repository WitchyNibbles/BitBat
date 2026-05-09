.PHONY: fmt lint test test-release bootstrap-monitor-model v2-api v2-api-dev autonomous-up autonomous-down

PORT ?= 8100
V2_PORT ?= 8101

fmt:
	poetry run ruff format src tests
	poetry run black src tests

lint:
	poetry run ruff check src tests
	poetry run mypy src tests

test:
	poetry run pytest

test-release:
	poetry run pytest tests/autonomous/test_phase8_d1_monitor_schema_complete.py tests/autonomous/test_agent_integration.py tests/test_cli.py tests/api/test_health.py tests/api/test_metrics.py -q -k "schema or monitor"
	poetry run pytest tests/gui/test_timeline.py tests/gui/test_phase5_timeline_complete.py tests/gui/test_phase6_timeline_ux_complete.py tests/gui/test_phase8_d2_timeline_complete.py tests/gui/test_phase9_timeline_readability_complete.py tests/gui/test_presets.py tests/gui/test_widgets.py tests/gui/test_performance_helpers.py tests/api/test_settings.py -q
	cd dashboard && npm run build

v2-api:
	poetry run uvicorn bitbat_v2.api.app:app --host 0.0.0.0 --port $(PORT)

v2-api-dev:
	poetry run uvicorn bitbat_v2.api.app:app --reload --host 0.0.0.0 --port $(PORT)

bootstrap-monitor-model:
	@test -n "$(CONFIG)" || (echo "Usage: make bootstrap-monitor-model CONFIG=path/to/config.yaml [START=YYYY-MM-DD] [SYMBOL=BTC-USD]" && exit 1)
	poetry run python scripts/bootstrap_monitor_model.py --config "$(CONFIG)" $(if $(START),--start "$(START)") $(if $(SYMBOL),--symbol "$(SYMBOL)")

autonomous-up:
	BITBAT_V2_PORT=$(V2_PORT) BITBAT_V2_AUTORUN_ENABLED=$${BITBAT_V2_AUTORUN_ENABLED:-true} BITBAT_V2_OPERATOR_TOKEN=$${BITBAT_V2_OPERATOR_TOKEN:-bitbat-local-dev-token} docker compose --profile v2 up --build

autonomous-down:
	BITBAT_V2_PORT=$(V2_PORT) docker compose --profile v2 down
