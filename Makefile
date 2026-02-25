.PHONY: fmt lint test test-release streamlit

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
	poetry run pytest tests/gui/test_timeline.py tests/gui/test_complete_gui.py tests/gui/test_phase5_timeline_complete.py tests/gui/test_phase6_timeline_ux_complete.py tests/gui/test_phase8_d2_timeline_complete.py tests/gui/test_phase10_supported_surface_complete.py tests/gui/test_phase11_runtime_stability_complete.py tests/gui/test_phase12_simplified_ui_regression_complete.py tests/gui/test_phase12_supported_views_smoke.py -q
	poetry run pytest tests/gui/test_streamlit_width_compat.py tests/gui/test_phase7_streamlit_compat_complete.py tests/gui/test_phase8_release_verification_complete.py -q

streamlit:
	poetry run streamlit run streamlit/app.py
