from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"
START_SCRIPT = ROOT / "scripts" / "start.sh"
README = ROOT / "README.md"
DASHBOARD_DOCKERFILE = ROOT / "dashboard" / "Dockerfile"
NGINX_CONFIG = ROOT / "deployment" / "nginx.conf"

pytestmark = pytest.mark.structural


def _load_compose() -> dict[str, object]:
    loaded = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    return cast(dict[str, object], loaded)


def _services(compose: dict[str, object]) -> dict[str, dict[str, Any]]:
    return cast(dict[str, dict[str, Any]], compose["services"])


def test_default_compose_starts_v2_api_without_profile_gate() -> None:
    compose = _load_compose()
    services = _services(compose)
    v2_api = services["bitbat-v2-api"]

    assert "profiles" not in v2_api
    assert "command" not in v2_api
    assert "BITBAT_PRIMARY_API=v2" in v2_api["environment"]
    assert "BITBAT_V2_OPERATOR_TOKEN=${BITBAT_V2_OPERATOR_TOKEN}" in v2_api["environment"]


def test_legacy_active_execution_services_are_profile_gated() -> None:
    compose = _load_compose()
    services = _services(compose)

    assert services["bitbat-api"]["profiles"] == ["legacy"]
    assert services["bitbat-ingest"]["profiles"] == ["legacy"]
    assert services["bitbat-monitor"]["profiles"] == ["legacy"]


def test_dashboard_depends_on_v2_api_and_receives_build_time_v2_env() -> None:
    compose = _load_compose()
    ui = _services(compose)["bitbat-ui"]

    assert "bitbat-api" not in ui["depends_on"]
    assert "bitbat-v2-api" in ui["depends_on"]
    assert ui["build"]["args"]["VITE_V2_API_URL"] == "${VITE_V2_API_URL:-http://localhost:8100}"
    assert "VITE_V2_OPERATOR_TOKEN" not in ui["build"]["args"]


def test_start_script_defaults_to_v2_api_and_legacy_services_are_opt_in() -> None:
    source = START_SCRIPT.read_text(encoding="utf-8")

    assert 'PRIMARY_API="${BITBAT_PRIMARY_API:-v2}"' in source
    assert 'LEGACY_SERVICES_ENABLED="${BITBAT_LEGACY_SERVICES_ENABLED:-false}"' in source
    assert "BITBAT_V2_OPERATOR_TOKEN must be set before starting the BitBat v2 API." in source
    assert "bitbat_v2.api.app:app" in source
    assert "bitbat.api.app:app" in source


def test_dashboard_build_accepts_v2_env_args() -> None:
    source = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG VITE_V2_API_URL=http://localhost:8100" in source
    assert "ENV VITE_V2_API_URL=${VITE_V2_API_URL}" in source
    assert "VITE_V2_OPERATOR_TOKEN" not in source


def test_nginx_proxy_profile_targets_v2_api_surface() -> None:
    source = NGINX_CONFIG.read_text(encoding="utf-8")

    assert "server bitbat-v2-api:8100;" in source
    assert "location /v1/" in source
    assert "server bitbat-api:8000;" not in source


def test_readme_calls_out_v2_as_primary_operator_path() -> None:
    source = README.read_text(encoding="utf-8")

    assert "Primary v2 API" in source
    assert "bitbat_v2` is the primary paper-trading operator path" in source
    assert "legacy `bitbat` API remains available during migration as diagnostic support" in source
    assert "docker compose --profile legacy up --build" in source
