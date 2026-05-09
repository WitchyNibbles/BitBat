from pathlib import Path

import pytest

import bitbat.config.loader as loader
from bitbat.config.loader import load_config

pytest.importorskip("yaml", reason="PyYAML required for config loader tests.")


pytestmark = pytest.mark.behavioral


def test_load_default_config() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "src" / "bitbat" / "config" / "default.yaml"

    config = load_config(config_path)

    assert config["data_dir"] == "data"
    assert config["freq"] == "5m"
    assert config["horizon"] == "30m"
    assert config["autonomous"]["enabled"] is True


def test_load_config_merges_repo_user_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_path = tmp_path / "src" / "bitbat" / "config" / "default.yaml"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text(
        "\n".join([
            "freq: 1h",
            "horizon: 4h",
            "tau: 0.01",
            "enter_threshold: 0.6",
            "autonomous:",
            "  enabled: true",
            "  accuracy_guardrail:",
            "    enabled: true",
            "    window_days: 30",
        ]),
        encoding="utf-8",
    )

    user_path = tmp_path / "config" / "user_config.yaml"
    user_path.parent.mkdir(parents=True, exist_ok=True)
    user_path.write_text(
        "\n".join([
            "freq: 5m",
            "tau: 0.003",
            "autonomous:",
            "  accuracy_guardrail:",
            "    window_days: 7",
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(loader, "DEFAULT_CONFIG_PATH", default_path)
    monkeypatch.setattr(loader, "_default_user_config_path", lambda: user_path)
    loader.reset_runtime_config()

    config = load_config()

    assert config["freq"] == "5m"
    assert config["horizon"] == "4h"
    assert config["tau"] == pytest.approx(0.003)
    assert config["enter_threshold"] == pytest.approx(0.6)
    assert config["autonomous"]["enabled"] is True
    assert config["autonomous"]["accuracy_guardrail"]["enabled"] is True
    assert config["autonomous"]["accuracy_guardrail"]["window_days"] == 7


def test_explicit_partial_config_inherits_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_path = tmp_path / "src" / "bitbat" / "config" / "default.yaml"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text(
        "\n".join([
            "freq: 1h",
            "horizon: 4h",
            "tau: 0.01",
            "autonomous:",
            "  enabled: true",
            "  drift_detection:",
            "    window_days: 30",
        ]),
        encoding="utf-8",
    )

    override_path = tmp_path / "custom.yaml"
    override_path.write_text(
        "\n".join([
            "freq: 15m",
            "tau: 0.007",
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(loader, "DEFAULT_CONFIG_PATH", default_path)
    loader.reset_runtime_config()

    config = load_config(override_path)

    assert config["freq"] == "15m"
    assert config["horizon"] == "4h"
    assert config["tau"] == pytest.approx(0.007)
    assert config["autonomous"]["enabled"] is True
    assert config["autonomous"]["drift_detection"]["window_days"] == 30
