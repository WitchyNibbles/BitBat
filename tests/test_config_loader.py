from pathlib import Path

import pytest

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
