from __future__ import annotations

from pathlib import Path

import pytest
from scripts import ensure_autonomous_stack_ready as ensure_mod

pytestmark = pytest.mark.integration


def test_candidate_artifacts_reflect_mode_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ensure_mod, "resolve_models_dir", lambda config: tmp_path / "models")

    artifacts = ensure_mod._candidate_artifacts("conservative", config={})

    assert artifacts == [
        tmp_path / "models" / "1h_24h" / "random_forest.pkl",
        tmp_path / "models" / "1h_24h" / "xgb.json",
    ]


def test_main_skips_training_when_artifact_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "models" / "1h_24h" / "random_forest.pkl"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(ensure_mod, "reset_runtime_config", lambda: None)
    monkeypatch.setattr(
        ensure_mod, "set_runtime_config", lambda path=None: {"preset": "conservative"}
    )
    monkeypatch.setattr(ensure_mod, "get_runtime_config", lambda: {"preset": "conservative"})
    monkeypatch.setattr(ensure_mod, "resolve_models_dir", lambda config: tmp_path / "models")

    called: list[str] = []

    def _fail_train(*, preset_name: str) -> dict[str, object]:
        called.append(preset_name)
        return {"status": "success"}

    monkeypatch.setattr(ensure_mod, "one_click_train", _fail_train)

    exit_code = ensure_mod.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "existing artifact found" in captured.out
    assert called == []


def test_main_trains_when_artifact_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "models" / "5m_30m" / "xgb.json"

    monkeypatch.setattr(ensure_mod, "reset_runtime_config", lambda: None)
    monkeypatch.setattr(
        ensure_mod,
        "set_runtime_config",
        lambda path=None: {"preset": "scalper"},
    )
    monkeypatch.setattr(ensure_mod, "get_runtime_config", lambda: {"preset": "scalper"})
    monkeypatch.setattr(ensure_mod, "resolve_models_dir", lambda config: tmp_path / "models")

    def _train(*, preset_name: str) -> dict[str, object]:
        assert preset_name == "scalper"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        return {"status": "success", "model_version": "model-v1"}

    monkeypatch.setattr(ensure_mod, "one_click_train", _train)

    exit_code = ensure_mod.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "bootstrap complete" in captured.out.lower()


def test_main_fails_when_training_reports_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(ensure_mod, "reset_runtime_config", lambda: None)
    monkeypatch.setattr(
        ensure_mod,
        "set_runtime_config",
        lambda path=None: {"preset": "balanced"},
    )
    monkeypatch.setattr(ensure_mod, "get_runtime_config", lambda: {"preset": "balanced"})
    monkeypatch.setattr(ensure_mod, "resolve_models_dir", lambda config: tmp_path / "models")
    monkeypatch.setattr(
        ensure_mod,
        "one_click_train",
        lambda *, preset_name: {"status": "failed", "error": "boom", "preset": preset_name},
    )

    exit_code = ensure_mod.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "bootstrap failed" in captured.err.lower()
