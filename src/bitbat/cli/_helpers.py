"""Shared private helper functions for the BitBat CLI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import click
import numpy as np
import pandas as pd
import xgboost as xgb

from bitbat.autonomous.schema_compat import SchemaCompatibilityError, format_missing_columns
from bitbat.config.loader import (
    get_runtime_config,
    get_runtime_config_path,
    get_runtime_config_source,
)
from bitbat.contracts import ensure_feature_contract
from bitbat.model.persist import default_model_artifact_path
from bitbat.timealign.calendar import ensure_utc

if TYPE_CHECKING:
    from bitbat.autonomous.db import MonitorDatabaseError


def _config() -> dict[str, Any]:
    return get_runtime_config()


def _sentiment_enabled() -> bool:
    return bool(_config().get("enable_sentiment", True))


def _resolve_news_source(source: str | None = None) -> str:
    configured = (
        source if source not in (None, "") else _config().get("news_source", "cryptocompare")
    )
    resolved = str(configured).strip().lower()
    if resolved not in {"gdelt", "cryptocompare"}:
        raise click.ClickException(
            f"Unsupported news_source '{resolved}'. Expected one of: gdelt, cryptocompare."
        )
    return resolved


def _news_backend(source: str) -> Any:
    if source == "gdelt":
        from bitbat.ingest import news_gdelt as backend
    else:
        from bitbat.ingest import news_cryptocompare as backend
    return backend


def _data_path(*parts: str | Path) -> Path:
    base = Path(_config()["data_dir"]).expanduser()
    return base.joinpath(*parts)


def _resolve_setting(value: Any | None, key: str) -> str:
    result = value if value not in (None, "") else _config().get(key)
    if result is None:
        raise KeyError(f"Configuration missing required setting '{key}'")
    return str(result)


def _parse_datetime(raw: str, label: str) -> datetime:
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise click.BadParameter(f"Invalid {label} datetime: {raw}") from exc


def _ensure_path_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise click.ClickException(f"{description} not found: {path}")


def _feature_dataset_path(freq: str, horizon: str) -> Path:
    return _data_path("features", f"{freq}_{horizon}", "dataset.parquet")


def _load_feature_dataset(
    freq: str,
    horizon: str,
    *,
    require_label: bool,
    require_forward_return: bool | None = None,
) -> pd.DataFrame:
    dataset_path = _feature_dataset_path(freq, horizon)
    _ensure_path_exists(dataset_path, "Feature dataset")
    dataset = pd.read_parquet(dataset_path)
    _require_fwd = require_forward_return if require_forward_return is not None else require_label
    dataset = ensure_feature_contract(
        dataset,
        require_label=require_label,
        require_forward_return=_require_fwd,
        require_features_full=_sentiment_enabled(),
    )
    return dataset.sort_values("timestamp_utc").set_index("timestamp_utc")


def _load_prices_indexed(freq: str) -> pd.DataFrame:
    from bitbat.io.prices import load_prices_for_cli

    data_dir = Path(_config()["data_dir"]).expanduser()
    try:
        return load_prices_for_cli(freq, data_dir=data_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


def _load_news() -> pd.DataFrame:
    source = _resolve_news_source()
    backend = _news_backend(source)
    news_root = _data_path("raw", "news", f"{source}_1h")
    news_path = backend._target_path(news_root)
    _ensure_path_exists(news_path, "News parquet")
    return ensure_utc(pd.read_parquet(news_path), "published_utc").sort_values("published_utc")


def _predictions_path(freq: str, horizon: str) -> Path:
    return _data_path("predictions", f"{freq}_{horizon}.parquet")


def _model_path(freq: str, horizon: str) -> Path:
    return default_model_artifact_path(freq, horizon, family="xgb")


def _resolve_model_families(selection: str | None) -> list[Literal["xgb", "random_forest"]]:
    configured = _config().get("model", {})
    default_family = str(configured.get("baseline_family", "xgb")).strip().lower()
    requested = str(selection or default_family).strip().lower()

    if requested == "both":
        return ["xgb", "random_forest"]
    if requested not in {"xgb", "random_forest"}:
        raise click.ClickException(
            f"Unsupported model family '{requested}'. Expected xgb, random_forest, or both."
        )
    return [requested]


def _predict_baseline(
    family: str,
    model: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    if family == "xgb":
        dtest = xgb.DMatrix(features, feature_names=list(features.columns))
        return np.asarray(model.predict(dtest), dtype="float64")
    return np.asarray(model.predict(features.astype(float)), dtype="float64")


def _raise_monitor_schema_error(exc: SchemaCompatibilityError, db_url: str) -> NoReturn:
    missing = format_missing_columns(exc.report) or "unknown"
    raise click.ClickException(
        "\n".join([
            "Autonomous DB schema is incompatible for monitor commands.",
            f"Missing columns: {missing}",
            (
                "Run: poetry run python scripts/init_autonomous_db.py "
                f'--database-url "{db_url}" --audit'
            ),
            (
                "Then: poetry run python scripts/init_autonomous_db.py "
                f'--database-url "{db_url}" --upgrade'
            ),
        ])
    ) from exc


def _raise_monitor_runtime_db_error(exc: MonitorDatabaseError) -> NoReturn:
    raise click.ClickException(
        "\n".join([
            "Autonomous monitor runtime database failure.",
            f"Step: {exc.step}",
            f"Error class: {exc.error_class}",
            f"Detail: {exc.detail}",
            f"Remediation: {exc.remediation}",
        ])
    ) from exc


def _monitor_config_source_label(source: str) -> str:
    labels = {
        "explicit": "--config",
        "env": "BITBAT_CONFIG",
        "default": "default-config",
    }
    return labels.get(source, source)


def _emit_monitor_startup_context(freq: str, horizon: str) -> None:
    source = get_runtime_config_source()
    config_path = get_runtime_config_path()
    click.echo(
        "Monitor startup config: "
        f"source={_monitor_config_source_label(source)}, path={config_path}"
    )
    click.echo(f"Resolved runtime pair: freq={freq}, horizon={horizon}")


def _raise_monitor_model_preflight_error(exc: FileNotFoundError) -> NoReturn:
    raise click.ClickException(
        "\n".join([
            "Autonomous monitor startup blocked: model artifact missing.",
            f"Detail: {exc}",
            "Remediation:",
            "  1. Use --config or BITBAT_CONFIG to select the intended freq/horizon pair.",
            "  2. Train/copy the expected model artifact: models/<freq>_<horizon>/xgb.json.",
        ])
    ) from exc
