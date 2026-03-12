"""Monitor command group for the BitBat CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import click
import pandas as pd

from bitbat.autonomous.schema_compat import SchemaCompatibilityError
from bitbat.cli._helpers import (
    _config,
    _emit_monitor_startup_context,
    _predictions_path,
    _raise_monitor_model_preflight_error,
    _raise_monitor_runtime_db_error,
    _raise_monitor_schema_error,
    _resolve_setting,
)


@click.group(help="Monitoring commands.")
def monitor() -> None:
    """Monitor command namespace."""


@monitor.command("refresh")
@click.option("--freq", default=None, help="Bar frequency.")
@click.option("--horizon", default=None, help="Prediction horizon.")
@click.option(
    "--cost-bps",
    "--cost_bps",
    type=float,
    default=None,
    help="Trading costs assumed for live metrics.",
)
def monitor_refresh(
    freq: str | None,
    horizon: str | None,
    cost_bps: float | None,
) -> None:
    """Refresh monitoring metrics from prediction store."""
    from bitbat.cli._helpers import _ensure_path_exists
    from bitbat.contracts import ensure_predictions_contract

    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    cost = cost_bps if cost_bps is not None else float(_config()["cost_bps"])

    predictions_path = _predictions_path(freq_val, horizon_val)
    _ensure_path_exists(predictions_path, "Predictions parquet")
    preds = ensure_predictions_contract(pd.read_parquet(predictions_path))
    if preds.empty:
        raise click.ClickException("No live predictions to monitor.")

    preds["timestamp_utc"] = pd.to_datetime(preds["timestamp_utc"], utc=True, errors="coerce")
    preds = preds.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    realised = preds["realized_r"].astype(float)
    realized_mask = realised.notna()
    hit_rate = 0.0
    if realized_mask.any():
        hit_rate = float((realised[realized_mask] > cost / 10000).mean())
    live_metrics = {
        "count": int(len(preds)),
        "avg_predicted_return": float(preds["predicted_return"].mean()),
        "realized_count": int(realized_mask.sum()),
        "hit_rate": hit_rate,
        "updated_at": datetime.now(UTC).isoformat(),
    }

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / f"live_{freq_val}_{horizon_val}.json"
    output_path.write_text(json.dumps(live_metrics, indent=2), encoding="utf-8")
    click.echo(f"Wrote monitoring snapshot to {output_path}")


@monitor.command("run-once")
@click.option("--freq", default=None, help="Bar frequency (defaults to config).")
@click.option("--horizon", default=None, help="Prediction horizon (defaults to config).")
def monitor_run_once(freq: str | None, horizon: str | None) -> None:
    """Run one autonomous monitoring iteration."""
    from bitbat.autonomous.agent import MonitoringAgent
    from bitbat.autonomous.db import AutonomousDB, MonitorDatabaseError
    from bitbat.autonomous.models import init_database

    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    db_url = str(
        _config().get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db")
    )
    _emit_monitor_startup_context(freq_val, horizon_val)

    try:
        init_database(db_url)
        db = AutonomousDB(db_url)
        agent = MonitoringAgent(db, freq=freq_val, horizon=horizon_val)
        result = agent.run_once()
    except FileNotFoundError as exc:
        _raise_monitor_model_preflight_error(exc)
    except SchemaCompatibilityError as exc:
        _raise_monitor_schema_error(exc, db_url)
    except MonitorDatabaseError as exc:
        _raise_monitor_runtime_db_error(exc)

    click.echo("Monitoring run completed")
    click.echo(f"  Validations: {result['validations']}")
    click.echo(f"  Drift detected: {result['drift_detected']}")
    click.echo(f"  Retraining triggered: {result['retraining_triggered']}")
    prediction_payload = result.get("prediction")
    prediction_state = result.get("prediction_state")
    if prediction_state is None and isinstance(prediction_payload, dict):
        prediction_state = (
            "generated" if prediction_payload.get("status") == "generated" else "none"
        )
    prediction_state = str(prediction_state or "unknown")

    prediction_reason = result.get("prediction_reason")
    if prediction_reason is None and isinstance(prediction_payload, dict):
        prediction_reason = prediction_payload.get("reason")
    prediction_reason = str(prediction_reason or "unknown")

    realization_state = str(result.get("realization_state") or "unknown")
    pending_validations = int(result.get("pending_validations", 0))
    prediction_message = str(result.get("prediction_message") or "")
    cycle_diagnostic = str(result.get("cycle_diagnostic") or "").strip()
    if not cycle_diagnostic:
        if prediction_state == "generated":
            cycle_diagnostic = "prediction_generated"
        elif prediction_message:
            cycle_diagnostic = f"{prediction_reason}: {prediction_message}"
        else:
            cycle_diagnostic = prediction_reason

    click.echo(f"  Prediction state: {prediction_state}")
    click.echo(f"  Prediction reason: {prediction_reason}")
    click.echo(f"  Realization state: {realization_state}")
    click.echo(f"  Pending validations: {pending_validations}")
    click.echo(f"  Cycle diagnostic: {cycle_diagnostic}")

    metrics = result.get("metrics", {})
    diagnostics = metrics.get("window_diagnostics") if isinstance(metrics, dict) else None
    if isinstance(diagnostics, dict):
        click.echo(f"  Regime: {diagnostics.get('regime', 'unknown')}")
        click.echo(f"  Drift score: {float(diagnostics.get('drift_score', 0.0)):.6f}")


@monitor.command("start")
@click.option("--freq", default=None, help="Bar frequency (defaults to config).")
@click.option("--horizon", default=None, help="Prediction horizon (defaults to config).")
@click.option("--interval", type=int, default=None, help="Seconds between monitoring cycles.")
def monitor_start(freq: str | None, horizon: str | None, interval: int | None) -> None:
    """Start continuous autonomous monitoring."""
    from bitbat.autonomous.agent import MonitoringAgent
    from bitbat.autonomous.db import AutonomousDB, MonitorDatabaseError
    from bitbat.autonomous.models import init_database

    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    db_url = str(
        _config().get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db")
    )
    interval_seconds = (
        interval
        if interval is not None
        else int(_config().get("autonomous", {}).get("validation_interval", 300))
    )
    _emit_monitor_startup_context(freq_val, horizon_val)

    try:
        init_database(db_url)
        db = AutonomousDB(db_url)
        agent = MonitoringAgent(db, freq=freq_val, horizon=horizon_val)
    except FileNotFoundError as exc:
        _raise_monitor_model_preflight_error(exc)
    except SchemaCompatibilityError as exc:
        _raise_monitor_schema_error(exc, db_url)
    except MonitorDatabaseError as exc:
        _raise_monitor_runtime_db_error(exc)
    click.echo(
        "Starting monitoring loop "
        f"(freq={freq_val}, horizon={horizon_val}, interval={interval_seconds}s)"
    )
    agent.run_forever(interval_seconds=interval_seconds)


@monitor.command("status")
@click.option("--freq", default=None, help="Bar frequency (defaults to config).")
@click.option("--horizon", default=None, help="Prediction horizon (defaults to config).")
def monitor_status(freq: str | None, horizon: str | None) -> None:
    """Show latest autonomous monitoring status."""
    from bitbat.autonomous.db import AutonomousDB
    from bitbat.autonomous.models import PerformanceSnapshot, RetrainingEvent, init_database

    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    db_url = str(
        _config().get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db")
    )

    try:
        init_database(db_url)
        db = AutonomousDB(db_url)
    except SchemaCompatibilityError as exc:
        _raise_monitor_schema_error(exc, db_url)

    with db.session() as session:
        lifecycle_counts = db.get_prediction_counts(
            session=session,
            freq=freq_val,
            horizon=horizon_val,
        )
        latest_snapshot = (
            session.query(PerformanceSnapshot)
            .filter(
                PerformanceSnapshot.freq == freq_val,
                PerformanceSnapshot.horizon == horizon_val,
            )
            .order_by(PerformanceSnapshot.snapshot_time.desc())
            .first()
        )
        last_retraining = (
            session.query(RetrainingEvent).order_by(RetrainingEvent.started_at.desc()).first()
        )

    click.echo(f"Monitoring status for {freq_val}/{horizon_val}")
    click.echo(f"  Total predictions: {int(lifecycle_counts['total_predictions'])}")
    click.echo(f"  Unrealized predictions: {int(lifecycle_counts['unrealized_predictions'])}")
    click.echo(f"  Realized predictions: {int(lifecycle_counts['realized_predictions'])}")
    click.echo(f"  Pending validations: {int(lifecycle_counts['unrealized_predictions'])}")
    if latest_snapshot is None:
        click.echo("  Latest snapshot: none")
    else:
        click.echo(f"  Latest snapshot: {latest_snapshot.snapshot_time}")
        click.echo(f"  Hit rate: {latest_snapshot.hit_rate or 0.0:.2%}")
        click.echo(f"  Sharpe: {latest_snapshot.sharpe_ratio or 0.0:.3f}")
        click.echo(f"  Realized predictions: {latest_snapshot.realized_predictions}")

    if last_retraining is None:
        click.echo("  Last retraining: none")
    else:
        click.echo(
            "  Last retraining: "
            f"id={last_retraining.id}, status={last_retraining.status}, "
            f"started_at={last_retraining.started_at}"
        )


@monitor.command("snapshots")
@click.option("--freq", default=None, help="Bar frequency (defaults to config).")
@click.option("--horizon", default=None, help="Prediction horizon (defaults to config).")
@click.option("--last", "last_count", type=int, default=10, help="Number of snapshots to show.")
def monitor_snapshots(freq: str | None, horizon: str | None, last_count: int) -> None:
    """Print recent performance snapshots."""
    from bitbat.autonomous.db import AutonomousDB
    from bitbat.autonomous.models import PerformanceSnapshot, init_database

    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    db_url = str(
        _config().get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db")
    )

    try:
        init_database(db_url)
        db = AutonomousDB(db_url)
    except SchemaCompatibilityError as exc:
        _raise_monitor_schema_error(exc, db_url)

    with db.session() as session:
        snapshots = (
            session.query(PerformanceSnapshot)
            .filter(
                PerformanceSnapshot.freq == freq_val,
                PerformanceSnapshot.horizon == horizon_val,
            )
            .order_by(PerformanceSnapshot.snapshot_time.desc())
            .limit(max(last_count, 1))
            .all()
        )

    if not snapshots:
        click.echo("No snapshots found.")
        return

    click.echo(f"Recent snapshots ({len(snapshots)}):")
    for snapshot in snapshots:
        click.echo(
            f"  {snapshot.snapshot_time} | "
            f"hit_rate={(snapshot.hit_rate or 0.0):.2%} | "
            f"sharpe={(snapshot.sharpe_ratio or 0.0):.3f} | "
            f"realized={snapshot.realized_predictions}"
        )
