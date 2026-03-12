"""Prediction validation CLI commands."""

from __future__ import annotations

import click

from bitbat.cli._helpers import _config, _resolve_setting


@click.group(help="Prediction validation commands.")
def validate() -> None:
    """Validation command namespace."""


@validate.command("run")
@click.option("--freq", default=None, help="Bar frequency (defaults to config).")
@click.option("--horizon", default=None, help="Prediction horizon (defaults to config).")
def validate_run(freq: str | None, horizon: str | None) -> None:
    """Validate pending predictions against realized outcomes."""
    from bitbat.autonomous.db import AutonomousDB
    from bitbat.autonomous.models import init_database
    from bitbat.autonomous.validator import PredictionValidator

    freq_val = _resolve_setting(freq, "freq")
    horizon_val = _resolve_setting(horizon, "horizon")
    db_url = str(
        _config().get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db")
    )

    click.echo(f"Starting validation: freq={freq_val}, horizon={horizon_val}")

    init_database(db_url)
    db = AutonomousDB(db_url)
    validator = PredictionValidator(db=db, freq=freq_val, horizon=horizon_val)
    results = validator.validate_all()

    click.echo("")
    click.echo("Validation complete")
    click.echo(f"  Validated: {results['validated_count']} predictions")
    click.echo(f"  Correct: {results['correct_count']}")
    click.echo(f"  Hit rate: {results['hit_rate']:.2%}")

    errors = list(results.get("errors", []))
    if errors:
        click.echo("")
        click.echo(f"Errors ({len(errors)}):")
        for error in errors[:5]:
            click.echo(f"  {error}")
        if len(errors) > 5:
            click.echo(f"  ... and {len(errors) - 5} more")
