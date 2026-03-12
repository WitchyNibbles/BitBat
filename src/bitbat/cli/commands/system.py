"""System lifecycle CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from bitbat.cli._helpers import _config
from bitbat.config.loader import resolve_models_dir


@click.group(help="System lifecycle commands.")
def system() -> None:
    """System command namespace."""


@system.command("reset")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
def system_reset(yes: bool) -> None:
    """Delete data/, models/, and autonomous.db for a clean-slate restart.

    After reset, run: bitbat ingest prices-once, features build, model train.
    """
    import shutil

    if not yes:
        click.confirm(
            "This will delete all data, models, and the monitor database. Continue?",
            abort=True,
        )

    cfg = _config()
    data_dir = Path(str(cfg.get("data_dir", "data"))).expanduser()
    models_dir = resolve_models_dir()
    db_url = str(cfg.get("autonomous", {}).get("database_url", "sqlite:///data/autonomous.db"))
    db_path = Path(db_url.replace("sqlite:///", ""))

    deleted: list[str] = []
    for target in [data_dir, models_dir]:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            deleted.append(str(target))

    # Delete autonomous.db explicitly only if it's outside data_dir
    # (data_dir rmtree already covers the typical case of data/autonomous.db)
    if db_path.exists() and not db_path.is_relative_to(data_dir):
        db_path.unlink(missing_ok=True)
        deleted.append(str(db_path))

    if deleted:
        click.echo(f"Reset complete. Deleted: {', '.join(deleted)}")
    else:
        click.echo("Nothing to delete — already clean.")
