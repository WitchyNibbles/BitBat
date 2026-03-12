"""BitBat command line interface."""

from __future__ import annotations

from pathlib import Path

import click

from bitbat import __version__
from bitbat.cli.commands import backtest as _backtest_mod
from bitbat.cli.commands import batch as _batch_mod
from bitbat.cli.commands import features as _features_mod
from bitbat.cli.commands import ingest as _ingest_mod
from bitbat.cli.commands import model as _model_mod
from bitbat.cli.commands import monitor as _monitor_mod
from bitbat.cli.commands import news as _news_mod
from bitbat.cli.commands import prices as _prices_mod
from bitbat.cli.commands import system as _system_mod
from bitbat.cli.commands import validate as _validate_mod

# Re-exports for test compatibility (symbols defined in sub-modules)
from bitbat.cli.commands.backtest import (  # noqa: F401
    run_strategy,
    summarize_backtest,
)
from bitbat.cli.commands.batch import (  # noqa: F401
    aggregate_sentiment,
    generate_price_features,
    load_model,
    predict_bar,
)
from bitbat.cli.commands.model import (  # noqa: F401
    HyperparamOptimizer,
    build_xy,
    compute_multiple_testing_safeguards,
    fit_random_forest,
    fit_xgb,
    regression_metrics,
    save_baseline_artifact,
    walk_forward,
    xgb,
)
from bitbat.config.loader import set_runtime_config


@click.group(name="bitbat", invoke_without_command=True)
@click.option(
    "--config",
    type=click.Path(exists=False, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Path to configuration file (overrides BITBAT_CONFIG).",
)
@click.option("--version", is_flag=True, help="Show version and exit.")
@click.pass_context
def _cli(ctx: click.Context, config: Path | None, version: bool) -> None:
    set_runtime_config(config)
    if version:
        click.echo(__version__)
        ctx.exit()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


_cli.add_command(_prices_mod.prices)
_cli.add_command(_news_mod.news)
_cli.add_command(_features_mod.features)
_cli.add_command(_backtest_mod.backtest)
_cli.add_command(_batch_mod.batch)
_cli.add_command(_validate_mod.validate)
_cli.add_command(_ingest_mod.ingest)
_cli.add_command(_system_mod.system)
_cli.add_command(_model_mod.model)
_cli.add_command(_monitor_mod.monitor)


def main() -> None:
    """Entry point used by tests and scripts."""
    _cli.main(standalone_mode=False)


if __name__ == "__main__":
    _cli()
