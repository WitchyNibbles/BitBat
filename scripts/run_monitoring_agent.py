#!/usr/bin/env python
"""Run the BitBat autonomous monitoring agent continuously."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from types import FrameType

# Allow script execution without package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bitbat.autonomous.agent import MonitoringAgent
from bitbat.autonomous.db import AutonomousDB, MonitorDatabaseError
from bitbat.autonomous.heartbeat import heartbeat_path_for_db_url, write_monitor_heartbeat
from bitbat.autonomous.models import init_database
from bitbat.config.loader import (
    get_runtime_config_path,
    get_runtime_config_source,
    set_runtime_config,
)


def _configure_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "monitoring_agent.log"),
            logging.StreamHandler(),
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autonomous monitoring agent.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (defaults to runtime default).",
    )
    return parser.parse_args()


def _heartbeat_path(db_url: str) -> Path:
    return heartbeat_path_for_db_url(db_url)


def _write_heartbeat(
    path: Path,
    *,
    status: str,
    freq: str,
    horizon: str,
    interval: int,
    db_url: str,
    config_source: str,
    config_path: str,
    error: str | None = None,
    cycle_prediction_state: str | None = None,
    cycle_prediction_reason: str | None = None,
    cycle_realization_state: str | None = None,
    cycle_diagnostic: str | None = None,
    cycle_ingestion_state: str | None = None,
    cycle_ingestion_failures: list[dict[str, object]] | None = None,
) -> None:
    write_monitor_heartbeat(
        path,
        status=status,
        freq=freq,
        horizon=horizon,
        interval=interval,
        db_url=db_url,
        config_source=config_source,
        config_path=config_path,
        error=error,
        cycle_prediction_state=cycle_prediction_state,
        cycle_prediction_reason=cycle_prediction_reason,
        cycle_realization_state=cycle_realization_state,
        cycle_diagnostic=cycle_diagnostic,
        cycle_ingestion_state=cycle_ingestion_state,
        cycle_ingestion_failures=cycle_ingestion_failures,
    )


def main() -> int:  # noqa: C901
    args = _parse_args()
    _configure_logging()
    logger = logging.getLogger(__name__)

    shutdown = {"requested": False}

    def _signal_handler(sig: int, frame: FrameType | None) -> None:
        del frame
        shutdown["requested"] = True
        logger.info("Received signal %s, shutting down monitoring agent.", sig)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    config = set_runtime_config(args.config)
    config_source = get_runtime_config_source()
    config_path = str(get_runtime_config_path())
    autonomous = config.get("autonomous", {})
    db_url = str(autonomous.get("database_url", "sqlite:///data/autonomous.db"))
    freq = str(config.get("freq", "1h"))
    horizon = str(config.get("horizon", "4h"))
    interval = int(autonomous.get("validation_interval", 3600))
    heartbeat = _heartbeat_path(db_url)

    init_database(db_url)
    db = AutonomousDB(db_url)
    agent = MonitoringAgent(db, freq=freq, horizon=horizon)

    logger.info(
        "Monitoring agent starting with freq=%s horizon=%s interval=%s db=%s",
        freq,
        horizon,
        interval,
        db_url,
    )
    _write_heartbeat(
        heartbeat,
        status="starting",
        freq=freq,
        horizon=horizon,
        interval=interval,
        db_url=db_url,
        config_source=config_source,
        config_path=config_path,
    )

    while not shutdown["requested"]:
        try:
            result = agent.run_once()
            prediction_state = None
            prediction_reason = None
            realization_state = None
            cycle_diagnostic = None
            ingestion_state = None
            ingestion_failures = None
            if isinstance(result, dict):
                if result.get("prediction_state") is not None:
                    prediction_state = str(result.get("prediction_state"))
                if result.get("prediction_reason") is not None:
                    prediction_reason = str(result.get("prediction_reason"))
                if result.get("realization_state") is not None:
                    realization_state = str(result.get("realization_state"))
                if result.get("cycle_diagnostic") is not None:
                    cycle_diagnostic = str(result.get("cycle_diagnostic"))
                if result.get("ingestion_state") is not None:
                    ingestion_state = str(result.get("ingestion_state"))
                if isinstance(result.get("ingestion_failures"), list):
                    ingestion_failures = result.get("ingestion_failures")
            _write_heartbeat(
                heartbeat,
                status="ok",
                freq=freq,
                horizon=horizon,
                interval=interval,
                db_url=db_url,
                config_source=config_source,
                config_path=config_path,
                cycle_prediction_state=prediction_state,
                cycle_prediction_reason=prediction_reason,
                cycle_realization_state=realization_state,
                cycle_diagnostic=cycle_diagnostic,
                cycle_ingestion_state=ingestion_state,
                cycle_ingestion_failures=ingestion_failures,
            )
        except MonitorDatabaseError as exc:
            logger.error(
                "Monitoring cycle DB failure step=%s detail=%s remediation=%s",
                exc.step,
                exc.detail,
                exc.remediation,
            )
            _write_heartbeat(
                heartbeat,
                status="error",
                freq=freq,
                horizon=horizon,
                interval=interval,
                db_url=db_url,
                config_source=config_source,
                config_path=config_path,
                error=f"{exc.step}: {exc.detail}. {exc.remediation}",
            )
        except Exception:  # pragma: no cover - defensive top-level loop
            logger.exception("Monitoring cycle failed.")
            _write_heartbeat(
                heartbeat,
                status="error",
                freq=freq,
                horizon=horizon,
                interval=interval,
                db_url=db_url,
                config_source=config_source,
                config_path=config_path,
                error="Monitoring cycle failed",
            )
        if not shutdown["requested"]:
            time.sleep(max(interval, 1))

    logger.info("Monitoring agent stopped.")
    _write_heartbeat(
        heartbeat,
        status="stopped",
        freq=freq,
        horizon=horizon,
        interval=interval,
        db_url=db_url,
        config_source=config_source,
        config_path=config_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
