#!/usr/bin/env python
"""Run the BitBat autonomous monitoring agent continuously."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

# Allow script execution without package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bitbat.autonomous.agent import MonitoringAgent
from bitbat.autonomous.db import AutonomousDB
from bitbat.autonomous.models import init_database
from bitbat.config.loader import set_runtime_config


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
    if db_url.startswith("sqlite:///"):
        db_path = Path(db_url.replace("sqlite:///", "", 1))
        return db_path.parent / "monitoring_agent_heartbeat.json"
    return Path("data") / "monitoring_agent_heartbeat.json"


def _write_heartbeat(
    path: Path,
    *,
    status: str,
    freq: str,
    horizon: str,
    interval: int,
    db_url: str,
    error: str | None = None,
) -> None:
    payload = {
        "status": status,
        "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        "freq": freq,
        "horizon": horizon,
        "interval_seconds": int(interval),
        "database_url": db_url,
    }
    if error is not None:
        payload["error"] = error

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True))
    except Exception:
        logging.getLogger(__name__).debug("Failed to write heartbeat file", exc_info=True)


def main() -> int:
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
    )

    while not shutdown["requested"]:
        try:
            agent.run_once()
            _write_heartbeat(
                heartbeat,
                status="ok",
                freq=freq,
                horizon=horizon,
                interval=interval,
                db_url=db_url,
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
