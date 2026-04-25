#!/usr/bin/env python
"""Compare BitBat v2 baseline and improved paper-only strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bitbat_v2.config import BitBatV2Config
from bitbat_v2.evaluation import compare_strategies, load_candles_from_parquet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/raw/prices/btcusd_yf_5m.parquet",
        help="Path to repo-local OHLCV parquet data.",
    )
    args = parser.parse_args()

    config = BitBatV2Config()
    candles = load_candles_from_parquet(Path(args.input), config)
    comparison = compare_strategies(candles, config)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
