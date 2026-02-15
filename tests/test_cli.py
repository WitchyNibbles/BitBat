from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import pytest

from bitbat.cli import main
from bitbat.io.fs import read_parquet, write_parquet


def _write_test_config(
    path: Path,
    *,
    enable_sentiment: bool,
    news_source: str = "gdelt",
    database_url: str | None = None,
) -> Path:
    lines = [
        'data_dir: "data"',
        'freq: "1h"',
        'horizon: "4h"',
        "enter_threshold: 0.6",
        "allow_short: false",
        "cost_bps: 4",
        "tau: 0.0015",
        'price_source: "yfinance"',
        f'news_source: "{news_source}"',
        f"enable_sentiment: {'true' if enable_sentiment else 'false'}",
        "news_throttle_seconds: 10.0",
        "news_retry_limit: 30",
        "seed: 42",
    ]
    if database_url is not None:
        lines.extend([
            "autonomous:",
            f'  database_url: "{database_url}"',
            "  validation_interval: 3600",
            "  retraining:",
            "    cooldown_hours: 24",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def cli_args(tmp_path: Path) -> tuple[list[str], Path]:
    output_root = tmp_path / "prices"
    argv = [
        "bitbat",
        "prices",
        "pull",
        "--symbol",
        "BTC-USD",
        "--interval",
        "1h",
        "--start",
        "2017-01-01",
        "--output",
        str(output_root),
    ]
    return argv, output_root


def test_cli_prices_pull_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    cli_args: tuple[list[str], Path],
) -> None:
    argv, output_root = cli_args

    sample_frame = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(
            ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            utc=False,
        ),
        "open": [100.0, 110.0],
        "high": [105.0, 115.0],
        "low": [95.0, 108.0],
        "close": [104.0, 112.0],
        "volume": [1_000.0, 2_000.0],
        "source": ["yfinance", "yfinance"],
    })

    calls: list[tuple[str, str, datetime, str | None]] = []

    def fake_fetch_yf(
        symbol: str,
        interval: str,
        start: datetime,
        *,
        output_root: str | Path | None = None,
    ) -> pd.DataFrame:
        from bitbat.ingest import prices as prices_module

        calls.append((
            symbol,
            interval,
            start,
            output_root if output_root is None else str(output_root),
        ))

        target = prices_module._target_path(symbol, interval, output_root)
        if target.exists():
            shutil.rmtree(target)

        target.parent.mkdir(parents=True, exist_ok=True)

        partitioned = sample_frame.copy()
        partitioned["year"] = partitioned["timestamp_utc"].dt.year
        write_parquet(partitioned, target, partition_cols=["year"])

        return sample_frame.copy()

    monkeypatch.setattr("bitbat.ingest.prices.fetch_yf", fake_fetch_yf)

    def dataset(path: Path) -> pd.DataFrame:
        frame = read_parquet(path)
        if "year" in frame.columns:
            frame = frame.drop(columns=["year"])
        return frame.sort_values("timestamp_utc").reset_index(drop=True)

    monkeypatch.setattr(sys, "argv", argv)
    main()

    target = output_root / "btcusd_yf_1h.parquet"
    assert target.exists()
    first = dataset(target)
    pd_testing.assert_frame_equal(first, sample_frame)

    monkeypatch.setattr(sys, "argv", argv)
    main()
    second = dataset(target)
    pd_testing.assert_frame_equal(second, sample_frame)

    # Ensure idempotency.
    assert len(second) == len(second["timestamp_utc"].unique())
    assert len(calls) == 2


@pytest.fixture()
def news_cli_args(tmp_path: Path) -> tuple[list[str], Path]:
    output_root = tmp_path / "news"
    argv = [
        "bitbat",
        "news",
        "pull",
        "--source",
        "gdelt",
        "--from",
        "2024-01-01",
        "--to",
        "2024-01-03",
        "--output",
        str(output_root),
    ]
    return argv, output_root


def _news_dataset(path: Path) -> pd.DataFrame:
    frame = read_parquet(path)
    for column in ("year", "month", "day", "hour"):
        if column in frame.columns:
            frame = frame.drop(columns=[column])
    return frame.sort_values("published_utc").reset_index(drop=True)


def test_cli_news_pull_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    news_cli_args: tuple[list[str], Path],
) -> None:
    from bitbat.ingest import news_gdelt as news_module

    argv, output_root = news_cli_args
    sample_frame = pd.DataFrame({
        "published_utc": pd.to_datetime(
            ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
            utc=False,
        ),
        "title": ["Bitcoin rally", "Regulation insights"],
        "url": ["http://example.com/a", "https://example.com/b"],
        "source": ["ExampleNews", "CryptoDaily"],
        "lang": ["en", "en"],
    })

    calls: list[tuple[datetime, datetime, str | None]] = []

    def fake_fetch(
        from_dt: datetime,
        to_dt: datetime,
        *,
        output_root: str | Path | None = None,
        throttle_seconds: float = 0.0,
        retry_limit: int = 3,
    ) -> pd.DataFrame:
        calls.append((from_dt, to_dt, output_root if output_root is None else str(output_root)))

        target = news_module._target_path(output_root)
        existing: pd.DataFrame | None = None
        if target.exists():
            existing = _news_dataset(target)

        combined = sample_frame.copy()
        if existing is not None and not existing.empty:
            combined = (
                pd.concat([existing, combined], axis=0, ignore_index=True)
                .sort_values("published_utc")
                .drop_duplicates(subset=["url"])
            )

        partitions = combined.copy()
        partitions["year"] = partitions["published_utc"].dt.year
        partitions["month"] = partitions["published_utc"].dt.month
        partitions["day"] = partitions["published_utc"].dt.day
        partitions["hour"] = partitions["published_utc"].dt.hour
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(partitions, target, partition_cols=["year", "month", "day", "hour"])
        return combined.reset_index(drop=True)

    monkeypatch.setattr("bitbat.ingest.news_gdelt.fetch", fake_fetch)

    monkeypatch.setattr(sys, "argv", argv)
    main()

    target = output_root / "gdelt_crypto_1h.parquet"
    assert target.exists()
    stored = _news_dataset(target)
    pd_testing.assert_frame_equal(stored, sample_frame)

    monkeypatch.setattr(sys, "argv", argv)
    main()
    stored_repeat = _news_dataset(target)
    pd_testing.assert_frame_equal(stored_repeat, sample_frame)

    assert len(calls) == 2


def test_cli_news_pull_cryptocompare_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from bitbat.ingest import news_cryptocompare as cc_module

    output_root = tmp_path / "news_cc"
    argv = [
        "bitbat",
        "news",
        "pull",
        "--source",
        "cryptocompare",
        "--from",
        "2024-01-01",
        "--to",
        "2024-01-03",
        "--output",
        str(output_root),
    ]
    sample_frame = pd.DataFrame({
        "published_utc": pd.to_datetime(
            ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
            utc=False,
        ),
        "title": ["Bitcoin rally", "Regulation insights"],
        "url": ["http://example.com/a", "https://example.com/b"],
        "source": ["ExampleNews", "CryptoDaily"],
        "lang": ["en", "en"],
    })

    def fake_fetch(
        from_dt: datetime,
        to_dt: datetime,
        *,
        output_root: str | Path | None = None,
        throttle_seconds: float = 0.0,
        retry_limit: int = 3,
    ) -> pd.DataFrame:
        del from_dt, to_dt, throttle_seconds, retry_limit
        target = cc_module._target_path(output_root)
        partitions = sample_frame.copy()
        partitions["year"] = partitions["published_utc"].dt.year
        partitions["month"] = partitions["published_utc"].dt.month
        partitions["day"] = partitions["published_utc"].dt.day
        partitions["hour"] = partitions["published_utc"].dt.hour
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        write_parquet(partitions, target, partition_cols=["year", "month", "day", "hour"])
        return sample_frame.copy()

    monkeypatch.setattr("bitbat.ingest.news_cryptocompare.fetch", fake_fetch)
    monkeypatch.setattr(sys, "argv", argv)
    main()

    target = output_root / "cryptocompare_btc_1h.parquet"
    assert target.exists()
    stored = _news_dataset(target)
    pd_testing.assert_frame_equal(stored, sample_frame)


def test_cli_model_cv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    freq = "1h"
    horizon = "4h"
    config_path = _write_test_config(
        tmp_path / "test_config.yaml",
        enable_sentiment=False,
    )

    feature_dir = tmp_path / "data" / "features" / f"{freq}_{horizon}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.date_range("2024-01-01 00:00:00", periods=48, freq="1h")
    dataset = pd.DataFrame({
        "timestamp_utc": idx,
        "feat_f1": np.linspace(0.0, 1.0, len(idx)),
        "label": pd.Series(
            (
                [
                    "down",
                    "flat",
                    "up",
                ]
                * 16
            )[: len(idx)],
            dtype="string",
        ),
        "r_forward": np.linspace(0.0, 0.01, len(idx)),
    })
    dataset.to_parquet(feature_dir / "dataset.parquet", index=False)

    monkeypatch.chdir(tmp_path)

    class FakeDMatrix:
        def __init__(self, data: pd.DataFrame, **kwargs: Any) -> None:
            self.data = data

    class FakeBooster:
        def predict(self, dmatrix: FakeDMatrix) -> np.ndarray:
            n = len(dmatrix.data)
            return np.tile(np.array([0.2, 0.3, 0.5]), (n, 1))

    def fake_fit_xgb(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs: Any,
    ) -> tuple[FakeBooster, dict[str, float]]:
        return FakeBooster(), {}

    def fake_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "balanced_accuracy": 0.8,
            "mcc": 0.2,
            "per_class": {},
            "pr_curves": {},
            "threshold": kwargs.get("threshold", 0.0),
        }
        (metrics_dir / "classification_metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
        (metrics_dir / "confusion_matrix.png").write_bytes(b"")
        return metrics

    monkeypatch.setattr("bitbat.cli.fit_xgb", fake_fit_xgb)
    monkeypatch.setattr("bitbat.cli.xgb.DMatrix", FakeDMatrix)
    monkeypatch.setattr("bitbat.cli.classification_metrics", fake_metrics)

    argv = [
        "bitbat",
        "--config",
        str(config_path),
        "model",
        "cv",
        "--freq",
        freq,
        "--horizon",
        horizon,
        "--start",
        "2024-01-01 00:00:00",
        "--end",
        "2024-01-03 00:00:00",
        "--windows",
        "2024-01-01 00:00:00",
        "2024-01-02 00:00:00",
        "2024-01-02 00:00:00",
        "2024-01-03 00:00:00",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    out = capsys.readouterr().out
    assert "Fold" in out
    assert "Aggregate" in out

    summary_path = Path("metrics") / "cv_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "average_balanced_accuracy" in summary


def test_cli_batch_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    freq = "1h"
    horizon = "4h"
    config_path = _write_test_config(
        tmp_path / "test_config.yaml",
        enable_sentiment=False,
    )

    prices_dir = tmp_path / "data" / "raw" / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    news_dir = tmp_path / "data" / "raw" / "news" / "gdelt_1h"
    news_dir.mkdir(parents=True, exist_ok=True)
    model_dir = tmp_path / "models" / f"{freq}_{horizon}"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "xgb.json").write_text("{}", encoding="utf-8")

    idx = pd.date_range("2024-01-01", periods=10, freq=freq)
    prices = pd.DataFrame({
        "timestamp_utc": idx,
        "open": np.linspace(100, 110, len(idx)),
        "high": np.linspace(101, 111, len(idx)),
        "low": np.linspace(99, 109, len(idx)),
        "close": np.linspace(100.5, 110.5, len(idx)),
        "volume": np.full(len(idx), 1000),
    })
    prices.to_parquet(prices_dir / "btcusd_yf_1h.parquet")

    news = pd.DataFrame({
        "published_utc": idx,
        "sentiment_score": np.linspace(-1, 1, len(idx)),
        "title": ["headline"] * len(idx),
        "url": [f"http://example.com/{i}" for i in range(len(idx))],
        "source": "UnitTest",
        "lang": "en",
    })
    news.to_parquet(news_dir / "gdelt_crypto_1h.parquet")

    def fake_fetch_prices(*args: Any, **kwargs: Any) -> None:
        return None

    def fake_fetch_news(*args: Any, **kwargs: Any) -> None:
        return None

    def fake_price_features(prices_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"feat_price": np.arange(len(prices_df))}, index=prices_df.index)

    def fake_sentiment(news_df: pd.DataFrame, bar_df: pd.DataFrame, freq: str) -> pd.DataFrame:
        index = pd.to_datetime(bar_df["timestamp_utc"])
        return pd.DataFrame({"feat_sent": np.linspace(0, 1, len(index))}, index=index)

    def fake_predict(*args: Any, **kwargs: Any) -> dict[str, Any]:
        timestamp = kwargs.get("timestamp")
        return {"timestamp": timestamp, "p_up": 0.55, "p_down": 0.25}

    class FakeModel:
        feature_names = ["feat_price"]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("bitbat.ingest.prices.fetch_yf", fake_fetch_prices)
    monkeypatch.setattr("bitbat.ingest.news_gdelt.fetch", fake_fetch_news)
    monkeypatch.setattr("bitbat.cli._generate_price_features", fake_price_features)
    monkeypatch.setattr("bitbat.cli.aggregate_sentiment", fake_sentiment)
    monkeypatch.setattr("bitbat.cli.load_model", lambda path: FakeModel())
    monkeypatch.setattr("bitbat.cli.predict_bar", fake_predict)

    argv = [
        "bitbat",
        "--config",
        str(config_path),
        "batch",
        "run",
        "--freq",
        freq,
        "--horizon",
        horizon,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    predictions_path = Path("data") / "predictions" / f"{freq}_{horizon}.parquet"
    assert predictions_path.exists()
    preds = pd.read_parquet(predictions_path)
    assert len(preds) == 1
    assert {
        "timestamp_utc",
        "p_up",
        "p_down",
        "freq",
        "horizon",
        "model_version",
        "realized_r",
        "realized_label",
    }.issubset(preds.columns)
    assert abs(preds.iloc[0]["p_up"] - 0.55) < 1e-9
    assert pd.isna(preds.iloc[0]["realized_r"])

    monkeypatch.setattr(sys, "argv", argv)
    main()
    preds_repeat = pd.read_parquet(predictions_path)
    assert len(preds_repeat) == 1


def test_cli_batch_realize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    freq = "1h"
    horizon = "4h"

    prices_dir = tmp_path / "data" / "raw" / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.date_range("2024-01-01", periods=10, freq=freq)
    close = np.linspace(100, 110, len(idx))
    prices = pd.DataFrame({
        "timestamp_utc": idx,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.full(len(idx), 1000),
    })
    prices.to_parquet(prices_dir / "btcusd_yf_1h.parquet")

    preds_dir = tmp_path / "data" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    prediction_ts = idx[4]
    preds = pd.DataFrame({
        "timestamp_utc": [prediction_ts],
        "p_up": [0.6],
        "p_down": [0.3],
        "freq": [freq],
        "horizon": [horizon],
        "model_version": ["test"],
        "realized_r": [np.nan],
        "realized_label": [pd.NA],
    })
    preds.to_parquet(preds_dir / "1h_4h.parquet", index=False)

    monkeypatch.chdir(tmp_path)

    argv = [
        "bitbat",
        "batch",
        "realize",
        "--freq",
        freq,
        "--horizon",
        horizon,
        "--tau",
        "0.01",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    realized_preds = pd.read_parquet(preds_dir / "1h_4h.parquet")
    ret = (close[4 + 4] / close[4]) - 1
    assert pytest.approx(realized_preds.loc[0, "realized_r"], abs=1e-9) == ret
    assert realized_preds.loc[0, "realized_label"] in {"up", "down", "flat"}

    monkeypatch.setattr(sys, "argv", argv)
    main()
    realized_repeat = pd.read_parquet(preds_dir / "1h_4h.parquet")
    pd_testing.assert_frame_equal(realized_preds, realized_repeat)


def test_cli_monitor_refresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    freq = "1h"
    horizon = "4h"

    predictions_dir = tmp_path / "data" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().replace(microsecond=0)
    idx = pd.date_range(now - pd.Timedelta(hours=9), periods=10, freq=freq)
    preds = pd.DataFrame({
        "timestamp_utc": idx,
        "p_up": np.linspace(0.5, 0.7, len(idx)),
        "p_down": np.linspace(0.3, 0.1, len(idx)),
        "freq": [freq] * len(idx),
        "horizon": [horizon] * len(idx),
        "model_version": ["test"] * len(idx),
        "realized_r": np.linspace(-0.01, 0.02, len(idx)),
        "realized_label": ["flat"] * len(idx),
    })
    preds.to_parquet(predictions_dir / "1h_4h.parquet", index=False)

    monkeypatch.chdir(tmp_path)

    argv = [
        "bitbat",
        "monitor",
        "refresh",
        "--freq",
        freq,
        "--horizon",
        horizon,
        "--cost_bps",
        "4",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    metrics_path = Path("metrics") / f"live_{freq}_{horizon}.json"
    assert metrics_path.exists()


def test_cli_validate_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "data" / "autonomous.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from bitbat.autonomous.models import init_database

    init_database(f"sqlite:///{db_path}")

    class FakeValidator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def validate_all(self) -> dict[str, Any]:
            return {
                "validated_count": 0,
                "correct_count": 0,
                "hit_rate": 0.0,
                "errors": [],
            }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("bitbat.autonomous.validator.PredictionValidator", FakeValidator)

    argv = [
        "bitbat",
        "validate",
        "run",
        "--freq",
        "1h",
        "--horizon",
        "4h",
        "--tau",
        "0.01",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    out = capsys.readouterr().out
    assert "Starting validation" in out
    assert "Validation complete" in out
    assert "Validated: 0 predictions" in out


def test_cli_monitor_run_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_url = f"sqlite:///{tmp_path / 'data' / 'monitor.db'}"
    config_path = _write_test_config(
        tmp_path / "monitor_config.yaml",
        enable_sentiment=False,
        database_url=db_url,
    )

    class FakeAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def run_once(self) -> dict[str, Any]:
            return {
                "validations": 2,
                "correct": 1,
                "hit_rate": 0.5,
                "drift_detected": False,
                "retraining_triggered": False,
                "validation_errors": [],
            }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("bitbat.autonomous.agent.MonitoringAgent", FakeAgent)

    argv = [
        "bitbat",
        "--config",
        str(config_path),
        "monitor",
        "run-once",
        "--freq",
        "1h",
        "--horizon",
        "4h",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    out = capsys.readouterr().out
    assert "Monitoring run completed" in out
    assert "Validations: 2" in out


def test_cli_monitor_status_and_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from bitbat.autonomous.db import AutonomousDB
    from bitbat.autonomous.models import init_database

    db_url = f"sqlite:///{tmp_path / 'data' / 'monitor_status.db'}"
    config_path = _write_test_config(
        tmp_path / "monitor_status_config.yaml",
        enable_sentiment=False,
        database_url=db_url,
    )

    init_database(db_url)
    db = AutonomousDB(db_url)

    now = datetime.now().replace(microsecond=0)
    with db.session() as session:
        db.store_model_version(
            session=session,
            version="v1",
            freq="1h",
            horizon="4h",
            training_start=now,
            training_end=now,
            training_samples=10,
            cv_score=0.6,
            features=[],
            hyperparameters={},
            training_metadata={},
            is_active=True,
        )
        db.store_performance_snapshot(
            session=session,
            model_version="v1",
            freq="1h",
            horizon="4h",
            window_days=30,
            metrics={
                "total_predictions": 10,
                "realized_predictions": 10,
                "hit_rate": 0.6,
                "sharpe_ratio": 0.5,
                "avg_return": 0.01,
                "max_drawdown": 0.1,
                "win_streak": 3,
                "lose_streak": 1,
            },
        )

    monkeypatch.chdir(tmp_path)
    argv_status = [
        "bitbat",
        "--config",
        str(config_path),
        "monitor",
        "status",
    ]
    monkeypatch.setattr(sys, "argv", argv_status)
    main()
    status_out = capsys.readouterr().out
    assert "Monitoring status" in status_out
    assert "Latest snapshot" in status_out

    argv_snapshots = [
        "bitbat",
        "--config",
        str(config_path),
        "monitor",
        "snapshots",
        "--last",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv_snapshots)
    main()
    snapshots_out = capsys.readouterr().out
    assert "Recent snapshots" in snapshots_out
