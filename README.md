<!--
BitBat is a bat.
Bats see patterns in the dark.
This is not financial advice.
-->

<div align="center">

# 🦇 BitBat

*an experimental AI oracle for Bitcoin price direction*

Forecasting short-horizon BTC price movements
using machine learning, market data, and a healthy fear of leakage.

**she / they** 🏳️‍⚧️
🕯️ witchy • careful • contract-driven

</div>

---

## 🕯️ what is this?

**BitBat** is a personal research project exploring **directional Bitcoin price prediction**
through a fully contract-enforced, end-to-end ML pipeline.

It focuses on:
- disciplined data ingestion
- leakage-safe feature engineering
- reproducible model training
- and boring, trustworthy evaluation

No hype. No black-box miracles. Just structured curiosity.

---

## 🩸 what BitBat does

- Ingests **BTC OHLCV market data** and **crypto-related news**
- Engineers **price & sentiment features** with strict time alignment
- Trains **gradient-boosted classifiers (XGBoost)** for directional moves
- Produces **contract-compliant predictions** with realized outcomes
- Supports **backtesting, batch inference, and live monitoring**
- Enforces **schema contracts and guardrails** at every step

If data violates expectations, the pipeline stops.
If metrics look suspicious, it complains loudly.

As it should.

---

## 🕸️ project structure (high level)

```text
data/         local data artifacts (generated)
docs/         full project documentation
models/       trained model checkpoints
predictions/  validated prediction outputs
src/bitbat/    core pipeline code & CLI
tests/        pytest guardrails against leakage & regression
```

📖 Full documentation lives in docs/
Start with [docs/README.md](docs/README.md) for the complete system overview and usage guides.

---

## 🔮 quickstart

```ssh
poetry install
poetry run bitbat --help
```

From there, the documentation walks through:
- ingestion
- feature engineering
- training & evaluation
- batch inference
- monitoring & backtesting

Step by step. No guesswork.

---

## 🧭 Streamlit UI

**Prerequisites:** `poetry install` (Streamlit ships with the core dependencies).

Run the web UI:
```bash
poetry run streamlit run streamlit/app.py
# or
make streamlit
```

The UI mirrors the CLI workflows (ingest → features → model → predictions → monitor/backtest)
and uses the same config values: `data_dir`, `freq`, `horizon`, and `tau`
from `src/bitbat/config/default.yaml` (or `BITBAT_CONFIG` overrides).

Outputs land in the same locations as the CLI:
- `${data_dir}/raw/` for ingested data
- `${data_dir}/features/{freq}_{horizon}/` for datasets + metadata
- `models/{freq}_{horizon}/` for trained models
- `${data_dir}/predictions/{freq}_{horizon}.parquet` for predictions
- `metrics/live_{freq}_{horizon}.json` for monitoring snapshots

---

## 🕯️ design principles

- contracts over vibes
- reproducibility over speed
- guardrails over optimism
- boring ML is good ML

Legacy systems are haunted.
Financial data is worse.
BitBat treats both accordingly.

---

## ⚠️ disclaimer

This project is **experimental and educational**.

It is **not financial advice**,
not a trading system,
and not a promise of profit.

If you lose money listening to a bat,
that’s between you and the moon. 🌙

---

<div align="center">

🕯️ the night is noisy — choose your signals carefully 🕯️

</div>
