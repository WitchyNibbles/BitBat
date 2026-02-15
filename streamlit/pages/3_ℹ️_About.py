"""About / Help page — plain-English explanation of BitBat."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="About — BitBat", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About BitBat")
st.markdown("Everything you need to know about how BitBat works.")

# ------------------------------------------------------------------
# How it works
# ------------------------------------------------------------------
st.header("How It Works")
st.markdown(
    """
BitBat is an autonomous Bitcoin price prediction system that:

1. **Collects Data** — Downloads Bitcoin price history (every hour) and crypto news headlines.

2. **Analyzes Patterns** — An AI model looks for patterns in price movements and news sentiment to forecast whether Bitcoin will go **UP**, **DOWN**, or stay **FLAT**.

3. **Makes Predictions** — Every hour, the system generates a prediction for the next period (1h, 4h, or 24h ahead, depending on your settings).

4. **Validates Results** — After each prediction period passes, the system checks whether it was right and calculates accuracy.

5. **Self-Improves** — If accuracy drops below a threshold, the system automatically retrains the model using the latest data.
"""
)

# ------------------------------------------------------------------
# Understanding predictions
# ------------------------------------------------------------------
st.header("Understanding Predictions")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
**📈 Bitcoin will go UP**
The model believes the price will rise by at least the configured sensitivity amount within the forecast period.
"""
    )
with col2:
    st.markdown(
        """
**📉 Bitcoin will go DOWN**
The model believes the price will fall by at least the configured sensitivity amount within the forecast period.
"""
    )
with col3:
    st.markdown(
        """
**➡️ Bitcoin will stay FLAT**
The model doesn't see a clear directional signal — no strong prediction either way.
"""
    )

st.markdown(
    """
**Confidence** shows how certain the model is. Higher confidence = stronger signal.
- **Very High (≥75%)** — Strong signal, model is very confident
- **High (65–75%)** — Good signal, model is reasonably confident
- **Medium (55–65%)** — Weak signal, treat with caution
- **Low (<55%)** — Very uncertain, consider skipping this signal
"""
)

# ------------------------------------------------------------------
# FAQ
# ------------------------------------------------------------------
st.header("Frequently Asked Questions")

with st.expander("Is BitBat financial advice?"):
    st.warning(
        "**No.** BitBat is a research tool for exploring machine-learning-based "
        "price prediction. It is **not** financial advice. Never invest money you "
        "cannot afford to lose based on automated predictions."
    )

with st.expander("How accurate is BitBat?"):
    st.markdown(
        """
Accuracy varies depending on market conditions and your chosen preset:
- A random coin flip gives ~50% accuracy.
- BitBat typically achieves 55–70% accuracy in stable market conditions.
- During high volatility (news events, market crashes), accuracy may drop.

View real-time accuracy on the **Performance** page.
"""
    )

with st.expander("What do the presets mean?"):
    st.markdown(
        """
| Preset | Forecast Period | Risk Level | Best For |
|--------|----------------|------------|----------|
| 🛡️ Conservative | 24 hours ahead | Low | Long-term holders |
| ⚖️ Balanced | 4 hours ahead | Medium | Most users |
| 🚀 Aggressive | 1 hour ahead | High | Active traders |

Change your preset on the **Settings** page.
"""
    )

with st.expander("Why does the system retrain itself?"):
    st.markdown(
        """
Markets change over time — patterns that worked 6 months ago may not work today.
BitBat automatically detects when its accuracy drops and retrains the model using the
most recent data. This usually takes a few minutes and happens in the background.

You can see retraining history on the **Performance** page.
"""
    )

with st.expander("How do I start BitBat?"):
    st.code(
        """# Option A — Docker (recommended):
docker-compose up -d

# Option B — run directly:
poetry run python scripts/run_monitoring_agent.py  # Monitoring agent
poetry run python scripts/run_ingestion_service.py  # Data ingestion""",
        language="bash",
    )

with st.expander("Where is my data stored?"):
    st.markdown(
        """
All data is stored locally on your machine:
- `data/` — Raw price and news data (Parquet files)
- `data/autonomous.db` — Prediction history and model versions (SQLite)
- `models/` — Trained model files
- `config/` — Your settings
- `logs/` — System logs

No data is sent to external servers (except API calls to fetch price/news data).
"""
    )

# ------------------------------------------------------------------
# Links
# ------------------------------------------------------------------
st.header("Advanced Features")
st.markdown(
    """
For power users who need access to the full technical pipeline
(ingest data, build features, train models, run backtests):

👉 Use the **Advanced Pipeline** page in the sidebar.
"""
)

# ------------------------------------------------------------------
# Disclaimer
# ------------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ **Disclaimer:** BitBat is provided for educational and research purposes only. "
    "Cryptocurrency trading carries substantial risk. Past prediction accuracy does not "
    "guarantee future results. Always do your own research before making financial decisions."
)
