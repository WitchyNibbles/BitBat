"""Alerts & Notifications page — configure and test alert channels."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bitbat.gui.widgets import db_query

st.set_page_config(page_title="Alerts — BitBat", page_icon="🔔", layout="wide")

st.title("🔔 Alerts & Notifications")
st.markdown("Configure where BitBat sends you alerts when something important happens.")

_DB = ROOT / "data" / "autonomous.db"
_RULES_PATH = ROOT / "config" / "alert_rules.yaml"

# ------------------------------------------------------------------
# Load / save helpers
# ------------------------------------------------------------------


def _load_rules() -> dict:
    if _RULES_PATH.exists():
        try:
            return yaml.safe_load(_RULES_PATH.read_text()) or {}
        except Exception:
            pass
    return {}


def _save_rules(rules: dict) -> None:
    _RULES_PATH.parent.mkdir(exist_ok=True)
    _RULES_PATH.write_text(yaml.dump(rules, sort_keys=False))


rules = _load_rules()

# ------------------------------------------------------------------
# Notification channels
# ------------------------------------------------------------------
st.header("Notification Channels")
st.markdown("Choose where you want to receive alerts:")

channels = rules.get("channels", {})

tabs = st.tabs(["📧 Email", "💬 Discord", "📱 Telegram"])

# ---- Email ----
with tabs[0]:
    email_cfg = channels.get("email", {})
    email_enabled = st.toggle(
        "Enable Email Alerts", value=bool(email_cfg.get("enabled", False)), key="email_toggle"
    )
    if email_enabled:
        st.info(
            "Email uses Gmail SMTP. Set `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD` "
            "environment variables (use an App Password, not your real password)."
        )
        email_addr = st.text_input(
            "Gmail Address",
            value=email_cfg.get("address", os.environ.get("GMAIL_ADDRESS", "")),
            placeholder="you@gmail.com",
        )
        gmail_env = os.environ.get("GMAIL_ADDRESS", "")
        gmail_pass_env = os.environ.get("GMAIL_APP_PASSWORD", "")
        if gmail_env and gmail_pass_env:
            st.success("✅ Gmail credentials detected in environment variables.")
        else:
            st.warning("⚠️ Gmail environment variables not set. Alerts will not be sent until configured.")

        if st.button("Send Test Email", key="test_email"):
            if not gmail_env or not gmail_pass_env:
                st.error("Cannot send test: GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set.")
            else:
                try:
                    from bitbat.autonomous.alerting import AlertManager
                    manager = AlertManager()
                    manager.send_email(
                        subject="BitBat Test Alert",
                        body="This is a test alert from BitBat. Your email alerts are working!",
                    )
                    st.success("✅ Test email sent!")
                except Exception as exc:
                    st.error(f"Failed to send test email: {exc}")
    else:
        email_addr = email_cfg.get("address", "")

# ---- Discord ----
with tabs[1]:
    discord_cfg = channels.get("discord", {})
    discord_enabled = st.toggle(
        "Enable Discord Alerts", value=bool(discord_cfg.get("enabled", False)), key="discord_toggle"
    )
    if discord_enabled:
        st.info(
            "Create a Discord webhook in your server settings and set "
            "`DISCORD_WEBHOOK_URL` environment variable, or paste it below."
        )
        discord_url = st.text_input(
            "Webhook URL",
            value=discord_cfg.get("webhook_url", os.environ.get("DISCORD_WEBHOOK_URL", "")),
            placeholder="https://discord.com/api/webhooks/...",
            type="password",
        )
        if st.button("Send Test Discord Message", key="test_discord"):
            url = discord_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
            if not url:
                st.error("No webhook URL provided.")
            else:
                try:
                    import requests
                    resp = requests.post(url, json={"content": "🦇 BitBat test alert — Discord is working!"}, timeout=10)
                    resp.raise_for_status()
                    st.success("✅ Test Discord message sent!")
                except Exception as exc:
                    st.error(f"Failed to send Discord message: {exc}")
    else:
        discord_url = discord_cfg.get("webhook_url", "")

# ---- Telegram ----
with tabs[2]:
    tg_cfg = channels.get("telegram", {})
    tg_enabled = st.toggle(
        "Enable Telegram Alerts", value=bool(tg_cfg.get("enabled", False)), key="tg_toggle"
    )
    if tg_enabled:
        st.info(
            "Create a bot with @BotFather, then set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` "
            "environment variables."
        )
        tg_token = st.text_input(
            "Bot Token",
            value=tg_cfg.get("bot_token", os.environ.get("TELEGRAM_BOT_TOKEN", "")),
            placeholder="123456:ABC-DEF...",
            type="password",
        )
        tg_chat_id = st.text_input(
            "Chat ID",
            value=tg_cfg.get("chat_id", os.environ.get("TELEGRAM_CHAT_ID", "")),
            placeholder="-100...",
        )
        if st.button("Send Test Telegram Message", key="test_tg"):
            token = tg_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat = tg_chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
            if not token or not chat:
                st.error("Both Bot Token and Chat ID are required.")
            else:
                try:
                    import requests
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    resp = requests.post(
                        url,
                        json={"chat_id": chat, "text": "🦇 BitBat test alert — Telegram is working!"},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    st.success("✅ Test Telegram message sent!")
                except Exception as exc:
                    st.error(f"Failed to send Telegram message: {exc}")
    else:
        tg_token = tg_cfg.get("bot_token", "")
        tg_chat_id = tg_cfg.get("chat_id", "")

# ------------------------------------------------------------------
# Alert rules editor
# ------------------------------------------------------------------
st.divider()
st.header("Alert Rules")
st.markdown("Choose which events trigger a notification:")

rule_defs = rules.get("rules", {})

r1, r2 = st.columns(2)
with r1:
    acc_drop = st.toggle(
        "Accuracy drops too low",
        value=bool(rule_defs.get("accuracy_drop", {}).get("enabled", True)),
        help="Alert when prediction accuracy falls below the threshold.",
    )
    acc_threshold = 0.50
    if acc_drop:
        acc_threshold = st.slider(
            "Minimum acceptable accuracy",
            min_value=0.40,
            max_value=0.75,
            value=float(rule_defs.get("accuracy_drop", {}).get("threshold", 0.50)),
            step=0.01,
            format="%.0f%%",
        )

    drift_alert = st.toggle(
        "Model drift detected",
        value=bool(rule_defs.get("drift_detected", {}).get("enabled", True)),
        help="Alert when the AI detects its predictions are becoming less reliable.",
    )

with r2:
    retrain_start = st.toggle(
        "Retraining starts",
        value=bool(rule_defs.get("retraining_started", {}).get("enabled", True)),
        help="Alert when the model begins retraining itself.",
    )
    retrain_done = st.toggle(
        "Retraining completes",
        value=bool(rule_defs.get("retraining_completed", {}).get("enabled", True)),
        help="Alert when retraining finishes (success or failure).",
    )
    high_conf = st.toggle(
        "High-confidence prediction",
        value=bool(rule_defs.get("high_confidence_prediction", {}).get("enabled", False)),
        help="Alert on very high-confidence predictions only.",
    )
    high_conf_threshold = 0.80
    if high_conf:
        high_conf_threshold = st.slider(
            "Minimum confidence for alert",
            min_value=0.65,
            max_value=0.95,
            value=float(rule_defs.get("high_confidence_prediction", {}).get("min_confidence", 0.80)),
            step=0.05,
            format="%.0f%%",
        )

    daily_summary = st.toggle(
        "Daily summary",
        value=bool(rule_defs.get("daily_summary", {}).get("enabled", False)),
        help="Receive a daily digest of predictions and performance.",
    )

# ------------------------------------------------------------------
# Save settings
# ------------------------------------------------------------------
st.divider()
if st.button("💾 Save Alert Settings", type="primary", width="stretch"):
    new_rules = {
        "channels": {
            "email": {
                "enabled": email_enabled,
                "address": locals().get("email_addr", ""),
            },
            "discord": {
                "enabled": discord_enabled,
                "webhook_url": locals().get("discord_url", ""),
            },
            "telegram": {
                "enabled": tg_enabled,
                "bot_token": locals().get("tg_token", ""),
                "chat_id": locals().get("tg_chat_id", ""),
            },
        },
        "rules": {
            "accuracy_drop": {"enabled": acc_drop, "threshold": acc_threshold},
            "drift_detected": {"enabled": drift_alert},
            "retraining_started": {"enabled": retrain_start},
            "retraining_completed": {"enabled": retrain_done},
            "high_confidence_prediction": {
                "enabled": high_conf,
                "min_confidence": high_conf_threshold,
            },
            "daily_summary": {"enabled": daily_summary},
        },
    }
    _save_rules(new_rules)
    st.success("✅ Alert settings saved!")
    st.info("Changes take effect on the next monitoring cycle.")

# ------------------------------------------------------------------
# In-app notification summary (sidebar badge)
# ------------------------------------------------------------------
unread_count = 0
recent_alerts = db_query(
    _DB,
    "SELECT created_at, level, message FROM system_logs "
    "WHERE level IN ('WARNING','ERROR') "
    "ORDER BY created_at DESC LIMIT 5",
)
unread_count = len(recent_alerts)

if unread_count > 0:
    with st.sidebar:
        st.markdown(f"### 🔔 {unread_count} Recent Alerts")
        for row in recent_alerts:
            icon = "⚠️" if row[1] == "WARNING" else "❌"
            st.caption(f"{icon} {row[0]} — {row[2]}")

# ------------------------------------------------------------------
# Alert history
# ------------------------------------------------------------------
st.divider()
st.header("Recent Alerts Sent")

alert_rows = db_query(
    _DB,
    "SELECT created_at, level, message FROM system_logs "
    "WHERE level IN ('WARNING','ERROR') "
    "ORDER BY created_at DESC LIMIT 20",
)

if alert_rows:
    alert_df = pd.DataFrame(alert_rows, columns=["Time", "Severity", "Message"])
    alert_df["Severity"] = alert_df["Severity"].map(
        {"WARNING": "⚠️ Warning", "ERROR": "❌ Error"}
    ).fillna(alert_df["Severity"])
    st.dataframe(alert_df, width="stretch", hide_index=True)
else:
    st.info(
        "No alerts recorded yet. Alerts will appear here once the monitoring system "
        "detects issues or significant events."
    )

# ------------------------------------------------------------------
# Help
# ------------------------------------------------------------------
with st.expander("❓ Help — Setting Up Alerts"):
    st.markdown(
        """
**Email (Gmail)**
1. Enable 2-Factor Authentication on your Google account
2. Go to Security → App Passwords → Generate an app password
3. Set environment variables before starting BitBat:
```bash
export GMAIL_ADDRESS="you@gmail.com"
export GMAIL_APP_PASSWORD="your-app-password"
```

**Discord**
1. Open your Discord server settings → Integrations → Webhooks
2. Create a new webhook and copy the URL
3. Set the environment variable:
```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

**Telegram**
1. Message @BotFather on Telegram → /newbot → follow instructions
2. Get your Chat ID by messaging @userinfobot
3. Set environment variables:
```bash
export TELEGRAM_BOT_TOKEN="123456:ABC..."
export TELEGRAM_CHAT_ID="-100..."
```
"""
    )
