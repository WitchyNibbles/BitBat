"""Alert routing for autonomous monitoring events."""

from __future__ import annotations

import json
import logging
import smtplib
from email.message import EmailMessage
from typing import Any

import requests

from bitbat.config.loader import get_runtime_config, load_config

logger = logging.getLogger(__name__)


def _alert_config() -> dict[str, Any]:
    config = get_runtime_config() or load_config()
    autonomous = config.get("autonomous", {})
    return autonomous.get("alerts", {})


def _details_text(details: dict[str, Any] | None) -> str:
    if not details:
        return ""
    return json.dumps(details, indent=2, default=str)


def send_email_alert(level: str, message: str, details: dict[str, Any] | None = None) -> bool:
    """Send alert through SMTP."""
    cfg = _alert_config()
    to_address = str(cfg.get("email_to", "")).strip()
    smtp_server = str(cfg.get("smtp_server", "")).strip()
    smtp_username = str(cfg.get("smtp_username", "")).strip()
    smtp_password = str(cfg.get("smtp_password", "")).strip()
    smtp_port = int(cfg.get("smtp_port", 587))

    if not to_address or not smtp_server:
        logger.warning("Email alert not configured; skipping send.")
        return False

    email = EmailMessage()
    email["Subject"] = f"[BitBat][{level.upper()}] {message}"
    email["From"] = smtp_username or "bitbat@localhost"
    email["To"] = to_address
    body = message
    details_blob = _details_text(details)
    if details_blob:
        body = f"{body}\n\nDetails:\n{details_blob}"
    email.set_content(body)

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as smtp:
            smtp.starttls()
            if smtp_username:
                smtp.login(smtp_username, smtp_password)
            smtp.send_message(email)
        return True
    except Exception as exc:
        logger.error("Failed to send email alert: %s", exc)
        return False


def send_discord_alert(level: str, message: str, details: dict[str, Any] | None = None) -> bool:
    """Send alert through Discord webhook."""
    cfg = _alert_config()
    webhook = str(cfg.get("discord_webhook_url", "") or cfg.get("slack_webhook_url", "")).strip()
    if not webhook:
        logger.warning("Discord webhook not configured; skipping send.")
        return False

    payload = {"content": f"**[{level.upper()}]** {message}"}
    details_blob = _details_text(details)
    if details_blob:
        payload["content"] += f"\n```json\n{details_blob}\n```"

    try:
        response = requests.post(webhook, json=payload, timeout=15)
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.error("Failed to send Discord alert: %s", exc)
        return False


def send_telegram_alert(level: str, message: str, details: dict[str, Any] | None = None) -> bool:
    """Send alert through Telegram bot API."""
    cfg = _alert_config()
    token = str(cfg.get("telegram_bot_token", "")).strip()
    chat_id = str(cfg.get("telegram_chat_id", "")).strip()
    if not token or not chat_id:
        logger.warning("Telegram alert not configured; skipping send.")
        return False

    details_blob = _details_text(details)
    text = f"[{level.upper()}] {message}"
    if details_blob:
        text += f"\n\n{details_blob}"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.error("Failed to send Telegram alert: %s", exc)
        return False


def send_alert(level: str, message: str, details: dict[str, Any] | None = None) -> dict[str, bool]:
    """Route alert to configured channels and return per-channel status."""
    cfg = _alert_config()
    level_norm = level.upper()

    sent: dict[str, bool] = {
        "email": False,
        "discord": False,
        "telegram": False,
    }

    if bool(cfg.get("email_enabled", False)):
        sent["email"] = send_email_alert(level_norm, message, details)

    discord_enabled = bool(cfg.get("discord_enabled", False) or cfg.get("slack_enabled", False))
    if discord_enabled:
        sent["discord"] = send_discord_alert(level_norm, message, details)

    if bool(cfg.get("telegram_enabled", False)):
        sent["telegram"] = send_telegram_alert(level_norm, message, details)

    return sent
