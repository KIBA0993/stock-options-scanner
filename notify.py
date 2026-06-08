#!/usr/bin/env python3
"""
notify.py — Trade Alert Notifier (Week 4 + 6)

Reads data/alerts.json, deduplicates against sent history, and delivers
new alerts via Telegram and/or email.

Pipeline:
  scanner.py → all_data.json → orchestrate.py → alerts.json → notify.py

Usage:
  python notify.py                         # send new alerts (Telegram + email if configured)
  python notify.py --morning-digest        # always send a daily market-open email (even if no trades)
  python notify.py --dry-run               # format and print messages without sending
  python notify.py --force                 # resend all alerts (ignore deduplication)
  python notify.py --channel email         # email only

Setup (add to config.json):
  Telegram:
    1. Message @BotFather on Telegram → /newbot → copy the token
    2. Start a chat with your bot, then run:
         curl "https://api.telegram.org/bot<TOKEN>/getUpdates"
       Copy the chat_id from the response
    3. Set in config.json:
         "notifications": {
           "telegram": { "enabled": true, "bot_token": "...", "chat_id": "..." }
         }

  Email (macOS Mail / SMTP):
    "notifications": {
      "email": {
        "enabled": true,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "you@gmail.com",
        "smtp_password": "your-app-password",
        "to": "you@gmail.com"
      }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import smtplib
import sys
from datetime import date, datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Literal, Optional

BASE_DIR     = Path.home() / "trading"
DATA_DIR     = BASE_DIR / "data"
LOG_DIR      = BASE_DIR / "logs"
CONFIG_PATH  = BASE_DIR / "config.json"
ALERTS_PATH  = DATA_DIR / "alerts.json"
BUDGET_PATH  = DATA_DIR / "budget.json"
SENT_PATH    = DATA_DIR / "sent_history.json"   # deduplication log

LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

Channel = Literal["telegram", "email", "all"]


# ─── Logging ───────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    log_file = LOG_DIR / "notify.log"
    fh = TimedRotatingFileHandler(str(log_file), when="D", backupCount=14)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[ch, fh])
    return logging.getLogger("notify")


logger = _setup_logging()


# ─── Config ────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(f"config.json not found at {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ─── Alerts ────────────────────────────────────────────────────────────────────
def load_alerts() -> dict:
    if not ALERTS_PATH.exists():
        logger.error(
            f"alerts.json not found at {ALERTS_PATH}.\n"
            "  Run the full pipeline first: trade-scan"
        )
        sys.exit(1)
    with open(ALERTS_PATH) as f:
        return json.load(f)


# ─── Deduplication ─────────────────────────────────────────────────────────────
def load_sent_history() -> dict:
    """Load the sent-alert registry. Keys are alert fingerprints."""
    if not SENT_PATH.exists():
        return {}
    with open(SENT_PATH) as f:
        return json.load(f)


def _fingerprint(alert: dict, scan_ts: str) -> str:
    """Unique identifier for an alert — symbol + direction + scan timestamp."""
    return f"{alert['symbol']}:{alert['direction']}:{scan_ts}"


def filter_new_alerts(
    alerts: list[dict],
    scan_ts: str,
    history: dict,
    force: bool,
) -> list[dict]:
    """Return only alerts that haven't been sent yet (unless --force)."""
    if force:
        logger.info("--force flag: resending all alerts")
        return alerts
    new = [a for a in alerts if _fingerprint(a, scan_ts) not in history]
    skipped = len(alerts) - len(new)
    if skipped:
        logger.info(f"Skipping {skipped} already-sent alert(s) (use --force to resend)")
    return new


def record_sent(alerts: list[dict], scan_ts: str, channels: list[str]) -> None:
    """Persist sent alert fingerprints to avoid future duplicates."""
    history = load_sent_history()
    now = datetime.now(timezone.utc).isoformat()
    for alert in alerts:
        fp = _fingerprint(alert, scan_ts)
        history[fp] = {
            "symbol":    alert["symbol"],
            "direction": alert["direction"],
            "score":     alert["score"],
            "sent_at":   now,
            "channels":  channels,
        }
    SENT_PATH.write_text(json.dumps(history, indent=2))
    logger.info(f"Recorded {len(alerts)} sent alert(s) → {SENT_PATH}")


# ─── Message Formatting ────────────────────────────────────────────────────────
def format_telegram(alert: dict, scan_ts: str, context: Optional[str]) -> str:
    """Format a single alert as a Telegram markdown message."""
    direction  = alert["direction"].upper()
    symbol     = alert["symbol"]
    score      = alert["score"]
    score_pct  = int(score * 100)
    risk       = alert.get("risk_level", "?")
    dte        = alert.get("suggested_dte") or "see rationale"
    rationale  = alert.get("rationale", "")
    creators   = alert.get("supporting_creators", [])
    signals    = alert.get("key_signals", [])
    method     = "🤖 LLM" if alert.get("scoring_method") == "llm" else "📐 Heuristic"

    dir_emoji  = "📈" if direction == "CALL" else "📉"
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")

    scan_dt = scan_ts[:16].replace("T", " ") if scan_ts else "unknown"

    creators_str = " ".join(f"@{c}" for c in creators) if creators else "heuristic"

    signals_str = ""
    for sig in signals[:5]:
        signals_str += f"\n  • {sig}"

    context_line = f"\n📌 *Context:* {context}" if context else ""

    return (
        f"{dir_emoji} *{symbol} — {direction}*\n"
        f"Score: *{score_pct}%* {risk_emoji} {risk} risk  │  {method}\n"
        f"DTE: {dte}\n"
        f"Frameworks: {creators_str}\n"
        f"\n_{rationale}_\n"
        f"\n*Signals:*{signals_str}"
        f"\n{context_line}"
        f"\n\n🕐 Scan: {scan_dt} UTC"
    )


def format_telegram_summary(
    new_alerts: list[dict],
    scan_ts: str,
    context: Optional[str],
) -> str:
    """Header message when sending multiple alerts."""
    n = len(new_alerts)
    context_str = f" │ {context}" if context else ""
    return (
        f"🔔 *{n} New Trade Alert{'s' if n != 1 else ''}*{context_str}\n"
        f"─────────────────────"
    )


def _load_budget_for_display(config: dict) -> dict:
    """Load budget info for inclusion in digest emails."""
    weekly_max  = config.get("budget", {}).get("weekly_trade_max", 5)
    total_usd   = config.get("budget", {}).get("total_usd", 500)
    per_trade   = round(total_usd / weekly_max) if weekly_max else total_usd
    used        = 0
    remaining   = weekly_max
    week_start  = date.today().isoformat()

    if BUDGET_PATH.exists():
        try:
            with open(BUDGET_PATH) as f:
                b = json.load(f)
            used      = b.get("surfaced_this_week", 0)
            remaining = max(0, weekly_max - used)
            week_start = b.get("week_start", week_start)
        except Exception:
            pass

    return {
        "weekly_max":  weekly_max,
        "total_usd":   total_usd,
        "per_trade":   per_trade,
        "used":        used,
        "remaining":   remaining,
        "week_start":  week_start,
    }


def _alert_card_html(a: dict) -> str:
    """Render a single alert as an HTML card."""
    direction = a["direction"].upper()
    dir_color = "#1a7f37" if direction == "CALL" else "#cf222e"
    dir_bg    = "#dafbe1" if direction == "CALL" else "#ffebe9"
    score_pct = int(a["score"] * 100)
    risk      = a.get("risk_level", "?")
    dte       = a.get("suggested_dte") or "see rationale"
    creators  = " ".join(f"@{c}" for c in a.get("supporting_creators", [])) or "heuristic"
    signals   = "".join(f"<li>{s}</li>" for s in a.get("key_signals", [])[:5])
    rationale = a.get("rationale", "")
    method    = "LLM" if a.get("scoring_method") == "llm" else "Heuristic"
    size_hint = a.get("position_size_hint", "")

    entry_note  = a.get("entry_note", "")
    stop_note   = a.get("stop_note", "")
    target_note = a.get("target_note", "")

    size_line = (f'<div style="margin:6px 0; font-size:13px; color:#1a7f37; font-weight:600;">'
                 f'💰 Position size: {size_hint}</div>') if size_hint else ""

    trade_plan = ""
    if entry_note or stop_note or target_note:
        rows = ""
        if entry_note:
            rows += f"""
            <tr>
              <td style="padding:5px 8px; font-weight:600; color:#57606a; white-space:nowrap; vertical-align:top;">🎯 Entry</td>
              <td style="padding:5px 8px; color:#24292f;">{entry_note}</td>
            </tr>"""
        if stop_note:
            rows += f"""
            <tr style="background:#fff5f5;">
              <td style="padding:5px 8px; font-weight:600; color:#cf222e; white-space:nowrap; vertical-align:top;">🛑 Stop</td>
              <td style="padding:5px 8px; color:#24292f;">{stop_note}</td>
            </tr>"""
        if target_note:
            rows += f"""
            <tr>
              <td style="padding:5px 8px; font-weight:600; color:#1a7f37; white-space:nowrap; vertical-align:top;">✅ Target</td>
              <td style="padding:5px 8px; color:#24292f;">{target_note}</td>
            </tr>"""
        trade_plan = f"""
        <div style="margin:12px 0 4px 0; font-size:13px; font-weight:600; color:#57606a; text-transform:uppercase; letter-spacing:.5px;">Trade Plan</div>
        <table style="width:100%; border-collapse:collapse; border:1px solid #d0d7de; border-radius:6px; overflow:hidden; font-size:13px;">
          {rows}
        </table>"""

    return f"""
    <div style="border:1px solid #d0d7de; border-radius:8px; padding:16px;
                margin-bottom:16px; font-family:system-ui,sans-serif;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:22px; font-weight:700;">{a['symbol']}</span>
        <span style="background:{dir_bg}; color:{dir_color}; font-weight:700;
                     padding:4px 12px; border-radius:12px; font-size:14px;">
          {direction}
        </span>
      </div>
      <div style="margin:8px 0; color:#57606a; font-size:14px;">
        Score: <b>{score_pct}%</b> &nbsp;│&nbsp; Risk: <b>{risk}</b>
        &nbsp;│&nbsp; DTE: <b>{dte}</b> &nbsp;│&nbsp; Scoring: {method}
      </div>
      <div style="margin:8px 0; font-size:13px; color:#57606a;">
        Frameworks: {creators}
      </div>
      {size_line}
      <p style="font-size:14px; color:#24292f; margin:10px 0 6px 0;">{rationale}</p>
      <ul style="font-size:13px; color:#57606a; margin:4px 0 8px 0; padding-left:20px;">
        {signals}
      </ul>
      {trade_plan}
    </div>
    """


def _budget_html(bud: dict) -> str:
    bar_pct = int(bud["used"] / bud["weekly_max"] * 100) if bud["weekly_max"] else 0
    bar_color = "#2da44e" if bar_pct < 60 else ("#d29922" if bar_pct < 100 else "#cf222e")
    return f"""
    <div style="border:1px solid #d0d7de; border-radius:8px; padding:12px 16px;
                margin-bottom:16px; background:#f6f8fa; font-family:system-ui,sans-serif;">
      <div style="font-size:13px; color:#57606a; margin-bottom:6px;">
        Weekly budget &nbsp;│&nbsp; ${bud['total_usd']} total
        &nbsp;│&nbsp; ~${bud['per_trade']}/trade &nbsp;│&nbsp; week of {bud['week_start']}
      </div>
      <div style="display:flex; align-items:center; gap:10px;">
        <div style="flex:1; background:#e1e4e8; border-radius:4px; height:8px;">
          <div style="width:{bar_pct}%; background:{bar_color}; height:8px; border-radius:4px;"></div>
        </div>
        <span style="font-size:13px; font-weight:600; color:{bar_color};">
          {bud['used']}/{bud['weekly_max']} trades used
        </span>
      </div>
    </div>
    """


def format_email_html(
    alerts: list[dict],
    scan_ts: str,
    context: Optional[str],
    config: Optional[dict] = None,
    morning_digest: bool = False,
) -> str:
    """Format alerts (or a no-trade digest) as an HTML email body."""
    scan_dt = scan_ts[:16].replace("T", " ") if scan_ts else "unknown"
    context_html = f"<p><b>Context:</b> {context}</p>" if context else ""
    bud = _load_budget_for_display(config or {})

    if morning_digest and not alerts:
        headline = "📋 Morning Market Digest — No trade signals today"
        body_html = f"""
        <div style="border:1px solid #d0d7de; border-radius:8px; padding:20px;
                    margin-bottom:16px; background:#fff; font-family:system-ui,sans-serif;">
          <p style="font-size:15px; color:#24292f; margin:0 0 8px 0;">
            The scanner ran at market open and found <b>no qualifying setups</b> that meet
            the current scoring threshold.
          </p>
          <p style="font-size:13px; color:#57606a; margin:0;">
            Additional scans are scheduled at <b>12:00 PM</b> and <b>2:30 PM ET</b>.
            You'll receive an email immediately if a trade signal appears.
          </p>
        </div>
        """
    else:
        n = len(alerts)
        headline = f"📊 Trade Alert{'s' if n != 1 else ''} — {n} new signal{'s' if n != 1 else ''}"
        body_html = "".join(_alert_card_html(a) for a in alerts)

    return f"""
    <html><body style="background:#f6f8fa; padding:24px; font-family:system-ui,sans-serif;">
      <div style="max-width:600px; margin:0 auto;">
        <h2 style="color:#24292f; margin-bottom:4px;">{headline}</h2>
        <p style="color:#57606a; font-size:13px; margin-top:4px;">
          Scan: {scan_dt} UTC &nbsp;│&nbsp; {date.today().strftime('%A, %B %-d, %Y')}
        </p>
        {context_html}
        {_budget_html(bud)}
        {body_html}
        <p style="color:#8c959f; font-size:12px; margin-top:24px;">
          Generated by trading/notify.py &nbsp;│&nbsp;
          Run <code>trade-scan</code> to refresh anytime
        </p>
      </div>
    </body></html>
    """


def format_email_text(
    alerts: list[dict],
    scan_ts: str,
    context: Optional[str],
    config: Optional[dict] = None,
    morning_digest: bool = False,
) -> str:
    """Plain-text fallback for email."""
    scan_dt = scan_ts[:16].replace("T", " ") if scan_ts else "unknown"
    bud = _load_budget_for_display(config or {})

    if morning_digest and not alerts:
        lines = [
            f"MORNING MARKET DIGEST — {scan_dt} UTC",
            f"Week of {bud['week_start']} | Budget: {bud['used']}/{bud['weekly_max']} trades used",
            "=" * 50,
            "",
            "No qualifying trade setups found at market open.",
            "Additional scans scheduled at 12:00 PM and 2:30 PM ET.",
            "You will receive an email immediately if a signal appears.",
        ]
        return "\n".join(lines)

    lines = [
        f"TRADE ALERTS — {scan_dt} UTC",
        f"Week of {bud['week_start']} | Budget: {bud['used']}/{bud['weekly_max']} trades | ~${bud['per_trade']}/trade",
    ]
    if context:
        lines.append(f"Context: {context}")
    lines.append("=" * 50)
    for a in alerts:
        size_hint = a.get("position_size_hint", "")
        lines += [
            f"\n{a['symbol']} — {a['direction'].upper()}",
            f"Score: {int(a['score']*100)}%  Risk: {a.get('risk_level')}  DTE: {a.get('suggested_dte')}",
        ]
        if size_hint:
            lines.append(f"Size: {size_hint}")
        lines += [
            f"Rationale: {a.get('rationale')}",
            "Signals:",
        ]
        for s in a.get("key_signals", [])[:5]:
            lines.append(f"  • {s}")
        if a.get("entry_note"):
            lines.append(f"Entry:  {a['entry_note']}")
        if a.get("stop_note"):
            lines.append(f"Stop:   {a['stop_note']}")
        if a.get("target_note"):
            lines.append(f"Target: {a['target_note']}")
    return "\n".join(lines)


# ─── Telegram Delivery ─────────────────────────────────────────────────────────
def send_telegram(
    alerts:   list[dict],
    scan_ts:  str,
    context:  Optional[str],
    tg_cfg:   dict,
    dry_run:  bool,
) -> bool:
    """Send alert messages via Telegram Bot API. Returns True on success."""
    token   = tg_cfg.get("bot_token", "")
    chat_id = tg_cfg.get("chat_id", "")

    def _send(text: str) -> bool:
        if dry_run:
            print(f"\n[DRY RUN — Telegram message]:\n{text}\n{'─'*50}")
            return True
        if not token or not chat_id:
            logger.warning("Telegram token/chat_id not set in config.json — skipping")
            return False
        try:
            import requests
        except ImportError:
            logger.error("requests not installed — run: pip install requests")
            return False
        base_url = f"https://api.telegram.org/bot{token}"
        resp = requests.post(
            f"{base_url}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if not resp.ok:
            logger.error(f"Telegram send failed: {resp.status_code} {resp.text[:200]}")
            return False
        return True

    success = True

    # Send a summary header if multiple alerts
    if len(alerts) > 1:
        summary = format_telegram_summary(alerts, scan_ts, context)
        _send(summary)

    # Send each alert individually (easier to read on mobile)
    for alert in alerts:
        msg = format_telegram(alert, scan_ts, context)
        ok = _send(msg)
        if not ok:
            success = False
        else:
            logger.info(f"Telegram sent: {alert['symbol']} {alert['direction'].upper()}")

    return success


# ─── Email Delivery ────────────────────────────────────────────────────────────
def send_email(
    alerts:         list[dict],
    scan_ts:        str,
    context:        Optional[str],
    em_cfg:         dict,
    dry_run:        bool,
    config:         Optional[dict] = None,
    morning_digest: bool = False,
) -> bool:
    """Send alert digest (or morning no-trade digest) via SMTP. Returns True on success."""
    smtp_host = em_cfg.get("smtp_host", "smtp.gmail.com")
    smtp_port = int(em_cfg.get("smtp_port", 587))
    smtp_user = em_cfg.get("smtp_user", "")
    smtp_pass = em_cfg.get("smtp_password", "")
    to_addr   = em_cfg.get("to", smtp_user)

    if not all([smtp_host, smtp_user, smtp_pass]):
        logger.warning("Email SMTP credentials not set in config.json — skipping")
        return False

    today_str = date.today().strftime("%a %b %-d")
    if morning_digest and not alerts:
        subject = f"📋 Morning Digest — No trades today ({today_str})"
    else:
        n = len(alerts)
        subject = f"📊 {n} Trade Alert{'s' if n != 1 else ''} — {today_str}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = to_addr

    plain = format_email_text(alerts, scan_ts, context, config=config, morning_digest=morning_digest)
    html  = format_email_html(alerts, scan_ts, context, config=config, morning_digest=morning_digest)
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html,  "html"))

    if dry_run:
        print(f"\n[DRY RUN — Email]:")
        print(f"  To:      {to_addr}")
        print(f"  Subject: {subject}")
        print(f"  Body:\n{plain}\n")
        return True

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_addr, msg.as_string())
        logger.info(f"Email sent to {to_addr}: {subject}")
        return True
    except Exception as exc:
        logger.error(f"Email send failed: {exc}")
        return False


# ─── Setup Helpers ─────────────────────────────────────────────────────────────
def print_setup_guide(config: dict) -> None:
    """Print actionable setup steps when no channels are configured."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              NOTIFY.PY — SETUP REQUIRED                              ║
╠══════════════════════════════════════════════════════════════════════╣

  No notification channels are enabled. To get alerts:

  ── TELEGRAM (recommended — works on mobile) ──────────────────────
  1. Open Telegram → message @BotFather → send /newbot
  2. Follow prompts, copy the bot token
  3. Start a chat with your new bot
  4. Run this to get your chat_id:
       curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
  5. Add to ~/trading/config.json:
       "notifications": {
         "telegram": {
           "enabled": true,
           "bot_token": "YOUR_BOT_TOKEN",
           "chat_id": "YOUR_CHAT_ID"
         }
       }

  ── EMAIL (Gmail) ─────────────────────────────────────────────────
  1. Enable 2FA on your Gmail account
  2. Create an App Password: myaccount.google.com/apppasswords
  3. Add to ~/trading/config.json:
       "notifications": {
         "email": {
           "enabled": true,
           "smtp_host": "smtp.gmail.com",
           "smtp_port": 587,
           "smtp_user": "you@gmail.com",
           "smtp_password": "your-16-char-app-password",
           "to": "you@gmail.com"
         }
       }

  ── TEST AFTER SETUP ──────────────────────────────────────────────
       python notify.py --dry-run    # preview without sending
       python notify.py --force      # force-send even if already sent

╚══════════════════════════════════════════════════════════════════════╝
""")


def print_no_alerts() -> None:
    print("""
  No new alerts to send.

  • If alerts exist but were already sent: use --force to resend
  • To run a fresh scan: trade-scan
  • To lower the score threshold: trade-scan-score --min-score 0.5
""")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Trade Alert Notifier")
    parser.add_argument("--morning-digest", action="store_true",
                        help="Always send an email (even if no trades) — used by 9:40 AM launchd job")
    parser.add_argument("--dry-run", action="store_true",
                        help="Format and print messages without sending")
    parser.add_argument("--force",   action="store_true",
                        help="Resend all alerts (ignore deduplication)")
    parser.add_argument("--channel", choices=["telegram", "email", "all"],
                        default="all", help="Which channel(s) to send to (default: all)")
    args = parser.parse_args()

    logger.info("=== Notifier started ===")
    config    = load_config()
    notif_cfg = config.get("notifications", {})
    tg_cfg    = notif_cfg.get("telegram", {})
    em_cfg    = notif_cfg.get("email", {})

    tg_enabled  = tg_cfg.get("enabled", False)
    em_enabled  = em_cfg.get("enabled", False)
    any_enabled = tg_enabled or em_enabled

    # Load alerts (may not exist yet on a fresh install — tolerate gracefully)
    if not ALERTS_PATH.exists() and not args.morning_digest:
        logger.error(
            f"alerts.json not found at {ALERTS_PATH}.\n"
            "  Run the full pipeline first: trade-scan"
        )
        sys.exit(1)

    all_alerts: list[dict] = []
    scan_ts = datetime.now(timezone.utc).isoformat()
    context: Optional[str] = None

    if ALERTS_PATH.exists():
        data       = load_alerts()
        all_alerts = data.get("alerts", [])
        scan_ts    = data.get("scan_timestamp", scan_ts)
        context    = data.get("context")

    logger.info(f"Loaded {len(all_alerts)} alert(s) from alerts.json")

    # ── Morning digest: always send even if 0 new alerts ────────────────────
    if args.morning_digest:
        if not any_enabled and not args.dry_run:
            print_setup_guide(config)
            return

        history    = load_sent_history()
        new_alerts = filter_new_alerts(all_alerts, scan_ts, history, force=args.force)

        sent_channels: list[str] = []
        if args.channel in ("email", "all") and (em_enabled or args.dry_run):
            ok = send_email(
                new_alerts, scan_ts, context, em_cfg,
                dry_run=args.dry_run, config=config, morning_digest=True,
            )
            if ok:
                sent_channels.append("email")

        if args.channel in ("telegram", "all") and tg_enabled and new_alerts:
            ok = send_telegram(new_alerts, scan_ts, context, tg_cfg, dry_run=args.dry_run)
            if ok:
                sent_channels.append("telegram")

        if sent_channels and not args.dry_run and new_alerts:
            record_sent(new_alerts, scan_ts, sent_channels)

        print(f"\n  ✓ Morning digest sent ({len(new_alerts)} trade alert(s)) via {sent_channels or ['(none)']}")
        return

    # ── Normal flow (opportunity alerts) ────────────────────────────────────
    if not any_enabled and not args.dry_run:
        if all_alerts:
            print_setup_guide(config)
            print("  Alert preview (not sent — no channels configured):\n")
            for a in all_alerts:
                print(format_telegram(a, scan_ts, context))
                print()
        else:
            print_no_alerts()
        return

    history    = load_sent_history()
    new_alerts = filter_new_alerts(all_alerts, scan_ts, history, force=args.force)

    if not new_alerts:
        print_no_alerts()
        return

    logger.info(f"Sending {len(new_alerts)} new alert(s) …")

    sent_channels = []
    success = True

    if args.channel in ("telegram", "all") and (tg_enabled or args.dry_run):
        ok = send_telegram(new_alerts, scan_ts, context, tg_cfg, dry_run=args.dry_run)
        if ok:
            sent_channels.append("telegram")
        else:
            success = False

    if args.channel in ("email", "all") and (em_enabled or args.dry_run):
        ok = send_email(
            new_alerts, scan_ts, context, em_cfg,
            dry_run=args.dry_run, config=config, morning_digest=False,
        )
        if ok:
            sent_channels.append("email")
        else:
            success = False

    if not sent_channels and not args.dry_run:
        print_setup_guide(config)
        print("  Alert preview:\n")
        for a in new_alerts:
            print(format_telegram(a, scan_ts, context))
            print()
        return

    if sent_channels and not args.dry_run:
        record_sent(new_alerts, scan_ts, sent_channels)

    print(f"\n{'─'*60}")
    if args.dry_run:
        print(f"  [DRY RUN] Would send {len(new_alerts)} alert(s)")
    else:
        status = "✓" if success else "⚠ partial"
        print(f"  {status} Sent {len(new_alerts)} alert(s) via {', '.join(sent_channels)}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
