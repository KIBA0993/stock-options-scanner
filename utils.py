#!/usr/bin/env python3
"""
utils.py — Shared utilities for the trading pipeline.

Extracted from journal.py and orchestrate.py to avoid duplication.
Imported by: journal.py, orchestrate.py, reflect.py
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

BASE_DIR     = Path.home() / "trading"
DATA_DIR     = BASE_DIR / "data"
ARCHIVE_DIR  = DATA_DIR / "archive"
BUDGET_PATH  = DATA_DIR / "budget.json"
SENT_PATH    = DATA_DIR / "sent_history.json"

WEEKLY_BUDGET = 10   # legacy default when config omits weekly_trade_max (unused if cap disabled)


def get_weekly_trade_cap(config: dict | None = None) -> int | None:
    """Max alerts per week from config.json. None = unlimited (cap disabled)."""
    if not config:
        return None
    cap = config.get("budget", {}).get("weekly_trade_max")
    if cap is None:
        return None
    cap = int(cap)
    return cap if cap > 0 else None

logger = logging.getLogger(__name__)


def monday_of_week(d: date) -> date:
    """Return the Monday of the ISO week containing `d`."""
    return d - timedelta(days=d.weekday())


def load_budget() -> dict:
    """Load weekly budget from budget.json, auto-creating or auto-resetting as needed."""
    if not BUDGET_PATH.exists():
        budget = {
            "week_start":         monday_of_week(date.today()).isoformat(),
            "surfaced_this_week": 0,
        }
        BUDGET_PATH.write_text(json.dumps(budget, indent=2))
        return budget

    with open(BUDGET_PATH) as f:
        budget = json.load(f)

    week_start = date.fromisoformat(budget["week_start"])
    if (date.today() - week_start).days >= 7:
        budget = {
            "week_start":         monday_of_week(date.today()).isoformat(),
            "surfaced_this_week": 0,
        }
        BUDGET_PATH.write_text(json.dumps(budget, indent=2))

    return budget


def save_budget(budget: dict, new_count: int) -> None:
    """Persist updated surfaced_this_week count."""
    budget["surfaced_this_week"] = new_count
    BUDGET_PATH.write_text(json.dumps(budget, indent=2))


def rsi_bucket(rsi: float | None) -> str:
    """Classify an RSI value into a named bucket for pattern matching."""
    if rsi is None:
        return "unknown"
    if rsi < 15:
        return "extreme_oversold"
    if rsi < 30:
        return "low"
    if rsi <= 70:
        return "neutral"
    if rsi <= 80:
        return "elevated_70_80"
    return "extreme_overbought"


def momentum_bucket(pct_5d: float | None) -> str:
    """Classify a 5-day price return into a named bucket for pattern matching."""
    if pct_5d is None:
        return "unknown"
    abs_pct = abs(pct_5d)
    if abs_pct < 2:
        return "flat"
    if abs_pct < 5:
        return "mild"
    if abs_pct < 10:
        return "extended"
    return "parabolic"


def rel_vol_bucket(rel_vol: float | None) -> str:
    """Classify relative volume into a named bucket for pattern matching."""
    if rel_vol is None:
        return "unknown"
    if rel_vol < 2:
        return "normal"
    if rel_vol < 4:
        return "elevated"
    return "extreme"


def load_sent_history() -> dict:
    """Load notify.py deduplication log (symbol alerts actually emailed)."""
    if not SENT_PATH.exists():
        return {}
    try:
        return json.loads(SENT_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load_week_sent_alerts(week_start: date) -> list[dict]:
    """
    Unique emailed alerts for the ISO week containing week_start (Mon–Sun).
    Dedupes by (symbol, direction), keeping the earliest send time.
    """
    week_end = week_start + timedelta(days=6)
    deduped: dict[tuple[str, str], dict] = {}

    for _key, rec in load_sent_history().items():
        sent_at = datetime.fromisoformat(rec["sent_at"].replace("Z", "+00:00"))
        sent_date = sent_at.date()
        if not (week_start <= sent_date <= week_end):
            continue
        sym = rec["symbol"].upper()
        direction = rec["direction"].lower()
        slot = (sym, direction)
        if slot not in deduped or sent_at < deduped[slot]["sent_at"]:
            deduped[slot] = {
                "symbol":    sym,
                "direction": direction,
                "score":     rec.get("score"),
                "sent_at":   sent_at,
                "sent_date": sent_date,
            }

    return sorted(deduped.values(), key=lambda r: r["sent_at"])


def find_archive_alert(
    week_start: date,
    symbol: str,
    direction: str,
) -> Optional[dict]:
    """Best-scoring archived call/put alert for symbol in the given week."""
    if not ARCHIVE_DIR.exists():
        return None

    week_end = week_start + timedelta(days=6)
    symbol = symbol.upper()
    direction = direction.lower()
    best: Optional[dict] = None
    best_score = -1.0

    for f in ARCHIVE_DIR.glob("scored-*.json"):
        try:
            date_str = f.stem.split("-")[1]
            file_date = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        except (IndexError, ValueError):
            continue
        if not (week_start <= file_date <= week_end):
            continue
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        scan_ts = data.get("scan_timestamp", "")
        for rec in data.get("all_scored", []):
            if (rec.get("symbol") or "").upper() != symbol:
                continue
            if rec.get("direction", "").lower() != direction:
                continue
            score = float(rec.get("score") or 0)
            if score > best_score:
                best_score = score
                best = dict(rec)
                best["_scan_date"] = scan_ts

    return best


def _archive_has_contract(rec: dict) -> bool:
    rc = rec.get("recommended_contract") or {}
    tiers = rc.get("tiers") or {}
    for key in ("atm", "slight_otm", "affordable"):
        tier = tiers.get(key)
        if tier and float(tier.get("mid_price") or 0) > 0:
            return True
    return False


def find_archive_alert_on_date(
    symbol: str,
    direction: str,
    alert_date: date,
) -> Optional[dict]:
    """Best alert for symbol+direction from scored archives on a specific day."""
    if not ARCHIVE_DIR.exists():
        return None

    symbol = symbol.upper()
    direction = direction.lower()
    prefix = f"scored-{alert_date.strftime('%Y%m%d')}-"
    best_with_contract: Optional[dict] = None
    best_with_contract_score = -1.0
    best_any: Optional[dict] = None
    best_any_score = -1.0

    for f in sorted(ARCHIVE_DIR.glob(f"{prefix}*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        scan_ts = data.get("scan_timestamp", "")
        candidates = list(data.get("alerts", [])) + list(data.get("all_scored", []))
        for rec in candidates:
            if (rec.get("symbol") or "").upper() != symbol:
                continue
            if rec.get("direction", "").lower() != direction:
                continue
            score = float(rec.get("score") or 0)
            enriched = dict(rec)
            enriched["_scan_date"] = scan_ts
            if score > best_any_score:
                best_any_score = score
                best_any = enriched
            if _archive_has_contract(rec) and score > best_with_contract_score:
                best_with_contract_score = score
                best_with_contract = enriched

    if best_with_contract:
        return best_with_contract
    if best_any:
        return best_any

    week_start = alert_date - timedelta(days=alert_date.weekday())
    return find_archive_alert(week_start, symbol, direction)
