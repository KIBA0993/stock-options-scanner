#!/usr/bin/env python3
"""
utils.py — Shared utilities for the trading pipeline.

Extracted from journal.py and orchestrate.py to avoid duplication.
Imported by: journal.py, orchestrate.py, reflect.py
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

BASE_DIR    = Path.home() / "trading"
DATA_DIR    = BASE_DIR / "data"
BUDGET_PATH = DATA_DIR / "budget.json"

WEEKLY_BUDGET = 10   # max alerts to surface per week

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
