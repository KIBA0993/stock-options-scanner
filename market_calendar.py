#!/usr/bin/env python3
"""
market_calendar.py — US equity market session calendar (NYSE).

Used by intraday_0dte.py, NAS cron guards, and daily reflect.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# NYSE full-day closures (no regular session)
_NYSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}


def is_trading_day(d: date | None = None) -> bool:
    """True if NYSE has a regular session on `d` (weekday, not a full holiday)."""
    d = d or date.today()
    if d.weekday() >= 5:
        return False
    return d not in _NYSE_HOLIDAYS


def is_last_trading_day_of_week(d: date | None = None) -> bool:
    """True on the final NYSE session of the ISO week (Mon–Sun)."""
    d = d or date.today()
    if not is_trading_day(d):
        return False
    week_end = d - timedelta(days=d.weekday()) + timedelta(days=6)
    probe = d + timedelta(days=1)
    while probe <= week_end:
        if is_trading_day(probe):
            return False
        probe += timedelta(days=1)
    return True


def last_trading_day_on_or_before(d: date | None = None) -> date:
    """Most recent NYSE session on or before `d` (walks back over weekends/holidays)."""
    d = d or date.today()
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def is_market_hours(dt: datetime | None = None) -> bool:
    """True during regular session 9:30–16:00 ET on a trading day."""
    dt = dt or datetime.now(ET)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    else:
        dt = dt.astimezone(ET)
    if not is_trading_day(dt.date()):
        return False
    t = dt.time()
    return MARKET_OPEN <= t < MARKET_CLOSE


def trading_day_or_exit() -> None:
    """Exit 0 silently when market is closed (for cron wrappers)."""
    import sys
    if not is_trading_day():
        sys.exit(2)
