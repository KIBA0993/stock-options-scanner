"""Tests for market_calendar.py"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import market_calendar as mc

ET = ZoneInfo("America/New_York")


def test_weekday_trading_day():
    assert mc.is_trading_day(date(2026, 6, 18)) is True


def test_weekend_not_trading():
    assert mc.is_trading_day(date(2026, 6, 14)) is False


def test_nyse_holiday():
    assert mc.is_trading_day(date(2026, 12, 25)) is False


def test_market_hours_on_holiday():
    dt = datetime(2026, 12, 25, 11, 0, tzinfo=ET)
    assert mc.is_market_hours(dt) is False


def test_trading_day_or_exit_on_holiday():
    from unittest.mock import patch
    import pytest

    with patch.object(mc, "is_trading_day", return_value=False):
        with pytest.raises(SystemExit) as exc:
            mc.trading_day_or_exit()
        assert exc.value.code == 2
