"""
test_intraday_0dte.py — Unit tests for rule-based 0–1 DTE intraday scanner.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

TRADING_DIR = Path.home() / "trading"
if str(TRADING_DIR) not in sys.path:
    sys.path.insert(0, str(TRADING_DIR))

import intraday_0dte as i0
import market_calendar as mc

ET = ZoneInfo("America/New_York")


@pytest.fixture
def tmp_trading(tmp_path, monkeypatch):
    monkeypatch.setattr(i0, "BASE_DIR", tmp_path)
    monkeypatch.setattr(i0, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(i0, "ARCHIVE_DIR", tmp_path / "data" / "archive")
    monkeypatch.setattr(i0, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(i0, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(i0, "ALERTS_PATH", tmp_path / "data" / "intraday_0dte_alerts.jsonl")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _make_bars(n: int = 20, trend: str = "up") -> pd.DataFrame:
    """Synthetic 5m bars for today."""
    base = datetime(2026, 6, 17, 9, 35, tzinfo=ET)
    idx = pd.date_range(base, periods=n, freq="5min", tz=ET)
    closes = [100.0 + i * 0.15 for i in range(n)] if trend == "up" else [100.0 - i * 0.15 for i in range(n)]
    return pd.DataFrame({
        "Open": closes,
        "High": [c + 0.1 for c in closes],
        "Low": [c - 0.1 for c in closes],
        "Close": closes,
        "Volume": [500_000] * n,
    }, index=idx)


class TestMarketHours:
    def test_weekday_during_session(self):
        dt = datetime(2026, 6, 17, 11, 0, tzinfo=ET)  # Wed
        assert mc.is_market_hours(dt) is True

    def test_weekend(self):
        dt = datetime(2026, 6, 14, 12, 0, tzinfo=ET)  # Sun
        assert mc.is_market_hours(dt) is False

    def test_before_open(self):
        dt = datetime(2026, 6, 17, 9, 0, tzinfo=ET)
        assert mc.is_market_hours(dt) is False


class TestScoring:
    def test_bullish_setup_scores_call(self):
        bars = _make_bars(25, "up")
        options = {"call_put_ratio": 1.3, "calls": [], "puts": []}
        cfg = {"min_score": 0.5, "or_minutes": 15, "min_relative_volume": 0.5}
        result = i0.score_symbol("SPY", bars, options, cfg)
        assert result["direction"] in ("call", "skip")
        if result["direction"] == "call":
            assert result["score"] >= 0.5
            assert result["suggested_dte"] == "0-1 days"

    def test_insufficient_bars_skip(self):
        result = i0.score_symbol("SPY", pd.DataFrame(), {}, {})
        assert result["direction"] == "skip"


class TestAlertIO:
    def test_append_and_load(self, tmp_trading):
        alert = {
            "symbol": "SPY",
            "direction": "call",
            "scan_timestamp": "2026-06-17T10:30:00-04:00",
            "score": 0.72,
        }
        i0.append_alert(alert)
        loaded = i0.load_alerts()
        assert len(loaded) == 1
        assert loaded[0]["symbol"] == "SPY"

    def test_dedup_blocks_repeat(self, tmp_trading, monkeypatch):
        now = datetime(2026, 6, 17, 11, 0, tzinfo=ET)
        monkeypatch.setattr(i0, "now_et", lambda: now)
        monkeypatch.setattr(i0, "date", type("D", (), {"today": staticmethod(lambda: date(2026, 6, 17))})())
        i0.append_alert({
            "symbol": "QQQ",
            "direction": "put",
            "alert_action": "entry",
            "scan_timestamp": (now - timedelta(minutes=10)).isoformat(),
            "score": 0.7,
        })
        candidate = {
            "symbol": "QQQ",
            "direction": "put",
            "alert_action": "entry",
            "score": 0.75,
        }
        assert i0.should_fire_alert(candidate, dedup_minutes=30) is False

    def test_load_week_filters(self, tmp_trading):
        ws = date(2026, 6, 16)  # Monday
        i0.append_alert({
            "symbol": "SPY",
            "direction": "call",
            "alert_action": "entry",
            "scan_timestamp": "2026-06-17T10:00:00-04:00",
            "score": 0.7,
        })
        i0.append_alert({
            "symbol": "QQQ",
            "direction": "put",
            "alert_action": "exit",
            "scan_timestamp": "2026-06-17T11:00:00-04:00",
            "score": 0.7,
        })
        i0.append_alert({
            "symbol": "IWM",
            "direction": "put",
            "scan_timestamp": "2026-06-10T10:00:00-04:00",
            "score": 0.7,
        })
        week = i0.load_week_intraday_alerts(ws)
        assert len(week) == 1
        assert week[0]["symbol"] == "SPY"


class TestExitAlerts:
    def test_open_positions_excludes_closed(self, tmp_trading, monkeypatch):
        monkeypatch.setattr(
            i0, "date",
            type("D", (), {"today": staticmethod(lambda: date(2026, 6, 17))})(),
        )
        entry_ts = "2026-06-17T10:00:00-04:00"
        i0.append_alert({
            "symbol": "SPY",
            "direction": "call",
            "alert_action": "entry",
            "scan_timestamp": entry_ts,
            "underlying_price": 600.0,
            "score": 0.75,
        })
        i0.append_alert({
            "symbol": "SPY",
            "direction": "call",
            "alert_action": "exit",
            "exit_for_entry_ts": entry_ts,
            "scan_timestamp": "2026-06-17T11:00:00-04:00",
            "score": 0.7,
        })
        assert i0.load_open_positions() == []

    def test_call_exit_on_bearish_reversal(self):
        bars = _make_bars(25, "down")
        entry = {
            "symbol": "SPY",
            "direction": "call",
            "underlying_price": float(bars["Close"].iloc[0] + 1.0),
            "scan_timestamp": "2026-06-17T10:00:00-04:00",
        }
        cfg = {"exit_min_score": 0.5, "or_minutes": 15}
        result = i0.score_exit_reversal(entry, bars, {}, cfg)
        assert result is not None
        assert result["alert_action"] == "exit"
        assert result["exit_reason"] == "reversal_exit"
        assert result["score"] >= 0.5
        assert "VWAP" in result["rationale"] or "EMA" in result["rationale"]

    def test_premium_stop_exit_builds(self, monkeypatch):
        monkeypatch.setattr(
            i0, "option_pnl_pct_for_entry", lambda entry, on_date=None: -35.0,
        )
        bars = _make_bars(10, "flat")
        entry = {
            "symbol": "SPY",
            "direction": "call",
            "underlying_price": 600.0,
            "scan_timestamp": "2026-06-17T10:00:00-04:00",
            "recommended_contract": {"tiers": {"atm": {"mid_price": 2.0, "strike": 600, "expiration": "2026-06-17"}}},
        }
        ex = i0.build_premium_stop_exit(entry, bars, {"premium_stop_pct": -30}, -35.0)
        assert ex["exit_reason"] == "premium_stop"
        assert ex["option_pnl_pct"] == -35.0

    def test_eod_exit_when_past_time(self, monkeypatch):
        monkeypatch.setattr(
            i0, "now_et",
            lambda: datetime(2026, 6, 17, 15, 50, tzinfo=ET),
        )
        assert i0.is_past_eod_exit({"eod_exit_enabled": True, "eod_exit_time": "15:45"}) is True
        bars = _make_bars(10, "flat")
        entry = {
            "symbol": "QQQ",
            "direction": "put",
            "underlying_price": 500.0,
            "scan_timestamp": "2026-06-17T10:00:00-04:00",
        }
        ex = i0.build_eod_exit(entry, bars, {"eod_exit_time": "15:45"})
        assert ex["exit_reason"] == "eod_exit"

    def test_run_scan_survives_cache_desync(self, tmp_trading, monkeypatch):
        """Regression: flip_exits_for_new_entries seeds bars_cache[sym] without
        options_cache[sym]; the exit loop must still populate options_cache and
        not KeyError when the same symbol has an open position to monitor."""
        open_pos = {
            "symbol": "QQQ", "direction": "call",
            "scan_timestamp": "2026-06-17T10:00:00-04:00",
            "recommended_contract": {"tiers": {"atm": {
                "strike": 480, "expiration": "2026-06-17", "mid_price": 1.2}}},
        }
        monkeypatch.setattr(i0, "is_trading_day", lambda *a, **k: True)
        monkeypatch.setattr(i0, "is_market_hours", lambda *a, **k: True)
        monkeypatch.setattr(i0, "minutes_since_open", lambda *a, **k: 60.0)
        monkeypatch.setattr(i0, "fetch_intraday_bars", lambda s: _make_bars())
        monkeypatch.setattr(i0, "fetch_0dte_options", lambda s, dte_max=1: {"calls": [], "puts": []})
        monkeypatch.setattr(i0, "score_symbol", lambda symbol, bars, options, cfg: {
            "symbol": symbol, "direction": "call", "score": 0.9,
            "underlying_price": 100.0, "key_signals": [], "skip_reason": None})
        monkeypatch.setattr(i0, "pick_option_contract", lambda **k: {"tiers": {"atm": {
            "strike": 480, "expiration": "2026-06-17", "mid_price": 1.2, "ask": 1.3}}})
        monkeypatch.setattr(i0, "should_fire_alert", lambda *a, **k: True)
        monkeypatch.setattr(i0, "load_open_positions", lambda *a, **k: [open_pos])
        monkeypatch.setattr(i0, "attach_exit_option_mid", lambda entry, exit_alert: exit_alert)

        config = {"intraday_0dte": {
            "or_wait_minutes": 0, "min_score": 0.5, "max_alerts_per_run": 2,
            "flip_exit_on_opposite_entry": True, "exit_alerts_enabled": True,
            "eod_exit_enabled": True, "eod_exit_time": "00:00",   # force EOD path
            "email_alerts_enabled": False}, "budget": {"total_usd": 500}}

        # Pre-fix this raised KeyError: 'QQQ'.
        result = i0.run_scan(config, ["QQQ"], dry_run=True)
        actions = {(a["symbol"], a.get("alert_action")) for a in result}
        assert ("QQQ", "entry") in actions
        assert any(act == "exit" for _, act in actions)   # EOD exit for the open position

    def test_flip_exit_on_opposite_entry(self, tmp_trading, monkeypatch):
        monkeypatch.setattr(
            i0, "date",
            type("D", (), {"today": staticmethod(lambda: date(2026, 6, 17))})(),
        )
        call_ts = "2026-06-17T10:00:00-04:00"
        i0.append_alert({
            "symbol": "QQQ",
            "direction": "call",
            "alert_action": "entry",
            "scan_timestamp": call_ts,
            "underlying_price": 500.0,
            "score": 0.75,
        })
        put_entry = {
            "symbol": "QQQ",
            "direction": "put",
            "alert_action": "entry",
            "scan_timestamp": "2026-06-17T11:00:00-04:00",
            "underlying_price": 498.0,
            "score": 0.8,
        }
        bars = _make_bars(15, "down")
        cache = {"QQQ": bars}
        exits = i0.flip_exits_for_new_entries(
            [put_entry],
            {"flip_exit_on_opposite_entry": True, "allow_hedge_spread": False},
            cache,
        )
        assert len(exits) == 1
        assert exits[0]["exit_reason"] == "flip_opposite_entry"
        assert exits[0]["exit_for_entry_ts"] == call_ts
        assert exits[0]["flip_trigger_direction"] == "put"

    def test_flip_skipped_in_hedge_mode(self, tmp_trading, monkeypatch):
        monkeypatch.setattr(
            i0, "date",
            type("D", (), {"today": staticmethod(lambda: date(2026, 6, 17))})(),
        )
        i0.append_alert({
            "symbol": "QQQ",
            "direction": "call",
            "alert_action": "entry",
            "scan_timestamp": "2026-06-17T10:00:00-04:00",
            "underlying_price": 500.0,
            "score": 0.75,
        })
        put_entry = {
            "symbol": "QQQ",
            "direction": "put",
            "alert_action": "entry",
            "scan_timestamp": "2026-06-17T11:00:00-04:00",
            "underlying_price": 498.0,
            "score": 0.8,
        }
        exits = i0.flip_exits_for_new_entries(
            [put_entry],
            {"flip_exit_on_opposite_entry": True, "allow_hedge_spread": True},
            {"QQQ": _make_bars(10, "flat")},
        )
        assert exits == []

    def test_exit_dedup_separate_from_entry(self, tmp_trading, monkeypatch):
        now = datetime(2026, 6, 17, 11, 0, tzinfo=ET)
        monkeypatch.setattr(i0, "now_et", lambda: now)
        i0.append_alert({
            "symbol": "SPY",
            "direction": "call",
            "alert_action": "exit",
            "scan_timestamp": (now - timedelta(minutes=5)).isoformat(),
            "score": 0.7,
        })
        candidate = {
            "symbol": "SPY",
            "direction": "call",
            "alert_action": "entry",
            "score": 0.75,
        }
        assert i0.should_fire_alert(candidate, dedup_minutes=30) is True

    def test_find_exit_for_entry(self, tmp_trading):
        entry_ts = "2026-06-17T10:00:00-04:00"
        entry = {
            "symbol": "SPY",
            "direction": "call",
            "alert_action": "entry",
            "scan_timestamp": entry_ts,
        }
        exit_a = {
            "symbol": "SPY",
            "direction": "call",
            "alert_action": "exit",
            "exit_for_entry_ts": entry_ts,
            "scan_timestamp": "2026-06-17T11:00:00-04:00",
            "exit_option_mid": 1.5,
        }
        assert i0.find_exit_for_entry(entry, [entry, exit_a]) == exit_a
        assert i0.find_exit_for_entry(entry, [entry]) is None


class TestReflectIntegration:
    def test_fetch_eod_outcome_with_mock(self, monkeypatch):
        class FakeHist:
            def __init__(self):
                self.index = pd.date_range("2026-06-17", periods=1, freq="D")
                self._close = 101.0

            def __getitem__(self, key):
                return self

            @property
            def empty(self):
                return False

            def __len__(self):
                return 1

            @property
            def loc(self):
                return self

            def __getattr__(self, name):
                if name == "iloc":
                    return type("I", (), {"__getitem__": lambda s, i: 101.0})()
                raise AttributeError(name)

        def fake_history(self, **kwargs):
            return pd.DataFrame(
                {"Close": [101.0]},
                index=pd.to_datetime(["2026-06-17"]),
            )

        monkeypatch.setattr(
            "yfinance.Ticker",
            lambda sym: type("T", (), {"history": fake_history})(),
        )
        pct = i0.fetch_eod_outcome("SPY", date(2026, 6, 17), 100.0)
        assert pct == 1.0
