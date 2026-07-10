"""Tests for market_data.py — provider fallback for price data.

Network (yfinance + Tiingo HTTP) is monkeypatched so the suite is offline and
deterministic. The key behaviors: yfinance-only when no fallback key, fallback
only on yfinance miss, graceful degradation, and caching.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

TRADING_DIR = Path.home() / "trading"
if str(TRADING_DIR) not in sys.path:
    sys.path.insert(0, str(TRADING_DIR))

import market_data as md


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    md.clear_cache()
    yield
    md.clear_cache()


# ─── fallback provider resolution ───────────────────────────────────────────────
class TestFallbackResolution:
    def test_none_when_unconfigured(self):
        assert md._fallback_provider({"data": {}}) is None

    def test_none_when_provider_none(self):
        assert md._fallback_provider({"data": {"fallback_provider": "none"}}) is None

    def test_none_when_tiingo_but_no_key(self):
        # named but keyless → treated as unconfigured (safe default)
        assert md._fallback_provider({"data": {"fallback_provider": "tiingo"}}) is None

    def test_tiingo_when_key_present(self):
        cfg = {"data": {"fallback_provider": "tiingo", "tiingo_api_key": "abc"}}
        assert md._fallback_provider(cfg) == "tiingo"


# ─── get_daily_close ────────────────────────────────────────────────────────────
class TestGetDailyClose:
    def test_uses_yfinance_when_available(self, monkeypatch):
        monkeypatch.setattr(md, "_yf_close", lambda s, d: 101.5)
        called = {"tiingo": 0}
        monkeypatch.setattr(md, "_tiingo_close", lambda *a: called.__setitem__("tiingo", 1))
        cfg = {"data": {"fallback_provider": "tiingo", "tiingo_api_key": "k"}}
        assert md.get_daily_close("NVDA", date(2026, 7, 2), cfg) == 101.5
        assert called["tiingo"] == 0   # fallback never consulted when yf works

    def test_falls_back_when_yfinance_empty(self, monkeypatch):
        monkeypatch.setattr(md, "_yf_close", lambda s, d: None)
        monkeypatch.setattr(md, "_tiingo_close", lambda s, d, k: 99.0)
        cfg = {"data": {"fallback_provider": "tiingo", "tiingo_api_key": "k"}}
        assert md.get_daily_close("NVDA", date(2026, 7, 2), cfg) == 99.0

    def test_returns_none_when_both_fail(self, monkeypatch):
        monkeypatch.setattr(md, "_yf_close", lambda s, d: None)
        monkeypatch.setattr(md, "_tiingo_close", lambda s, d, k: None)
        cfg = {"data": {"fallback_provider": "tiingo", "tiingo_api_key": "k"}}
        assert md.get_daily_close("NVDA", date(2026, 7, 2), cfg) is None

    def test_no_fallback_key_is_yfinance_only(self, monkeypatch):
        monkeypatch.setattr(md, "_yf_close", lambda s, d: None)
        # If fallback were wrongly invoked this would raise (no key in cfg)
        monkeypatch.setattr(md, "_tiingo_close",
                            lambda *a: (_ for _ in ()).throw(AssertionError("should not call")))
        assert md.get_daily_close("NVDA", date(2026, 7, 2), {"data": {}}) is None

    def test_caches_result(self, monkeypatch):
        calls = {"n": 0}
        def yf(s, d):
            calls["n"] += 1
            return 100.0
        monkeypatch.setattr(md, "_yf_close", yf)
        d = date(2026, 7, 2)
        md.get_daily_close("NVDA", d, {"data": {}})
        md.get_daily_close("NVDA", d, {"data": {}})
        assert calls["n"] == 1   # second call served from cache


# ─── get_last_price ─────────────────────────────────────────────────────────────
class TestGetLastPrice:
    def test_prefers_yfinance(self, monkeypatch):
        monkeypatch.setattr(md, "_yf_last", lambda s: 250.0)
        monkeypatch.setattr(md, "_tiingo_last",
                            lambda *a: (_ for _ in ()).throw(AssertionError("should not call")))
        cfg = {"data": {"fallback_provider": "tiingo", "tiingo_api_key": "k"}}
        assert md.get_last_price("MU", cfg) == 250.0

    def test_falls_back(self, monkeypatch):
        monkeypatch.setattr(md, "_yf_last", lambda s: None)
        monkeypatch.setattr(md, "_tiingo_last", lambda s, k: 248.0)
        cfg = {"data": {"fallback_provider": "tiingo", "tiingo_api_key": "k"}}
        assert md.get_last_price("MU", cfg) == 248.0


# ─── Tiingo parsing ─────────────────────────────────────────────────────────────
class TestTiingoParsing:
    def test_close_prefers_exact_date_and_adjclose(self, monkeypatch):
        rows = [
            {"date": "2026-07-02T00:00:00.000Z", "close": 100.0, "adjClose": 100.2},
            {"date": "2026-07-06T00:00:00.000Z", "close": 105.0, "adjClose": 105.1},
        ]
        monkeypatch.setattr(md, "_tiingo_get", lambda *a, **k: rows)
        assert md._tiingo_close("NVDA", date(2026, 7, 2), "k") == 100.2

    def test_close_uses_first_when_date_absent(self, monkeypatch):
        rows = [{"date": "2026-07-06T00:00:00.000Z", "close": 105.0, "adjClose": 105.1}]
        monkeypatch.setattr(md, "_tiingo_get", lambda *a, **k: rows)
        # on_date 07-02 has no exact row → first session on/after
        assert md._tiingo_close("NVDA", date(2026, 7, 2), "k") == 105.1

    def test_close_none_on_empty(self, monkeypatch):
        monkeypatch.setattr(md, "_tiingo_get", lambda *a, **k: None)
        assert md._tiingo_close("NVDA", date(2026, 7, 2), "k") is None


# ─── integration: option_outcome routes through market_data ─────────────────────
class TestOptionOutcomeIntegration:
    def test_underlying_close_on_uses_market_data(self, monkeypatch):
        import option_outcome as oo
        monkeypatch.setattr(md, "_yf_close", lambda s, d: 123.45)
        assert oo.underlying_close_on("NVDA", date(2026, 7, 2)) == 123.45
