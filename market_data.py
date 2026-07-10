#!/usr/bin/env python3
"""
market_data.py — Unified market-data access with provider fallback.

Why this exists
---------------
yfinance is the system's only price source and it's flaky — it silently returns
empty frames on rate-limits, transient Yahoo outages, or odd tickers. Every
downstream (validate.py, option_outcome.py, backtest.py, reflect.py) then loses
a data point. This module keeps yfinance as the primary source but falls back
to a second provider (Tiingo by default — free EOD tier) when yfinance returns
nothing.

Safe by default
---------------
With no fallback key configured, every function behaves EXACTLY like a direct
yfinance call — same data, same failure modes. The fallback only activates when
you add a key to config.json's "data" block, so this can ship before you have a
key and start helping the moment you do.

Config (config.json)
--------------------
  "data": {
    "fallback_provider": "tiingo",     // "tiingo" | "none"
    "tiingo_api_key": ""               // add directly to config.json, never chat
  }

Public API
----------
  get_daily_close(symbol, on_date)  -> float | None
  get_last_price(symbol)            -> float | None

Both accept an optional `config` dict; if omitted it's loaded from config.json.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf

BASE_DIR    = Path.home() / "trading"
CONFIG_PATH = BASE_DIR / "config.json"

logger = logging.getLogger("market_data")

# Simple process-lifetime cache: (symbol, iso_date) -> close. Avoids re-hitting
# providers for the same close within a single run (reflect/validate loop over
# many symbols × horizons).
_close_cache: dict[tuple[str, str], Optional[float]] = {}

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"


# ─── Config ─────────────────────────────────────────────────────────────────────
_cfg_cache: Optional[dict] = None


def _load_config() -> dict:
    global _cfg_cache
    if _cfg_cache is not None:
        return _cfg_cache
    try:
        _cfg_cache = json.loads(CONFIG_PATH.read_text())
    except Exception:
        _cfg_cache = {}
    return _cfg_cache


def _data_cfg(config: Optional[dict]) -> dict:
    return (config or _load_config()).get("data", {})


def _fallback_provider(config: Optional[dict]) -> Optional[str]:
    """Return the active fallback provider name, or None if unconfigured."""
    dc = _data_cfg(config)
    provider = (dc.get("fallback_provider") or "none").lower()
    if provider == "none":
        return None
    if provider == "tiingo" and dc.get("tiingo_api_key"):
        return "tiingo"
    # Provider named but no key → treat as unconfigured (yfinance-only).
    return None


# ─── yfinance primary ───────────────────────────────────────────────────────────
def _yf_close(symbol: str, on_date: date) -> Optional[float]:
    try:
        end = on_date + timedelta(days=5)
        hist = yf.Ticker(symbol).history(
            start=on_date.isoformat(), end=end.isoformat(),
            interval="1d", auto_adjust=True,
        )
        if hist is None or hist.empty:
            return None
        day_rows = hist[hist.index.date == on_date]
        if day_rows.empty:
            day_rows = hist.head(1)   # first session on/after on_date
        return float(day_rows["Close"].iloc[-1])
    except Exception as exc:
        logger.debug(f"yfinance close failed {symbol} {on_date}: {exc}")
        return None


def _yf_last(symbol: str) -> Optional[float]:
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t, "fast_info", None)
        if fi:
            px = fi.get("last_price") if isinstance(fi, dict) else getattr(fi, "last_price", None)
            if px:
                return float(px)
        hist = t.history(period="2d", interval="1d", auto_adjust=True)
        if hist is not None and not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception as exc:
        logger.debug(f"yfinance last price failed {symbol}: {exc}")
    return None


# ─── Tiingo fallback ────────────────────────────────────────────────────────────
def _tiingo_get(url: str, params: dict, key: str) -> Optional[list]:
    try:
        import requests
        params = {**params, "token": key, "format": "json"}
        resp = requests.get(url, params=params, timeout=15,
                            headers={"Content-Type": "application/json"})
        if resp.status_code != 200:
            logger.debug(f"tiingo {resp.status_code} for {url}")
            return None
        data = resp.json()
        return data if isinstance(data, list) else None
    except Exception as exc:
        logger.debug(f"tiingo request failed {url}: {exc}")
        return None


def _tiingo_close(symbol: str, on_date: date, key: str) -> Optional[float]:
    end = on_date + timedelta(days=5)
    rows = _tiingo_get(
        f"{TIINGO_BASE}/{symbol}/prices",
        {"startDate": on_date.isoformat(), "endDate": end.isoformat()},
        key,
    )
    if not rows:
        return None
    # Prefer the exact date; else the first session on/after it. Tiingo dates are
    # ISO strings like "2026-07-02T00:00:00.000Z"; adjClose mirrors yf auto_adjust.
    exact = [r for r in rows if str(r.get("date", "")).startswith(on_date.isoformat())]
    row = exact[0] if exact else rows[0]
    px = row.get("adjClose", row.get("close"))
    return float(px) if px is not None else None


def _tiingo_last(symbol: str, key: str) -> Optional[float]:
    rows = _tiingo_get(f"{TIINGO_BASE}/{symbol}/prices", {}, key)
    if not rows:
        return None
    row = rows[-1]
    px = row.get("adjClose", row.get("close"))
    return float(px) if px is not None else None


# ─── Public API ─────────────────────────────────────────────────────────────────
def get_daily_close(symbol: str, on_date: date, config: Optional[dict] = None) -> Optional[float]:
    """Daily close for `symbol` on `on_date` (or the first session after).

    yfinance first; on empty/failure, the configured fallback provider. Cached
    per (symbol, date) for the process lifetime.
    """
    ckey = (symbol.upper(), on_date.isoformat())
    if ckey in _close_cache:
        return _close_cache[ckey]

    px = _yf_close(symbol, on_date)
    if px is None:
        provider = _fallback_provider(config)
        if provider == "tiingo":
            px = _tiingo_close(symbol, on_date, _data_cfg(config)["tiingo_api_key"])
            if px is not None:
                logger.info(f"fallback(tiingo) supplied close for {symbol} {on_date}")

    _close_cache[ckey] = px
    return px


def get_last_price(symbol: str, config: Optional[dict] = None) -> Optional[float]:
    """Latest price for `symbol`. yfinance first, then configured fallback."""
    px = _yf_last(symbol)
    if px is None:
        provider = _fallback_provider(config)
        if provider == "tiingo":
            px = _tiingo_last(symbol, _data_cfg(config)["tiingo_api_key"])
            if px is not None:
                logger.info(f"fallback(tiingo) supplied last price for {symbol}")
    return px


def clear_cache() -> None:
    """Reset the in-process close cache and config cache (mainly for tests)."""
    global _cfg_cache
    _close_cache.clear()
    _cfg_cache = None
