#!/usr/bin/env python3
"""
scanner.py — US Market Scanner (Week 1)

Scans for relative volume spikes on US stocks, fetches TA and options data,
writes ~/trading/data/all_data.json for orchestrate.py.

Data sources:
  - tradingview-screener: volume ranking + RSI/MACD/EMA/recommendation (single API call)
  - yfinance: options chains, earnings proximity, news headlines

Usage:
  python scanner.py
  python scanner.py --context "Fed meeting today, bearish bias"
  python scanner.py --top-n 15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from tradingview_screener import Column, Query

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path.home() / "trading"
DATA_DIR = BASE_DIR / "data"
LOG_DIR  = BASE_DIR / "logs"
CONFIG_PATH = BASE_DIR / "config.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging  (INFO → console, WARNING+ → rotating file)
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    log_file = LOG_DIR / "scanner.log"
    file_handler = TimedRotatingFileHandler(str(log_file), when="D", backupCount=7)
    file_handler.setLevel(logging.WARNING)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[console_handler, file_handler])
    return logging.getLogger("scanner")


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(
            f"config.json not found at {CONFIG_PATH}.\n"
            "  Run: cp ~/trading/config.json.template ~/trading/config.json\n"
            "  Then fill in your API keys."
        )
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Volume scan + TA — tradingview-screener (single API call)
# ---------------------------------------------------------------------------
def get_volume_leaders(config: dict) -> tuple[list[dict], bool]:
    """
    Query TradingView screener for US stocks with elevated relative volume.
    Returns (candidates, had_errors).

    Each candidate dict:
      symbol, name, price, change_pct, volume, avg_volume_10d,
      relative_volume, rsi, macd, macd_signal, ema20, ema50, ema200,
      tv_recommendation
    """
    scan = config.get("scan", {})
    min_rel_vol   = scan.get("min_relative_volume", 2.0)
    min_price     = scan.get("min_price", 10.0)
    min_vol       = scan.get("min_total_volume", 2_000_000)
    top_n         = scan.get("pre_filter_top_n", 10)

    logger.info(f"Scanning US stocks (rel_vol>{min_rel_vol}x, price>${min_price}, vol>{min_vol:,})...")
    try:
        count, df = (
            Query()
            .set_markets("america")
            .select(
                "name",
                "close",
                "volume",
                "average_volume_10d_calc",
                "relative_volume_10d_calc",
                "change",
                "RSI",
                "MACD.macd",
                "MACD.signal",
                "EMA20",
                "EMA50",
                "EMA200",
                "Recommend.All",
            )
            .where(
                Column("volume") > min_vol,
                Column("close") > min_price,
                Column("relative_volume_10d_calc") > min_rel_vol,
            )
            .order_by("relative_volume_10d_calc", ascending=False)
            .limit(top_n)
            .get_scanner_data()
        )
    except Exception as exc:
        logger.error(f"tradingview-screener error: {exc}")
        return [], True

    if df is None or df.empty:
        logger.info("No candidates met volume scan criteria.")
        return [], False

    # ticker column is formatted as 'EXCHANGE:SYMBOL', e.g. 'NASDAQ:AAPL'
    candidates = []
    for _, row in df.iterrows():
        raw_ticker = str(row.get("ticker", ""))
        symbol = raw_ticker.split(":")[-1] if ":" in raw_ticker else raw_ticker

        # TradingView recommendation: float in [-1, 1]; positive = buy bias
        rec = row.get("Recommend.All")
        if pd.isna(rec) or rec is None:
            rec = None
        else:
            rec = round(float(rec), 3)

        def _safe(val, digits=2):
            """Return rounded float or None for NaN/None."""
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return round(float(val), digits)

        candidates.append({
            "symbol":          symbol,
            "tv_ticker":       raw_ticker,
            "name":            str(row.get("name", symbol)),
            "price":           _safe(row.get("close")),
            "change_pct":      _safe(row.get("change")),
            "volume":          int(row.get("volume", 0)),
            "avg_volume_10d":  _safe(row.get("average_volume_10d_calc"), 0),
            "relative_volume": _safe(row.get("relative_volume_10d_calc")),
            "rsi":             _safe(row.get("RSI")),
            "macd":            _safe(row.get("MACD.macd")),
            "macd_signal":     _safe(row.get("MACD.signal")),
            "ema20":           _safe(row.get("EMA20")),
            "ema50":           _safe(row.get("EMA50")),
            "ema200":          _safe(row.get("EMA200")),
            "tv_recommendation": rec,
        })

    logger.info(f"Screener returned {len(candidates)} candidates ({count} total matches in market).")
    return candidates, False


# ---------------------------------------------------------------------------
# Earnings proximity check
# ---------------------------------------------------------------------------
def check_earnings(symbols: list[str], hours: int = 48) -> dict[str, bool]:
    """Return True for each ticker with earnings within `hours` hours."""
    cutoff = datetime.now(timezone.utc) + timedelta(hours=hours)
    result: dict[str, bool] = {}

    for symbol in symbols:
        try:
            cal = yf.Ticker(symbol).calendar
            if cal is None or (hasattr(cal, "empty") and cal.empty):
                result[symbol] = False
                continue

            if isinstance(cal, dict):
                dates = cal.get("Earnings Date", [])
                if not isinstance(dates, list):
                    dates = [dates]
            else:
                dates = cal.get("Earnings Date", pd.Series()).dropna().tolist()

            result[symbol] = any(
                _to_utc(d) <= cutoff for d in dates if d is not None
            )
        except Exception as exc:
            logger.warning(f"Earnings check failed for {symbol}: {exc}")
            result[symbol] = False  # safe default: don't exclude

    return result


def _to_utc(dt_val) -> datetime:
    """Coerce various date/datetime types to UTC-aware datetime."""
    if isinstance(dt_val, datetime):
        return dt_val.astimezone(timezone.utc) if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)
    if isinstance(dt_val, date):
        return datetime(dt_val.year, dt_val.month, dt_val.day, tzinfo=timezone.utc)
    ts = pd.Timestamp(dt_val)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime().astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Options data (yfinance)
# ---------------------------------------------------------------------------
def get_options_data(symbols: list[str]) -> dict[str, dict]:
    """
    Fetch near-term options chains via yfinance.
    Summarises call/put volume, OI, returns raw chain for LLM.
    """
    result: dict[str, dict] = {}
    today = date.today()

    for symbol in symbols:
        empty = {
            "calls": [], "puts": [], "total_volume": 0,
            "total_call_volume": 0, "total_put_volume": 0,
            "call_put_ratio": None,
        }
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options or []

            near = [e for e in expirations
                    if 7 <= (date.fromisoformat(e) - today).days <= 45]
            if not near:
                near = list(expirations[:3])

            all_calls, all_puts = [], []
            for exp in near[:3]:
                try:
                    chain = ticker.option_chain(exp)
                    cols = ["strike", "volume", "openInterest", "bid", "ask", "impliedVolatility"]
                    c = chain.calls[[c for c in cols if c in chain.calls.columns]].copy()
                    p = chain.puts[[c for c in cols if c in chain.puts.columns]].copy()
                    c["expiration"] = exp
                    p["expiration"] = exp
                    all_calls.append(c)
                    all_puts.append(p)
                except Exception as exc:
                    logger.warning(f"{symbol} exp {exp}: {exc}")

            if not all_calls:
                result[symbol] = empty
                continue

            calls = pd.concat(all_calls, ignore_index=True).fillna(0)
            puts  = pd.concat(all_puts,  ignore_index=True).fillna(0)
            call_vol = int(calls.get("volume", pd.Series([0])).sum())
            put_vol  = int(puts.get("volume",  pd.Series([0])).sum())

            result[symbol] = {
                "calls": calls.to_dict(orient="records"),
                "puts":  puts.to_dict(orient="records"),
                "total_volume":      call_vol + put_vol,
                "total_call_volume": call_vol,
                "total_put_volume":  put_vol,
                "call_put_ratio":    round(call_vol / put_vol, 2) if put_vol > 0 else None,
            }
        except Exception as exc:
            logger.warning(f"Options fetch failed for {symbol}: {exc}")
            result[symbol] = empty

    return result


# ---------------------------------------------------------------------------
# 5-day price return (yfinance) — used by reflect.py pattern detection
# ---------------------------------------------------------------------------
def get_5day_returns(symbols: list[str]) -> dict[str, float | None]:
    """
    Fetch the 5-trading-day price return for each symbol.
    Returns pct change from 5 trading days ago to today's last close.
    Returns None for a symbol if data is unavailable.
    """
    result: dict[str, float | None] = {}
    for symbol in symbols:
        try:
            hist = yf.Ticker(symbol).history(period="10d", interval="1d", auto_adjust=True)
            if hist is None or hist.empty or len(hist) < 2:
                result[symbol] = None
                continue
            closes = hist["Close"].dropna().tolist()
            # Use last 6 data points to get ~5 trading days of change
            if len(closes) >= 6:
                result[symbol] = round((closes[-1] - closes[-6]) / closes[-6] * 100, 2)
            elif len(closes) >= 2:
                result[symbol] = round((closes[-1] - closes[0]) / closes[0] * 100, 2)
            else:
                result[symbol] = None
        except Exception as exc:
            logger.warning(f"5-day return fetch failed for {symbol}: {exc}")
            result[symbol] = None
    return result


# ---------------------------------------------------------------------------
# News (yfinance v1.4+ format)
# ---------------------------------------------------------------------------
def get_news(symbols: list[str]) -> dict[str, list]:
    """Fetch up to 5 recent headlines per ticker (yfinance v1.4+ nested format)."""
    result: dict[str, list] = {}
    for symbol in symbols:
        try:
            raw_items = yf.Ticker(symbol).news or []
            items = []
            for n in raw_items[:5]:
                # yfinance >=1.4: news is nested under 'content' key
                content = n.get("content", n)  # fallback to flat dict for older versions
                title     = content.get("title", n.get("title", ""))
                publisher = (content.get("provider", {}).get("displayName")
                             or n.get("publisher", ""))
                pub_date  = content.get("pubDate", n.get("providerPublishTime", 0))
                summary   = content.get("summary", n.get("summary", ""))
                items.append({
                    "title":        title,
                    "publisher":    publisher,
                    "published_at": pub_date,
                    "summary":      summary[:300] if summary else "",
                })
            result[symbol] = items
        except Exception as exc:
            logger.warning(f"News fetch failed for {symbol}: {exc}")
            result[symbol] = []
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="US Market Scanner")
    parser.add_argument("--context", type=str, default=None,
                        help="Free-text context bias passed to LLM (e.g. 'bearish macro today')")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Override pre-filter top N (default: from config.json)")
    args = parser.parse_args()

    logger.info("=== Scanner started ===")
    config = load_config()

    if args.top_n:
        config.setdefault("scan", {})["pre_filter_top_n"] = args.top_n

    # Step 1: Volume scan + TA via tradingview-screener
    candidates, had_errors = get_volume_leaders(config)

    if not candidates:
        logger.info("No candidates today.")
        _write_output([], args.context, had_errors=had_errors)
        print("\nNo candidates today — market conditions didn't trigger scan criteria.")
        return

    symbols = [c["symbol"] for c in candidates]

    # Step 2: Earnings check
    logger.info(f"Checking earnings for: {symbols}")
    earnings = check_earnings(symbols)

    # Step 3: Options chains
    logger.info(f"Fetching options for: {symbols}")
    options = get_options_data(symbols)

    # Step 4: News
    logger.info(f"Fetching news for: {symbols}")
    news = get_news(symbols)

    # Step 5: 5-day returns (for reflect.py pattern detection)
    logger.info(f"Fetching 5-day returns for: {symbols}")
    returns_5d = get_5day_returns(symbols)

    # Step 6: Assemble all_data.json records
    ticker_records = []
    for c in candidates:
        sym = c["symbol"]
        od  = options.get(sym, {})

        # Derive EMA alignment signal from available EMAs
        ema_alignment = _ema_alignment(c.get("ema20"), c.get("ema50"), c.get("ema200"))

        ticker_records.append({
            "symbol":              sym,
            "name":                c.get("name", sym),
            "price":               c.get("price", 0),
            "change_pct":          c.get("change_pct"),
            "change_5d_pct":       returns_5d.get(sym),
            "relative_volume":     c.get("relative_volume", 0),
            "volume":              c.get("volume", 0),
            "avg_volume_10d":      c.get("avg_volume_10d"),
            "earnings_within_48h": earnings.get(sym, False),
            # TA fields from tradingview-screener
            "patterns": {
                "rsi":             c.get("rsi"),
                "macd":            c.get("macd"),
                "macd_signal":     c.get("macd_signal"),
                "ema20":           c.get("ema20"),
                "ema50":           c.get("ema50"),
                "ema200":          c.get("ema200"),
                "ema_alignment":   ema_alignment,
                "tv_recommendation": c.get("tv_recommendation"),
            },
            # Options
            "options_total_volume":  od.get("total_volume", 0),
            "options_call_volume":   od.get("total_call_volume", 0),
            "options_put_volume":    od.get("total_put_volume", 0),
            "call_put_ratio":        od.get("call_put_ratio"),
            "options_chain": {
                "calls": od.get("calls", []),
                "puts":  od.get("puts", []),
            },
            # Placeholders for Week 4 additions
            "ohlcv":    [],
            # News
            "news": news.get(sym, []),
        })

    _write_output(ticker_records, args.context, had_errors)
    _print_summary(ticker_records)


def _ema_alignment(ema20: Optional[float], ema50: Optional[float],
                   ema200: Optional[float]) -> Optional[str]:
    """Classify EMA alignment from available EMA values."""
    if ema20 is None or ema50 is None:
        return None
    if ema200 is not None:
        if ema20 > ema50 > ema200:
            return "bullish"
        if ema20 < ema50 < ema200:
            return "bearish"
        return "mixed"
    # Only 20 and 50 available
    if ema20 > ema50:
        return "bullish"
    if ema20 < ema50:
        return "bearish"
    return "neutral"


def _write_output(ticker_records: list, context: Optional[str], had_errors: bool) -> None:
    output = {
        "scan_timestamp": datetime.now(timezone.utc).isoformat(),
        "data_quality":   "partial" if had_errors else "complete",
        "context":        context,
        "tickers":        ticker_records,
    }
    out_path = DATA_DIR / "all_data.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(
        f"Written: {out_path} "
        f"({len(ticker_records)} candidates, quality={output['data_quality']})"
    )


def _print_summary(records: list) -> None:
    if not records:
        return
    print("\n─── SCAN RESULTS ────────────────────────────────────────────────────")
    print(f"{'TICKER':<8} {'REL_VOL':>8} {'PRICE':>8} {'RSI':>6} {'EMA':>8} {'OPT_VOL':>10} {'C/P':>5}  FLAGS")
    print("─" * 72)
    for r in records:
        pat = r.get("patterns") or {}
        rsi_str = f"{pat['rsi']:.0f}" if pat.get("rsi") else " -- "
        ema_str = (pat.get("ema_alignment") or "n/a")[:4]
        cp = f"{r['call_put_ratio']:.1f}" if r["call_put_ratio"] else "N/A"
        flags = ""
        if r["earnings_within_48h"]:
            flags += " EARNINGS"
        if r["call_put_ratio"] and r["call_put_ratio"] > 2.0:
            flags += " CALL-FLOW"
        elif r["call_put_ratio"] and r["call_put_ratio"] < 0.5:
            flags += " PUT-FLOW"
        rec = pat.get("tv_recommendation")
        if rec is not None:
            if rec > 0.3:
                flags += " TV:BUY"
            elif rec < -0.3:
                flags += " TV:SELL"
        print(
            f"{r['symbol']:<8} {r['relative_volume']:>7.1f}x "
            f"{r['price']:>8.2f} "
            f"{rsi_str:>6} "
            f"{ema_str:>8} "
            f"{r['options_total_volume']:>10,} "
            f"{cp:>5}{flags}"
        )
    out_path = DATA_DIR / "all_data.json"
    print(f"\n→ all_data.json written: {out_path}")
    print("→ Run orchestrate.py to score candidates against creator frameworks.\n")


if __name__ == "__main__":
    main()
