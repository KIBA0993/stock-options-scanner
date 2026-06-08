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

    # Only listed exchanges that have options markets — excludes OTC/pink sheets
    LISTED_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "NYSE ARCA", "NYSE MKT"}

    logger.info(
        f"Scanning US stocks (rel_vol>{min_rel_vol}x, price>${min_price}, "
        f"vol>{min_vol:,}, exchanges={sorted(LISTED_EXCHANGES)})..."
    )
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
            .limit(top_n * 3)   # fetch extra to absorb OTC drop-offs
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
    otc_dropped = 0
    for _, row in df.iterrows():
        raw_ticker = str(row.get("ticker", ""))
        exchange   = raw_ticker.split(":")[0].upper() if ":" in raw_ticker else ""
        symbol     = raw_ticker.split(":")[-1] if ":" in raw_ticker else raw_ticker

        # Drop OTC / pink-sheet / foreign listings — no listed options
        if exchange and exchange not in LISTED_EXCHANGES:
            otc_dropped += 1
            logger.debug(f"Dropped OTC/unlisted: {raw_ticker}")
            continue

        if len(candidates) >= top_n:
            break   # have enough after filtering

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

    if otc_dropped:
        logger.info(f"Dropped {otc_dropped} OTC/unlisted ticker(s) — no listed options.")
    logger.info(f"Screener returned {len(candidates)} listed candidates ({count} total matches in market).")
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
def get_specific_tickers(symbols: list[str]) -> tuple[list[dict], bool]:
    """
    Fetch TA data for a specific list of symbols via tradingview-screener,
    bypassing the volume filter. Used with --symbols flag.
    """
    from tradingview_screener import Query, Column
    try:
        # NOTE: Column("ticker").isin() with OR breaks the query — use name-based filter only.
        # The screener's `name` column holds the bare ticker symbol (e.g. "MU", "MRVL").
        _count, df = (
            Query()
            .set_markets("america")
            .select(
                "name", "close", "volume", "average_volume_10d_calc",
                "relative_volume_10d_calc", "change",
                "RSI", "MACD.macd", "MACD.signal",
                "EMA20", "EMA50", "EMA200", "Recommend.All",
            )
            .where(Column("name").isin(symbols))
            .limit(len(symbols) * 3)
            .get_scanner_data()
        )
    except Exception:
        # Fallback: fetch each symbol individually via yfinance TA approximation
        return _get_tickers_via_yfinance(symbols), False

    if df is None or df.empty:
        return _get_tickers_via_yfinance(symbols), False

    candidates = []
    found = set()
    for _, row in df.iterrows():
        raw_ticker = str(row.get("ticker", ""))
        sym = raw_ticker.split(":")[-1] if ":" in raw_ticker else raw_ticker
        if sym.upper() not in [s.upper() for s in symbols]:
            continue
        if sym.upper() in found:
            continue
        found.add(sym.upper())

        rec = row.get("Recommend.All")
        if pd.isna(rec) or rec is None:
            rec = None
        else:
            rec = round(float(rec), 3)

        def _safe(val, digits=2):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return round(float(val), digits)

        candidates.append({
            "symbol": sym.upper(), "tv_ticker": raw_ticker,
            "name": str(row.get("name", sym)),
            "price": _safe(row.get("close")),
            "change_pct": _safe(row.get("change")),
            "volume": int(row.get("volume", 0)),
            "avg_volume_10d": _safe(row.get("average_volume_10d_calc"), 0),
            "relative_volume": _safe(row.get("relative_volume_10d_calc")),
            "rsi": _safe(row.get("RSI")),
            "macd": _safe(row.get("MACD.macd")),
            "macd_signal": _safe(row.get("MACD.signal")),
            "ema20": _safe(row.get("EMA20")),
            "ema50": _safe(row.get("EMA50")),
            "ema200": _safe(row.get("EMA200")),
            "tv_recommendation": rec,
        })

    # For any symbol not found in screener, fall back to yfinance entirely
    missing = [s for s in symbols if s.upper() not in found]
    if missing:
        candidates += _get_tickers_via_yfinance(missing)

    # For symbols found but with missing TA (screener returned NaN for RSI/EMA/MACD),
    # patch those fields from yfinance — screener sometimes fails TA for certain tickers
    ta_missing = [c["symbol"] for c in candidates if c.get("rsi") is None]
    if ta_missing:
        logger.info(f"Patching TA via yfinance for {ta_missing} (screener returned NaN)")
        yf_data = {d["symbol"]: d for d in _get_tickers_via_yfinance(ta_missing)}
        for c in candidates:
            if c["symbol"] in yf_data:
                yf = yf_data[c["symbol"]]
                for field in ("rsi", "ema20", "ema50", "ema200", "macd", "macd_signal"):
                    if c.get(field) is None and yf.get(field) is not None:
                        c[field] = yf[field]
                # Also patch price/change_pct if screener returned None
                if c.get("price") is None and yf.get("price") is not None:
                    c["price"] = yf["price"]
                if c.get("change_pct") is None and yf.get("change_pct") is not None:
                    c["change_pct"] = yf["change_pct"]

    return candidates, False


def _fetch_ohlcv_30d(symbols: list[str]) -> dict[str, list]:
    """Fetch last 30 trading days of OHLCV from yfinance for a list of symbols."""
    result: dict[str, list] = {}
    try:
        import yfinance as yf
        data = yf.download(symbols, period="45d", interval="1d",
                           auto_adjust=True, progress=False, group_by="ticker")
        # Handle single vs multi-ticker download shape
        for sym in symbols:
            try:
                df = data[sym] if len(symbols) > 1 else data
                df = df.dropna(how="all").tail(30)
                bars = []
                for idx in df.itertuples():
                    ts = str(idx.Index.date()) if hasattr(idx.Index, "date") else str(idx.Index)
                    bars.append({
                        "date":   ts,
                        "open":   round(float(idx.Open),  4),
                        "high":   round(float(idx.High),  4),
                        "low":    round(float(idx.Low),   4),
                        "close":  round(float(idx.Close), 4),
                        "volume": int(idx.Volume),
                    })
                result[sym.upper()] = bars
            except Exception as exc:
                logger.debug(f"OHLCV parse failed for {sym}: {exc}")
    except Exception as exc:
        logger.warning(f"OHLCV batch download failed: {exc}")
    return result


def _calc_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Calculate RSI from a list of closing prices."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_g  = sum(gains)  / period
    avg_l  = sum(losses) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return round(100 - 100 / (1 + rs), 2)


def _calc_ema(closes: list[float], period: int) -> Optional[float]:
    """Calculate EMA from a list of closing prices."""
    if len(closes) < period:
        return None
    k   = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return round(ema, 4)


def _calc_macd(closes: list[float]) -> tuple[Optional[float], Optional[float]]:
    """Return (MACD line, signal line) using 12/26/9 settings."""
    ema12 = _calc_ema(closes, 12)
    ema26 = _calc_ema(closes, 26)
    if ema12 is None or ema26 is None:
        return None, None
    # Build a MACD series for signal calculation
    macd_series = []
    k12, k26 = 2/13, 2/27
    e12 = sum(closes[:12]) / 12
    e26 = sum(closes[:26]) / 26
    for price in closes[26:]:
        e12 = price * k12 + e12 * (1 - k12)
        e26 = price * k26 + e26 * (1 - k26)
        macd_series.append(e12 - e26)
    if len(macd_series) < 9:
        return round(ema12 - ema26, 4), None
    signal = sum(macd_series[-9:]) / 9
    k9 = 2/10
    for m in macd_series[-9:]:
        signal = m * k9 + signal * (1 - k9)
    return round(macd_series[-1], 4), round(signal, 4)


def _get_tickers_via_yfinance(symbols: list[str]) -> list[dict]:
    """Fallback: fetch TA fields via yfinance with manual RSI/EMA/MACD calculation."""
    results = []
    for sym in symbols:
        try:
            t    = yf.Ticker(sym)
            hist = t.history(period="300d", interval="1d", auto_adjust=True)
            if hist is None or hist.empty:
                continue
            closes = hist["Close"].dropna().tolist()
            vols   = hist["Volume"].dropna().tolist()

            price     = round(closes[-1], 2)  if closes            else None
            chg_1d    = round((closes[-1]-closes[-2])/closes[-2]*100, 2) if len(closes)>=2 else None
            avg_vol   = round(sum(vols[-10:])/len(vols[-10:]), 0)        if len(vols)>=10  else None
            today_vol = int(vols[-1])                                    if vols            else 0
            rel_vol   = round(today_vol / avg_vol, 2) if avg_vol and avg_vol > 0 else None

            rsi    = _calc_rsi(closes)
            ema20  = _calc_ema(closes, 20)
            ema50  = _calc_ema(closes, 50)
            ema200 = _calc_ema(closes, 200)
            macd, macd_sig = _calc_macd(closes)

            # Last 30 OHLCV bars for backtest.py and reflect.py
            ohlcv_30d = []
            for idx in hist.tail(30).itertuples():
                ts = str(idx.Index.date()) if hasattr(idx.Index, "date") else str(idx.Index)
                ohlcv_30d.append({
                    "date":   ts,
                    "open":   round(float(idx.Open), 4),
                    "high":   round(float(idx.High), 4),
                    "low":    round(float(idx.Low),  4),
                    "close":  round(float(idx.Close), 4),
                    "volume": int(idx.Volume),
                })

            results.append({
                "symbol": sym.upper(), "tv_ticker": f"yf:{sym.upper()}",
                "name": sym.upper(), "price": price, "change_pct": chg_1d,
                "volume": today_vol, "avg_volume_10d": avg_vol, "relative_volume": rel_vol,
                "rsi": rsi, "macd": macd, "macd_signal": macd_sig,
                "ema20": ema20, "ema50": ema50, "ema200": ema200,
                "tv_recommendation": None,
                "ohlcv_30d": ohlcv_30d,
            })
        except Exception as exc:
            logger.warning(f"yfinance fallback failed for {sym}: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="US Market Scanner")
    parser.add_argument("--context", type=str, default=None,
                        help="Free-text context bias passed to LLM (e.g. 'bearish macro today')")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Override pre-filter top N (default: from config.json)")
    parser.add_argument("--symbols", nargs="+", metavar="TICKER",
                        help="Force-analyze specific tickers (bypasses volume filter)")
    args = parser.parse_args()

    logger.info("=== Scanner started ===")
    config = load_config()

    if args.top_n:
        config.setdefault("scan", {})["pre_filter_top_n"] = args.top_n

    # Step 1: Volume scan OR specific symbols
    if args.symbols:
        syms_upper = [s.upper() for s in args.symbols]
        logger.info(f"Specific-symbol mode: {syms_upper}")
        candidates, had_errors = get_specific_tickers(syms_upper)
    else:
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

    # Step 5b: Last-30-day OHLCV (for backtest.py; skip if already in candidate from yfinance path)
    logger.info(f"Fetching 30-day OHLCV for: {symbols}")
    ohlcv_map = _fetch_ohlcv_30d(symbols)

    # Step 6: Drop tickers with no listed options (skip in --symbols mode — user asked explicitly)
    if not args.symbols:
        no_options = [s for s in symbols if not options.get(s, {}).get("calls") and
                      not options.get(s, {}).get("puts")]
        if no_options:
            logger.info(f"Dropped {len(no_options)} ticker(s) with no listed options: {no_options}")
            candidates = [c for c in candidates if c["symbol"] not in no_options]
            symbols    = [c["symbol"] for c in candidates]

        if not candidates:
            logger.info("No optionable candidates after filtering.")
            _write_output([], args.context, had_errors=had_errors)
            print("\nNo optionable candidates today — all high-volume stocks were OTC or lack options.")
            return

    # Step 7: Assemble all_data.json records
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
            # Last 30 days of OHLCV for backtest / reflect.py pattern validation
            "ohlcv":    c.get("ohlcv_30d") or ohlcv_map.get(sym, []),
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


def _market_session_note() -> str:
    """Return a note about expected data characteristics based on time of day (ET)."""
    import pytz
    try:
        et_now = datetime.now(pytz.timezone("America/New_York"))
    except Exception:
        return ""
    h, m = et_now.hour, et_now.minute
    minutes_since_open = (h - 9) * 60 + m - 30  # minutes since 9:30 AM ET
    if h < 9 or (h == 9 and m < 30):
        return "Pre-market scan: relative_volume will be near-zero — do NOT penalize low relative_volume."
    if 0 <= minutes_since_open < 30:
        return (
            f"Early session ({minutes_since_open}min since open): relative_volume is characteristically "
            "low in the first 30 minutes. Do NOT use low relative_volume as a bearish signal."
        )
    if minutes_since_open < 90:
        return f"Mid-morning session ({minutes_since_open}min since open): volume still building."
    if h >= 15 and m >= 30:
        return "Near close: volume surge normal. Confirm direction aligns with day's trend."
    return ""


def _write_output(ticker_records: list, context: Optional[str], had_errors: bool) -> None:
    output = {
        "scan_timestamp":    datetime.now(timezone.utc).isoformat(),
        "data_quality":      "partial" if had_errors else "complete",
        "context":           context,
        "market_session_note": _market_session_note(),
        "tickers":           ticker_records,
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
