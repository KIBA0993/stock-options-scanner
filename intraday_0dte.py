#!/usr/bin/env python3
"""
intraday_0dte.py — Rule-based 0–1 DTE index options scanner (no LLM).

Watches SPY / QQQ / IWM during market hours, scores intraday setups
(ORB, VWAP, RSI, EMA, relative volume), picks 0–1 DTE contracts, and
appends alerts to intraday_0dte_alerts.jsonl for Friday reflect.py review.

Usage:
  python intraday_0dte.py
  python intraday_0dte.py --dry-run
  python intraday_0dte.py --symbols SPY QQQ
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, time, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from orchestrate import pick_option_contract
from market_calendar import is_market_hours, is_trading_day
from utils import monday_of_week

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path.home() / "trading"
DATA_DIR = BASE_DIR / "data"
ARCHIVE_DIR = DATA_DIR / "archive"
LOG_DIR = BASE_DIR / "logs"
CONFIG_PATH = BASE_DIR / "config.json"
ALERTS_PATH = DATA_DIR / "intraday_0dte_alerts.jsonl"

ET = ZoneInfo("America/New_York")
OR_MINUTES = 15  # opening range window

DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM"]
DEFAULT_MIN_SCORE = 0.70
DEFAULT_MAX_ALERTS_PER_RUN = 2
DEFAULT_DEDUP_MINUTES = 30
DEFAULT_EXIT_MIN_SCORE = 0.65
DEFAULT_EXIT_DEDUP_MINUTES = 15
DEFAULT_EXIT_MIN_HOLD_MINUTES = 5
DEFAULT_PREMIUM_STOP_PCT = -30.0
DEFAULT_EOD_EXIT_TIME = "15:45"
EXIT_REASON_REVERSAL = "reversal_exit"
EXIT_REASON_PREMIUM_STOP = "premium_stop"
EXIT_REASON_EOD = "eod_exit"
EXIT_REASON_FLIP = "flip_opposite_entry"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "intraday_0dte.log"
    fh = TimedRotatingFileHandler(str(log_file), when="D", backupCount=7)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[ch, fh])
    return logging.getLogger("intraday_0dte")


logger = _setup_logging()


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(f"config.json not found at {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


def intraday_cfg(config: dict) -> dict:
    return config.get("intraday_0dte", {})


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------
def now_et() -> datetime:
    return datetime.now(ET)


def minutes_since_open(dt: Optional[datetime] = None) -> float:
    dt = dt or now_et()
    open_dt = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    return max(0.0, (dt - open_dt).total_seconds() / 60)


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------
def fetch_intraday_bars(symbol: str, interval: str = "5m") -> pd.DataFrame:
    """Today's 5m bars (falls back to last 2 sessions if pre-market)."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2d", interval=interval, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET)
    else:
        df.index = df.index.tz_convert(ET)
    today = now_et().date()
    today_bars = df[df.index.date == today]
    return today_bars if not today_bars.empty else df


def compute_vwap(bars: pd.DataFrame) -> float:
    if bars.empty:
        return 0.0
    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    vol = bars["Volume"].replace(0, pd.NA).fillna(1).astype(float)
    return float((tp * vol).sum() / vol.sum())


def compute_rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 1) if pd.notna(val) else None


def opening_range(bars: pd.DataFrame, or_minutes: int = OR_MINUTES) -> tuple[float, float]:
    """High/low of first `or_minutes` after 9:30 ET."""
    if bars.empty:
        return 0.0, 0.0
    open_dt = bars.index[0].replace(hour=9, minute=30, second=0, microsecond=0)
    if bars.index[0].date() != open_dt.date():
        open_dt = bars.index[0].normalize().replace(hour=9, minute=30)
    end_dt = open_dt + timedelta(minutes=or_minutes)
    or_bars = bars[(bars.index >= open_dt) & (bars.index < end_dt)]
    if or_bars.empty:
        or_bars = bars.head(max(1, or_minutes // 5))
    return float(or_bars["High"].max()), float(or_bars["Low"].min())


def fetch_0dte_options(symbol: str, dte_max: int = 1) -> dict:
    """Options chain for 0–1 DTE expirations."""
    empty = {"calls": [], "puts": [], "total_call_volume": 0, "total_put_volume": 0}
    today = date.today()
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options or []
        near = [
            e for e in expirations
            if 0 <= (date.fromisoformat(e) - today).days <= dte_max
        ]
        if not near:
            return empty

        all_calls, all_puts = [], []
        for exp in near[:2]:
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
            return empty

        calls = pd.concat(all_calls, ignore_index=True).fillna(0)
        puts = pd.concat(all_puts, ignore_index=True).fillna(0)
        call_vol = int(calls.get("volume", pd.Series([0])).sum())
        put_vol = int(puts.get("volume", pd.Series([0])).sum())
        return {
            "calls": calls.to_dict(orient="records"),
            "puts": puts.to_dict(orient="records"),
            "total_call_volume": call_vol,
            "total_put_volume": put_vol,
            "call_put_ratio": round(call_vol / put_vol, 2) if put_vol > 0 else None,
        }
    except Exception as exc:
        logger.warning(f"0DTE options fetch failed for {symbol}: {exc}")
        return empty


# ---------------------------------------------------------------------------
# Scoring (rules only)
# ---------------------------------------------------------------------------
def score_symbol(
    symbol: str,
    bars: pd.DataFrame,
    options: dict,
    cfg: dict,
) -> dict:
    """
    Score bullish (call) and bearish (put) setups independently.
    Returns analysis dict with best direction + score.
    """
    if bars.empty or len(bars) < 3:
        return _skip(symbol, "insufficient intraday bars")

    price = float(bars["Close"].iloc[-1])
    vwap = compute_vwap(bars)
    or_high, or_low = opening_range(bars, cfg.get("or_minutes", OR_MINUTES))
    rsi = compute_rsi(bars["Close"])
    closes = bars["Close"]
    ema9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = closes.ewm(span=21, adjust=False).mean().iloc[-1]

    vol_so_far = int(bars["Volume"].sum())
    mins = max(minutes_since_open(), 1)
    vol_per_min = vol_so_far / mins
    # Rough session pace vs 6.5h full day
    expected_pace = vol_per_min * 390
    avg_daily = float(bars["Volume"].sum())  # placeholder; refined below
    try:
        hist = yf.Ticker(symbol).history(period="12d", interval="1d", auto_adjust=True)
        if hist is not None and len(hist) >= 5:
            avg_daily = float(hist["Volume"].tail(10).mean())
    except Exception:
        pass
    rel_vol = round(expected_pace / avg_daily, 2) if avg_daily > 0 else 1.0

    call_score, call_signals = _score_direction(
        direction="call",
        price=price,
        vwap=vwap,
        or_high=or_high,
        or_low=or_low,
        rsi=rsi,
        ema9=ema9,
        ema21=ema21,
        rel_vol=rel_vol,
        options=options,
        cfg=cfg,
    )
    put_score, put_signals = _score_direction(
        direction="put",
        price=price,
        vwap=vwap,
        or_high=or_high,
        or_low=or_low,
        rsi=rsi,
        ema9=ema9,
        ema21=ema21,
        rel_vol=rel_vol,
        options=options,
        cfg=cfg,
    )

    if call_score >= put_score:
        direction, score, signals = "call", call_score, call_signals
    else:
        direction, score, signals = "put", put_score, put_signals

    min_score = float(cfg.get("min_score", DEFAULT_MIN_SCORE))
    if score < min_score:
        return {
            "symbol": symbol,
            "direction": "skip",
            "score": round(score, 3),
            "would_have_direction": direction,
            "skip_reason": f"score {score:.2f} below min {min_score}",
            "key_signals": signals,
            "patterns": {"rsi": rsi},
            "relative_volume": rel_vol,
            "underlying_price": price,
            "vwap": round(vwap, 2),
            "or_high": round(or_high, 2),
            "or_low": round(or_low, 2),
        }

    return {
        "symbol": symbol,
        "direction": direction,
        "score": round(score, 3),
        "would_have_direction": direction,
        "key_signals": signals,
        "rationale": "; ".join(signals[:4]),
        "patterns": {"rsi": rsi},
        "relative_volume": rel_vol,
        "underlying_price": price,
        "vwap": round(vwap, 2),
        "or_high": round(or_high, 2),
        "or_low": round(or_low, 2),
        "suggested_dte": "0-1 days",
        "risk_level": "high",
        "scoring_method": "intraday_rules",
        "supporting_creators": [],
    }


def _score_direction(
    direction: str,
    price: float,
    vwap: float,
    or_high: float,
    or_low: float,
    rsi: Optional[float],
    ema9: float,
    ema21: float,
    rel_vol: float,
    options: dict,
    cfg: dict,
) -> tuple[float, list[str]]:
    score = 0.0
    signals: list[str] = []
    min_rel = float(cfg.get("min_relative_volume", 1.0))
    cp_ratio = options.get("call_put_ratio")

    if direction == "call":
        if price > vwap:
            score += 0.22
            signals.append(f"price ${price:.2f} above VWAP ${vwap:.2f}")
        if or_high > 0 and price > or_high:
            score += 0.22
            signals.append(f"above opening range high ${or_high:.2f}")
        if rsi is not None and 48 <= rsi <= 68:
            score += 0.18
            signals.append(f"RSI {rsi} in bullish zone")
        elif rsi is not None and rsi > 75:
            score -= 0.15
            signals.append(f"RSI {rsi} overbought — fade risk")
        if ema9 > ema21:
            score += 0.15
            signals.append("EMA9 > EMA21 bullish alignment")
        if rel_vol >= min_rel:
            score += 0.13
            signals.append(f"relative volume {rel_vol}x session pace")
        if cp_ratio is not None and cp_ratio > 1.1:
            score += 0.10
            signals.append(f"call/put vol ratio {cp_ratio}")
    else:
        if price < vwap:
            score += 0.22
            signals.append(f"price ${price:.2f} below VWAP ${vwap:.2f}")
        if or_low > 0 and price < or_low:
            score += 0.22
            signals.append(f"below opening range low ${or_low:.2f}")
        if rsi is not None and 32 <= rsi <= 52:
            score += 0.18
            signals.append(f"RSI {rsi} in bearish zone")
        elif rsi is not None and rsi < 25:
            score -= 0.15
            signals.append(f"RSI {rsi} oversold — bounce risk")
        if ema9 < ema21:
            score += 0.15
            signals.append("EMA9 < EMA21 bearish alignment")
        if rel_vol >= min_rel:
            score += 0.13
            signals.append(f"relative volume {rel_vol}x session pace")
        if cp_ratio is not None and cp_ratio < 0.9:
            score += 0.10
            signals.append(f"call/put vol ratio {cp_ratio} (put-heavy)")

    return min(1.0, max(0.0, score)), signals


def _skip(symbol: str, reason: str) -> dict:
    return {
        "symbol": symbol,
        "direction": "skip",
        "score": 0.0,
        "skip_reason": reason,
        "would_have_direction": "neutral",
        "key_signals": [],
    }


# ---------------------------------------------------------------------------
# Alert I/O & dedup
# ---------------------------------------------------------------------------
def load_alerts() -> list[dict]:
    if not ALERTS_PATH.exists():
        return []
    records = []
    with open(ALERTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def append_alert(alert: dict) -> None:
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_PATH, "a") as f:
        f.write(json.dumps(alert, default=str) + "\n")


def should_fire_alert(
    candidate: dict,
    dedup_minutes: int = DEFAULT_DEDUP_MINUTES,
) -> bool:
    """Skip if same symbol+direction+action alerted recently today."""
    if candidate.get("direction") == "skip":
        return False
    sym = candidate["symbol"].upper()
    direction = candidate["direction"].lower()
    action = candidate.get("alert_action", "entry")
    today = date.today().isoformat()
    cutoff = now_et() - timedelta(minutes=dedup_minutes)

    for rec in reversed(load_alerts()):
        if not rec.get("scan_timestamp", "").startswith(today):
            continue
        if (
            rec.get("symbol", "").upper() == sym
            and rec.get("direction", "").lower() == direction
            and rec.get("alert_action", "entry") == action
        ):
            try:
                ts = datetime.fromisoformat(rec["scan_timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=ET)
                else:
                    ts = ts.astimezone(ET)
                if ts > cutoff:
                    logger.info(
                        f"Dedup: {sym} {direction} {action} at "
                        f"{ts.strftime('%H:%M')} — skip"
                    )
                    return False
            except (ValueError, TypeError):
                return False
    return True


def _parse_alert_ts(rec: dict) -> Optional[datetime]:
    raw = rec.get("scan_timestamp", "")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            return ts.replace(tzinfo=ET)
        return ts.astimezone(ET)
    except (ValueError, TypeError):
        return None


def load_open_positions(for_date: Optional[date] = None) -> list[dict]:
    """Today's entry alerts without a matching exit alert."""
    target = for_date or date.today()
    target_str = target.isoformat()
    entries: list[dict] = []
    closed_entry_ts: set[str] = set()

    for rec in load_alerts():
        ts = rec.get("scan_timestamp", "")
        if not ts.startswith(target_str):
            continue
        direction = rec.get("direction", "").lower()
        action = rec.get("alert_action", "entry")
        if action == "exit":
            entry_ts = rec.get("exit_for_entry_ts")
            if entry_ts:
                closed_entry_ts.add(entry_ts)
        elif direction in ("call", "put"):
            entries.append(rec)

    open_positions = [
        e for e in entries
        if e.get("scan_timestamp") not in closed_entry_ts
    ]
    return open_positions


def load_alerts_for_date(target: date) -> list[dict]:
    """All intraday alerts (entries + exits) on a calendar day."""
    target_str = target.isoformat()
    return [
        rec for rec in load_alerts()
        if rec.get("scan_timestamp", "").startswith(target_str)
    ]


def load_entry_alerts_for_date(target: date) -> list[dict]:
    """Entry alerts only for a calendar day."""
    return [
        a for a in load_alerts_for_date(target)
        if a.get("alert_action", "entry") == "entry"
        and a.get("direction", "").lower() in ("call", "put")
    ]


def find_exit_for_entry(entry: dict, alerts: list[dict] | None = None) -> Optional[dict]:
    """Matching exit alert for an entry, if any."""
    entry_ts = entry.get("scan_timestamp", "")
    if not entry_ts:
        return None
    pool = alerts if alerts is not None else load_alerts()
    for rec in pool:
        if (
            rec.get("alert_action") == "exit"
            and rec.get("exit_for_entry_ts") == entry_ts
        ):
            return rec
    return None


def attach_exit_option_mid(entry: dict, exit_alert: dict) -> dict:
    """Add exit_option_mid to exit alert using entry contract + live chain."""
    from option_outcome import option_mid_on_date, pick_contract_tier

    contract = pick_contract_tier(entry)
    if not contract:
        return exit_alert
    try:
        exit_date = date.fromisoformat((exit_alert.get("scan_timestamp") or "")[:10])
    except ValueError:
        exit_date = date.today()
    mid = option_mid_on_date(
        entry.get("symbol", ""),
        entry.get("direction", ""),
        float(contract["strike"]),
        str(contract["expiration"]),
        exit_date,
    )
    if mid is not None:
        exit_alert = dict(exit_alert)
        exit_alert["exit_option_mid"] = mid
        exit_alert["exit_contract_label"] = contract.get("label")
    return exit_alert


def parse_hhmm(time_str: str) -> time:
    hour, minute = time_str.strip().split(":", 1)
    return time(int(hour), int(minute))


def is_past_eod_exit(cfg: dict, dt: Optional[datetime] = None) -> bool:
    if not cfg.get("eod_exit_enabled", True):
        return False
    dt = dt or now_et()
    return dt.time() >= parse_hhmm(str(cfg.get("eod_exit_time", DEFAULT_EOD_EXIT_TIME)))


def option_pnl_pct_for_entry(
    entry: dict,
    on_date: Optional[date] = None,
) -> Optional[float]:
    """Current option P&L % vs entry contract mid."""
    from option_outcome import option_mid_on_date, pick_contract_tier

    contract = pick_contract_tier(entry)
    if not contract:
        return None
    entry_mid = float(contract.get("mid_price") or 0)
    if entry_mid <= 0:
        return None
    mark_date = on_date or date.today()
    current_mid = option_mid_on_date(
        entry.get("symbol", ""),
        entry.get("direction", ""),
        float(contract["strike"]),
        str(contract["expiration"]),
        mark_date,
    )
    if current_mid is None:
        return None
    return round((current_mid - entry_mid) / entry_mid * 100, 2)


def finalize_exit_alert(exit_candidate: dict) -> dict:
    ts = now_et().isoformat()
    return {
        **exit_candidate,
        "scan_timestamp": ts,
        "week_start": monday_of_week(date.today()).isoformat(),
        "pipeline": "intraday_0dte",
    }


def _exit_underlying_fields(entry: dict, bars: pd.DataFrame) -> dict:
    price = float(bars["Close"].iloc[-1]) if not bars.empty else float(
        entry.get("underlying_price") or 0,
    )
    entry_price = float(entry.get("underlying_price") or price)
    direction = entry.get("direction", "").lower()
    und_pct = None
    if entry_price > 0:
        if direction == "call":
            und_pct = round((price - entry_price) / entry_price * 100, 2)
        else:
            und_pct = round((entry_price - price) / entry_price * 100, 2)
    return {
        "symbol": entry["symbol"].upper(),
        "direction": direction,
        "alert_action": "exit",
        "exit_for_entry_ts": entry.get("scan_timestamp"),
        "entry_underlying_price": round(entry_price, 2),
        "underlying_price": round(price, 2),
        "underlying_move_pct": und_pct,
        "recommended_contract": entry.get("recommended_contract"),
        "scoring_method": "intraday_rules",
        "risk_level": "high",
        "suggested_dte": "0-1 days",
        "supporting_creators": [],
    }


def build_premium_stop_exit(
    entry: dict,
    bars: pd.DataFrame,
    cfg: dict,
    option_pnl_pct: float,
) -> dict:
    stop_pct = float(cfg.get("premium_stop_pct", DEFAULT_PREMIUM_STOP_PCT))
    return {
        **_exit_underlying_fields(entry, bars),
        "exit_reason": EXIT_REASON_PREMIUM_STOP,
        "score": 1.0,
        "option_pnl_pct": option_pnl_pct,
        "premium_stop_pct": stop_pct,
        "key_signals": [f"option P&L {option_pnl_pct}% at or below stop {stop_pct}%"],
        "rationale": f"Premium stop {stop_pct}% hit (option P&L {option_pnl_pct}%)",
        "patterns": {},
    }


def build_eod_exit(entry: dict, bars: pd.DataFrame, cfg: dict) -> dict:
    eod_time = str(cfg.get("eod_exit_time", DEFAULT_EOD_EXIT_TIME))
    option_pnl = option_pnl_pct_for_entry(entry)
    signals = [f"Scheduled end-of-day exit ({eod_time} ET)"]
    if option_pnl is not None:
        signals.append(f"option P&L {option_pnl}%")
    return {
        **_exit_underlying_fields(entry, bars),
        "exit_reason": EXIT_REASON_EOD,
        "score": 1.0,
        "option_pnl_pct": option_pnl,
        "key_signals": signals,
        "rationale": f"End-of-day exit ({eod_time} ET)",
        "patterns": {},
    }


def build_flip_opposite_exit(
    entry: dict,
    trigger_entry: dict,
    bars: pd.DataFrame,
) -> dict:
    sym = entry["symbol"].upper()
    closed_dir = entry["direction"].lower()
    new_dir = trigger_entry["direction"].lower()
    return {
        **_exit_underlying_fields(entry, bars),
        "exit_reason": EXIT_REASON_FLIP,
        "score": 1.0,
        "flip_trigger_entry_ts": trigger_entry.get("scan_timestamp"),
        "flip_trigger_direction": new_dir,
        "key_signals": [
            f"New {sym} {new_dir} entry — closing open {closed_dir} (not hedge mode)",
        ],
        "rationale": (
            f"Opposite-direction entry: new {sym} {new_dir} — exit open {closed_dir}"
        ),
        "patterns": {},
    }


def flip_exits_for_new_entries(
    new_entries: list[dict],
    cfg: dict,
    bars_cache: dict[str, pd.DataFrame],
) -> list[dict]:
    """When a new entry fires, exit same-symbol opposite-direction open positions."""
    if not cfg.get("flip_exit_on_opposite_entry", True):
        return []
    if cfg.get("allow_hedge_spread", False):
        return []

    open_positions = load_open_positions()
    scheduled: set[str] = set()
    exits: list[dict] = []

    for trigger in new_entries:
        sym = trigger["symbol"].upper()
        new_dir = trigger["direction"].lower()
        opposite = "put" if new_dir == "call" else "call"
        if sym not in bars_cache:
            bars_cache[sym] = fetch_intraday_bars(sym)
        bars = bars_cache[sym]

        for pos in open_positions:
            pos_ts = pos.get("scan_timestamp", "")
            if pos_ts in scheduled:
                continue
            if pos["symbol"].upper() != sym:
                continue
            if pos.get("direction", "").lower() != opposite:
                continue
            candidate = build_flip_opposite_exit(pos, trigger, bars)
            candidate = attach_exit_option_mid(pos, candidate)
            exits.append(finalize_exit_alert(candidate))
            scheduled.add(pos_ts)
            logger.info(
                f"  EXIT {sym} {opposite} flip_opposite_entry "
                f"(new {new_dir} entry {trigger.get('scan_timestamp', '')[:16]})"
            )
    return exits


def _compute_exit_reversal(
    entry: dict,
    bars: pd.DataFrame,
    options: dict,
    cfg: dict,
) -> Optional[dict]:
    """
    Full exit-reversal scoring for an open position.
    Always returns eval dict when bars are sufficient (used for telemetry + alerts).
    """
    if bars.empty or len(bars) < 3:
        return None

    direction = entry.get("direction", "").lower()
    if direction not in ("call", "put"):
        return None

    symbol = entry["symbol"].upper()
    price = float(bars["Close"].iloc[-1])
    entry_price = float(entry.get("underlying_price") or price)
    vwap = compute_vwap(bars)
    or_high, or_low = opening_range(bars, cfg.get("or_minutes", OR_MINUTES))
    rsi = compute_rsi(bars["Close"])
    prev_rsi = compute_rsi(bars["Close"].iloc[:-1]) if len(bars) > 15 else None
    closes = bars["Close"]
    ema9 = closes.ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = closes.ewm(span=21, adjust=False).mean().iloc[-1]
    prev_ema9 = closes.ewm(span=9, adjust=False).mean().iloc[-2]
    prev_ema21 = closes.ewm(span=21, adjust=False).mean().iloc[-2]

    score = 0.0
    signals: list[str] = []

    if direction == "call":
        if price < vwap:
            score += 0.24
            signals.append(f"lost VWAP — ${price:.2f} below ${vwap:.2f}")
        if or_low > 0 and price < or_low:
            score += 0.22
            signals.append(f"broke below OR low ${or_low:.2f}")
        elif or_high > 0 and price < or_high and entry_price >= or_high * 0.998:
            score += 0.12
            signals.append(f"failed hold above OR high ${or_high:.2f}")
        if ema9 < ema21:
            score += 0.18
            signals.append("EMA9 < EMA21 bearish cross")
            if prev_ema9 >= prev_ema21:
                score += 0.10
                signals.append("fresh EMA bearish cross")
        if rsi is not None:
            if rsi >= 72:
                score += 0.14
                signals.append(f"RSI {rsi} overbought — take profit / reversal risk")
            elif prev_rsi is not None and prev_rsi >= 60 and rsi < 55:
                score += 0.16
                signals.append(f"RSI rolling over ({prev_rsi:.0f} → {rsi:.0f})")
            elif rsi < 48:
                score += 0.12
                signals.append(f"RSI {rsi} lost bullish momentum")
        if entry_price > 0 and price < entry_price * 0.997:
            score += 0.10
            signals.append(
                f"underlying −{((entry_price - price) / entry_price * 100):.2f}% from entry"
            )
        opp_score, opp_signals = _score_direction(
            "put", price, vwap, or_high, or_low, rsi, ema9, ema21, 1.0, options, cfg,
        )
        if opp_score >= 0.55:
            score += 0.12
            signals.append(f"bearish setup score {opp_score:.2f}")
            signals.extend(opp_signals[:1])
    else:  # put
        if price > vwap:
            score += 0.24
            signals.append(f"reclaimed VWAP — ${price:.2f} above ${vwap:.2f}")
        if or_high > 0 and price > or_high:
            score += 0.22
            signals.append(f"broke above OR high ${or_high:.2f}")
        elif or_low > 0 and price > or_low and entry_price <= or_low * 1.002:
            score += 0.12
            signals.append(f"failed hold below OR low ${or_low:.2f}")
        if ema9 > ema21:
            score += 0.18
            signals.append("EMA9 > EMA21 bullish cross")
            if prev_ema9 <= prev_ema21:
                score += 0.10
                signals.append("fresh EMA bullish cross")
        if rsi is not None:
            if rsi <= 28:
                score += 0.14
                signals.append(f"RSI {rsi} oversold — cover / bounce risk")
            elif prev_rsi is not None and prev_rsi <= 40 and rsi > 45:
                score += 0.16
                signals.append(f"RSI bouncing ({prev_rsi:.0f} → {rsi:.0f})")
            elif rsi > 52:
                score += 0.12
                signals.append(f"RSI {rsi} lost bearish momentum")
        if entry_price > 0 and price > entry_price * 1.003:
            score += 0.10
            signals.append(
                f"underlying +{((price - entry_price) / entry_price * 100):.2f}% from entry"
            )
        opp_score, opp_signals = _score_direction(
            "call", price, vwap, or_high, or_low, rsi, ema9, ema21, 1.0, options, cfg,
        )
        if opp_score >= 0.55:
            score += 0.12
            signals.append(f"bullish setup score {opp_score:.2f}")
            signals.extend(opp_signals[:1])

    score = min(1.0, max(0.0, score))
    exit_min_score = float(cfg.get("exit_min_score", DEFAULT_EXIT_MIN_SCORE))

    pnl_pct = None
    if entry_price > 0:
        if direction == "call":
            pnl_pct = round((price - entry_price) / entry_price * 100, 2)
        else:
            pnl_pct = round((entry_price - price) / entry_price * 100, 2)

    vwap_dist = (price - vwap) / vwap if vwap > 0 else 0.0

    return {
        "symbol": symbol,
        "direction": direction,
        "alert_action": "exit",
        "exit_reason": EXIT_REASON_REVERSAL,
        "score": round(score, 3),
        "exit_min_score": exit_min_score,
        "would_fire_exit": score >= exit_min_score,
        "exit_for_entry_ts": entry.get("scan_timestamp"),
        "entry_underlying_price": round(entry_price, 2),
        "underlying_price": round(price, 2),
        "underlying_move_pct": pnl_pct,
        "key_signals": signals[:8],
        "rationale": "; ".join(signals[:4]),
        "patterns": {"rsi": rsi, "prev_rsi": prev_rsi},
        "vwap": round(vwap, 2),
        "vwap_dist_pct": round(vwap_dist * 100, 4),
        "or_high": round(or_high, 2),
        "or_low": round(or_low, 2),
        "ema9": round(float(ema9), 4),
        "ema21": round(float(ema21), 4),
        "prev_ema9": round(float(prev_ema9), 4),
        "prev_ema21": round(float(prev_ema21), 4),
        "recommended_contract": entry.get("recommended_contract"),
        "scoring_method": "intraday_rules",
        "risk_level": "high",
        "suggested_dte": "0-1 days",
        "supporting_creators": [],
    }


def score_exit_reversal(
    entry: dict,
    bars: pd.DataFrame,
    options: dict,
    cfg: dict,
) -> Optional[dict]:
    """Return exit candidate only when reversal score meets threshold."""
    result = _compute_exit_reversal(entry, bars, options, cfg)
    if not result or not result.get("would_fire_exit"):
        return None
    slim = dict(result)
    slim.pop("exit_min_score", None)
    slim.pop("would_fire_exit", None)
    return slim


def evaluate_exit_reversal(
    entry: dict,
    bars: pd.DataFrame,
    options: dict,
    cfg: dict,
) -> Optional[dict]:
    """Full exit eval for telemetry (includes sub-threshold scores)."""
    return _compute_exit_reversal(entry, bars, options, cfg)


def minutes_since_entry(entry: dict) -> float:
    entry_ts = _parse_alert_ts(entry)
    if not entry_ts:
        return 9999.0
    return max(0.0, (now_et() - entry_ts).total_seconds() / 60)


def load_week_intraday_alerts(week_start: date) -> list[dict]:
    """Alerts from intraday_0dte_alerts.jsonl for ISO week Mon–Sun."""
    week_end = week_start + timedelta(days=6)
    out = []
    for rec in load_alerts():
        ts = rec.get("scan_timestamp", "")
        try:
            d = date.fromisoformat(ts[:10])
        except (ValueError, TypeError):
            continue
        if week_start <= d <= week_end and rec.get("direction") in ("call", "put"):
            if rec.get("alert_action", "entry") != "entry":
                continue
            out.append(rec)
    return out


# Re-export option P&L helpers for reflect + tests
from option_outcome import (  # noqa: E402
    evaluate_intraday_alert,
    fetch_intraday_option_outcome as fetch_option_eod_outcome,
    underlying_close_on as _underlying_close,
)


def fetch_eod_outcome(
    symbol: str,
    alert_date: date,
    entry_price: float,
) -> Optional[float]:
    """Underlying % move from entry_price to same-day close."""
    close = _underlying_close(symbol, alert_date)
    if close is None or entry_price <= 0:
        return None
    return round((close - entry_price) / entry_price * 100, 2)


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------
def run_scan(config: dict, symbols: list[str], dry_run: bool = False) -> list[dict]:
    cfg = intraday_cfg(config)
    or_wait = int(cfg.get("or_wait_minutes", OR_MINUTES))

    if not is_trading_day():
        logger.info("NYSE closed today (weekend/holiday) — exiting")
        return []

    if not is_market_hours():
        logger.info("Outside market hours — exiting")
        return []

    if minutes_since_open() < or_wait:
        logger.info(f"Waiting for opening range ({or_wait} min) — exiting")
        return []

    min_score = float(cfg.get("min_score", DEFAULT_MIN_SCORE))
    max_alerts = int(cfg.get("max_alerts_per_run", DEFAULT_MAX_ALERTS_PER_RUN))
    dedup_min = int(cfg.get("dedup_minutes", DEFAULT_DEDUP_MINUTES))
    dte_max = int(cfg.get("dte_max", 1))
    exit_enabled = bool(cfg.get("exit_alerts_enabled", True))
    max_exit_alerts = int(cfg.get("max_exit_alerts_per_run", max_alerts))
    exit_dedup_min = int(cfg.get("exit_dedup_minutes", DEFAULT_EXIT_DEDUP_MINUTES))
    exit_min_hold = int(cfg.get("exit_min_hold_minutes", DEFAULT_EXIT_MIN_HOLD_MINUTES))
    total_budget = config.get("budget", {}).get("total_usd", 500)

    all_scored: list[dict] = []
    fired: list[dict] = []
    exit_fired: list[dict] = []

    for symbol in symbols:
        logger.info(f"Scanning {symbol} …")
        bars = fetch_intraday_bars(symbol)
        options = fetch_0dte_options(symbol, dte_max=dte_max)
        scored = score_symbol(symbol, bars, options, cfg)
        all_scored.append(scored)

        if scored.get("direction") == "skip":
            logger.info(f"  {symbol}: skip ({scored.get('skip_reason', 'low score')})")
            continue

        if not should_fire_alert(scored, dedup_min):
            continue

        try:
            contract = pick_option_contract(
                symbol=symbol,
                direction=scored["direction"],
                current_price=scored["underlying_price"],
                options_chain=options,
                dte_hint="0-1 days",
                budget=total_budget,
                config=config,
            )
            scored["recommended_contract"] = contract
        except Exception as exc:
            logger.warning(f"Contract pick failed for {symbol}: {exc}")
            scored["recommended_contract"] = None

        ts = now_et().isoformat()
        alert = {
            **scored,
            "alert_action": "entry",
            "scan_timestamp": ts,
            "week_start": monday_of_week(date.today()).isoformat(),
            "pipeline": "intraday_0dte",
        }
        fired.append(alert)
        logger.info(
            f"  ENTRY {symbol} {scored['direction']} score={scored['score']:.2f}"
        )

        if len(fired) >= max_alerts:
            break

    # ── Exit alerts ─────────────────────────────────────────────────────────
    exit_evals: list[dict] = []
    scheduled_closes: set[str] = set()
    bars_cache: dict[str, pd.DataFrame] = {}
    options_cache: dict[str, dict] = {}

    premium_stop_enabled = bool(cfg.get("premium_stop_exit_enabled", True))
    premium_stop_pct = float(cfg.get("premium_stop_pct", DEFAULT_PREMIUM_STOP_PCT))
    eod_active = is_past_eod_exit(cfg)

    if exit_enabled and fired:
        for flip_alert in flip_exits_for_new_entries(fired, cfg, bars_cache):
            ets = flip_alert.get("exit_for_entry_ts")
            if ets and ets not in scheduled_closes:
                exit_fired.append(flip_alert)
                scheduled_closes.add(ets)

    if exit_enabled:
        open_positions = load_open_positions()
        if open_positions:
            logger.info(
                f"Monitoring {len(open_positions)} open entry alert(s) for exit …"
            )

        reversal_fired = 0
        for entry in open_positions:
            entry_ts = entry.get("scan_timestamp", "")
            if entry_ts in scheduled_closes:
                continue

            sym = entry["symbol"].upper()
            hold_min = minutes_since_entry(entry)
            hold_blocked = hold_min < exit_min_hold
            if sym not in bars_cache:
                bars_cache[sym] = fetch_intraday_bars(sym)
                options_cache[sym] = fetch_0dte_options(sym, dte_max=dte_max)
            bars = bars_cache[sym]
            options = options_cache[sym]

            exit_candidate: Optional[dict] = None

            # 1) EOD — close all remaining open positions
            if eod_active:
                exit_candidate = build_eod_exit(entry, bars, cfg)
            # 2) Premium stop — hard loss cap on option mark
            elif premium_stop_enabled:
                opt_pnl = option_pnl_pct_for_entry(entry)
                if opt_pnl is not None and opt_pnl <= premium_stop_pct:
                    exit_candidate = build_premium_stop_exit(
                        entry, bars, cfg, opt_pnl,
                    )
            # 3) Trend reversal (respect hold + dedup + per-scan cap)
            else:
                eval_raw = evaluate_exit_reversal(entry, bars, options, cfg)
                if eval_raw:
                    dedup_blocked = False
                    if eval_raw.get("would_fire_exit") and not hold_blocked:
                        dedup_blocked = not should_fire_alert(eval_raw, exit_dedup_min)
                    exit_evals.append({
                        **eval_raw,
                        "hold_minutes": round(hold_min, 1),
                        "hold_blocked": hold_blocked,
                        "exit_dedup_blocked": dedup_blocked,
                        "exit_fired_this_scan": False,
                    })
                if not hold_blocked and reversal_fired < max_exit_alerts:
                    exit_candidate = score_exit_reversal(entry, bars, options, cfg)
                    if exit_candidate and not should_fire_alert(
                        exit_candidate, exit_dedup_min,
                    ):
                        exit_candidate = None

            if not exit_candidate:
                continue

            exit_candidate = attach_exit_option_mid(entry, exit_candidate)
            exit_alert = finalize_exit_alert(exit_candidate)
            exit_fired.append(exit_alert)
            scheduled_closes.add(entry_ts)

            if exit_candidate.get("exit_reason") == EXIT_REASON_REVERSAL:
                reversal_fired += 1
                for ev in exit_evals:
                    if ev.get("exit_for_entry_ts") == entry_ts:
                        ev["exit_fired_this_scan"] = True
                        break

            reason = exit_candidate.get("exit_reason", "exit")
            logger.info(
                f"  EXIT {sym} {exit_candidate['direction']} "
                f"{reason} (entry {entry_ts[:16]})"
            )

    all_fired = fired + exit_fired

    if dry_run:
        logger.info(
            f"Dry run — would fire {len(fired)} entry + {len(exit_fired)} exit alert(s)"
        )
        return all_fired

    for alert in all_fired:
        append_alert(alert)

    # Archive snapshot for the week (reflect can also read jsonl)
    if all_scored:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = ARCHIVE_DIR / f"intraday-{datetime.now().strftime('%Y%m%d-%H%M')}.json"
        archive_path.write_text(json.dumps({
            "scan_timestamp": now_et().isoformat(),
            "pipeline": "intraday_0dte",
            "all_scored": all_scored,
            "alerts": fired,
            "exit_alerts": exit_fired,
        }, indent=2, default=str))
        logger.info(f"Archive → {archive_path}")

    # Real-time email for each new alert (entry + exit) — optional via config
    if all_fired:
        if cfg.get("email_alerts_enabled", True):
            try:
                from notify import notify_intraday_alerts
                n = notify_intraday_alerts(all_fired, config)
                if n:
                    logger.info(f"Sent {n} intraday email alert(s)")
            except Exception as exc:
                logger.error(f"Intraday email notify failed (non-fatal): {exc}")
        else:
            logger.info(
                f"Intraday email disabled — {len(all_fired)} alert(s) logged only"
            )

    try:
        from intraday_telemetry import log_scan_telemetry
        scored_map = {
            s["symbol"].upper(): s
            for s in all_scored
            if s.get("symbol")
        }
        log_scan_telemetry(
            config,
            symbols,
            scan_source="scan_10m",
            scored_by_symbol=scored_map,
            open_positions=load_open_positions(),
            exit_evals=exit_evals,
            run_meta={
                "entries_fired": len(fired),
                "exits_fired": len(exit_fired),
                "symbols_scanned": len(symbols),
            },
        )
    except Exception as exc:
        logger.warning(f"Telemetry logging failed (non-fatal): {exc}")

    return all_fired


def main() -> None:
    parser = argparse.ArgumentParser(description="0–1 DTE rule-based intraday scanner")
    parser.add_argument("--dry-run", action="store_true", help="Score only; do not log alerts")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override symbol list")
    parser.add_argument("--force", action="store_true", help="Run even outside market hours (testing)")
    args = parser.parse_args()

    config = load_config()
    cfg = intraday_cfg(config)
    symbols = args.symbols or cfg.get("symbols", DEFAULT_SYMBOLS)

    if not args.force and not is_trading_day():
        print("NYSE closed today (weekend/holiday). Use --force to test.")
        sys.exit(0)

    if not args.force and not is_market_hours():
        print("Outside market hours (Mon–Fri 9:30–16:00 ET). Use --force to test.")
        sys.exit(0)

    alerts = run_scan(config, [s.upper() for s in symbols], dry_run=args.dry_run)
    if alerts:
        entries = [a for a in alerts if a.get("alert_action", "entry") == "entry"]
        exits = [a for a in alerts if a.get("alert_action") == "exit"]
        if entries:
            print(f"\n{len(entries)} entry alert(s):")
            for a in entries:
                print(f"  {a['symbol']} {a['direction']} score={a['score']:.0%}")
        if exits:
            print(f"\n{len(exits)} exit alert(s):")
            for a in exits:
                print(
                    f"  SELL {a['symbol']} {a['direction']} "
                    f"reversal={a['score']:.0%}"
                )
    else:
        print("No new intraday alerts.")


if __name__ == "__main__":
    main()
