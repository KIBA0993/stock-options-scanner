#!/usr/bin/env python3
"""
intraday_telemetry.py — Structured snapshots for 0DTE framework backtesting.

Data gaps this module fills (not in intraday_0dte_alerts.jsonl today):
  F1 Anti-chase     — vwap_dist_pct, anti_chase flags on EVERY scan (not only fired alerts)
  F2 Freshness      — prev_close, prev_rsi, fresh_or/vwap/rsi flags per scan
  F3 Premium stop   — position_mark records: option mid + P&L each scan while open
  F4 One/symbol     — would_fire_entry + dedup_blocked on all qualifying scans
  F5 5-min rescan   — scan snapshots every 5 min (via separate cron), not just 10-min alerts

Output: data/intraday_0dte_telemetry.jsonl (append-only, one JSON object per line)

Record types (schema_version 2):
  scan_snapshot  — per-symbol market + score + options context every 5/10 min
  position_mark  — open position option P&L path each scan
  exit_eval      — exit-reversal score for every open position (10m scan only)
  scan_run       — run-level summary (counts, fired alerts)
"""

from __future__ import annotations

TELEMETRY_SCHEMA_VERSION = 2

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from intraday_0dte import (
    DEFAULT_DEDUP_MINUTES,
    DEFAULT_MIN_SCORE,
    ET,
    OR_MINUTES,
    compute_rsi,
    compute_vwap,
    fetch_0dte_options,
    fetch_intraday_bars,
    intraday_cfg,
    load_config,
    load_open_positions,
    minutes_since_open,
    opening_range,
    should_fire_alert,
    _score_direction,
)
from market_calendar import is_market_hours, is_trading_day
from option_outcome import option_mid_on_date, pick_contract_tier

logger = logging.getLogger("intraday_telemetry")

BASE_DIR = Path.home() / "trading"
DEFAULT_TELEMETRY_PATH = BASE_DIR / "data" / "intraday_0dte_telemetry.jsonl"


def telemetry_cfg(config: dict) -> dict:
    defaults = {
        "enabled": True,
        "path": "data/intraday_0dte_telemetry.jsonl",
        "snapshot_on_scan": True,
        "position_marks_on_scan": True,
        "five_min_snapshots": True,
        "rich_logging": True,
        "log_exit_evals": True,
        "log_scan_run": True,
    }
    merged = {**defaults, **config.get("intraday_telemetry", {})}
    fb = config.get("framework_backtest", {})
    merged.setdefault("anti_chase_rsi_call", fb.get("anti_chase_rsi_call", 62.0))
    merged.setdefault("anti_chase_rsi_put", fb.get("anti_chase_rsi_put", 38.0))
    merged.setdefault("anti_chase_vwap_pct", fb.get("anti_chase_vwap_pct", 0.0015))
    merged.setdefault("premium_stop_pct", fb.get("premium_stop_pct", -30.0))
    return merged


def telemetry_path(config: dict) -> Path:
    rel = telemetry_cfg(config).get("path", "data/intraday_0dte_telemetry.jsonl")
    p = Path(rel)
    return p if p.is_absolute() else BASE_DIR / p


def append_telemetry(record: dict, config: dict) -> None:
    if not telemetry_cfg(config).get("enabled", True):
        return
    path = telemetry_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _safe_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def _bar_ohlcv(bars: pd.DataFrame, idx: int) -> dict[str, Optional[float]]:
    if bars.empty or abs(idx) > len(bars):
        return {}
    row = bars.iloc[idx]
    return {
        "open": _safe_float(row.get("Open")),
        "high": _safe_float(row.get("High")),
        "low": _safe_float(row.get("Low")),
        "close": _safe_float(row.get("Close")),
        "volume": int(row.get("Volume") or 0),
    }


def _options_flow_snapshot(options: dict) -> dict:
    calls = options.get("calls") or []
    puts = options.get("puts") or []
    return {
        "total_call_volume": int(options.get("total_call_volume") or 0),
        "total_put_volume": int(options.get("total_put_volume") or 0),
        "call_put_ratio": _safe_float(options.get("call_put_ratio")),
        "chain_call_count": len(calls),
        "chain_put_count": len(puts),
    }


def _contract_tier_snapshot(tier: Optional[dict]) -> Optional[dict]:
    if not tier:
        return None
    return {
        "strike": _safe_float(tier.get("strike")),
        "expiration": tier.get("expiration"),
        "direction": tier.get("direction"),
        "bid": _safe_float(tier.get("bid")),
        "ask": _safe_float(tier.get("ask")),
        "mid_price": _safe_float(tier.get("mid_price")),
        "spread": _safe_float(tier.get("spread")),
        "spread_pct": _safe_float(tier.get("spread_pct")),
        "volume": tier.get("volume"),
        "open_interest": tier.get("open_interest"),
        "iv_pct": tier.get("iv_pct"),
        "dte_days": tier.get("dte_days"),
        "pct_otm": _safe_float(tier.get("pct_otm")),
        "cost_per_contract": _safe_float(tier.get("cost_per_contract")),
        "label": tier.get("label"),
    }


def _pick_contracts_for_directions(
    symbol: str,
    price: float,
    options: dict,
    config: dict,
) -> dict[str, Optional[dict]]:
    """ATM contract quote for call and put (when chain available)."""
    out: dict[str, Optional[dict]] = {"call": None, "put": None}
    budget = config.get("budget", {}).get("total_usd", 500)
    try:
        from orchestrate import pick_option_contract
        for direction in ("call", "put"):
            contract = pick_option_contract(
                symbol, direction, price, options, "0-1 days", budget, config,
            )
            tier = pick_contract_tier({"recommended_contract": contract})
            out[direction] = _contract_tier_snapshot(tier)
    except Exception:
        pass
    return out


def _derived_market_context(
    price: float,
    prev_close: float,
    vwap: float,
    or_high: float,
    or_low: float,
    ema9: float,
    ema21: float,
) -> dict:
    vwap_dist = (price - vwap) / vwap if vwap > 0 else 0.0
    or_mid = (or_high + or_low) / 2 if or_high > 0 and or_low > 0 else 0.0
    or_dist = (price - or_mid) / or_mid if or_mid > 0 else 0.0
    ema_spread = (ema9 - ema21) / price if price > 0 else 0.0
    return {
        "above_vwap": price > vwap,
        "in_opening_range": or_low <= price <= or_high if or_low > 0 else None,
        "or_dist_pct": _safe_float(or_dist * 100),
        "ema_spread_pct": _safe_float(ema_spread * 100),
        "bar_change_pct": _safe_float((price - prev_close) / prev_close * 100) if prev_close else None,
    }


def compute_freshness_flags(
    direction: str,
    price: float,
    prev_close: float,
    vwap: float,
    or_high: float,
    or_low: float,
    rsi: Optional[float],
    prev_rsi: Optional[float],
) -> dict[str, bool]:
    d = direction.lower()
    if d == "call":
        return {
            "fresh_or_break": or_high > 0 and price > or_high and prev_close <= or_high,
            "fresh_vwap_cross": price > vwap and prev_close <= vwap,
            "rsi_momentum_ok": (
                rsi is not None
                and prev_rsi is not None
                and rsi > prev_rsi + 0.5
            ),
        }
    return {
        "fresh_or_break": or_low > 0 and price < or_low and prev_close >= or_low,
        "fresh_vwap_cross": price < vwap and prev_close >= vwap,
        "rsi_momentum_ok": (
            rsi is not None
            and prev_rsi is not None
            and rsi < prev_rsi - 0.5
        ),
    }


def compute_anti_chase_flags(
    direction: str,
    price: float,
    vwap: float,
    rsi: Optional[float],
    cfg: dict,
) -> dict[str, bool]:
    rsi_call = float(cfg.get("anti_chase_rsi_call", 62.0))
    rsi_put = float(cfg.get("anti_chase_rsi_put", 38.0))
    vwap_pct = float(cfg.get("anti_chase_vwap_pct", 0.0015))
    d = direction.lower()
    if d == "call":
        block_rsi = rsi is not None and rsi > rsi_call
        block_vwap = vwap > 0 and (price - vwap) / vwap > vwap_pct
    else:
        block_rsi = rsi is not None and rsi < rsi_put
        block_vwap = vwap > 0 and (vwap - price) / vwap > vwap_pct
    return {
        "block_rsi": block_rsi,
        "block_vwap": block_vwap,
        "would_block": block_rsi or block_vwap,
    }


def momentum_freshness_passes(flags: dict[str, bool]) -> bool:
    return bool(
        flags.get("fresh_or_break")
        or flags.get("fresh_vwap_cross")
        or flags.get("rsi_momentum_ok")
    )


def build_scan_snapshot(
    symbol: str,
    bars: pd.DataFrame,
    options: dict,
    cfg: dict,
    tcfg: dict,
    *,
    scan_timestamp: str,
    scan_source: str = "scan_10m",
    scored: Optional[dict] = None,
    dedup_minutes: int = DEFAULT_DEDUP_MINUTES,
    min_score: Optional[float] = None,
    config: Optional[dict] = None,
) -> dict:
    """Full per-symbol snapshot for framework backtesting."""
    min_score = float(min_score if min_score is not None else cfg.get("min_score", DEFAULT_MIN_SCORE))
    sym = symbol.upper()

    if bars.empty or len(bars) < 3:
        return {
            "record_type": "scan_snapshot",
            "scan_timestamp": scan_timestamp,
            "scan_source": scan_source,
            "symbol": sym,
            "error": "insufficient_bars",
        }

    price = float(bars["Close"].iloc[-1])
    prev_close = float(bars["Close"].iloc[-2])
    vwap = compute_vwap(bars)
    or_high, or_low = opening_range(bars, cfg.get("or_minutes", OR_MINUTES))
    rsi = compute_rsi(bars["Close"])
    prev_rsi = compute_rsi(bars["Close"].iloc[:-1]) if len(bars) > 15 else None
    closes = bars["Close"]
    ema9 = float(closes.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
    prev_ema9 = float(closes.ewm(span=9, adjust=False).mean().iloc[-2])
    prev_ema21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-2])

    vol_so_far = int(bars["Volume"].sum())
    mins = max(minutes_since_open(), 1)
    vol_per_min = vol_so_far / mins
    expected_pace = vol_per_min * 390
    avg_daily = float(bars["Volume"].sum())
    try:
        import yfinance as yf
        hist = yf.Ticker(sym).history(period="12d", interval="1d", auto_adjust=True)
        if hist is not None and len(hist) >= 5:
            avg_daily = float(hist["Volume"].tail(10).mean())
    except Exception:
        pass
    rel_vol = round(expected_pace / avg_daily, 2) if avg_daily > 0 else 1.0

    call_score, call_signals = _score_direction(
        "call", price, vwap, or_high, or_low, rsi, ema9, ema21, rel_vol, options, cfg,
    )
    put_score, put_signals = _score_direction(
        "put", price, vwap, or_high, or_low, rsi, ema9, ema21, rel_vol, options, cfg,
    )
    if call_score >= put_score:
        best_dir, best_score = "call", call_score
    else:
        best_dir, best_score = "put", put_score

    vwap_dist = (price - vwap) / vwap if vwap > 0 else 0.0
    fresh_call = compute_freshness_flags(
        "call", price, prev_close, vwap, or_high, or_low, rsi, prev_rsi,
    )
    fresh_put = compute_freshness_flags(
        "put", price, prev_close, vwap, or_high, or_low, rsi, prev_rsi,
    )
    anti_call = compute_anti_chase_flags("call", price, vwap, rsi, tcfg)
    anti_put = compute_anti_chase_flags("put", price, vwap, rsi, tcfg)
    fresh_best = fresh_call if best_dir == "call" else fresh_put
    anti_best = anti_call if best_dir == "call" else anti_put

    qualifies = best_score >= min_score
    candidate = {
        "symbol": sym,
        "direction": best_dir if qualifies else "skip",
        "score": round(best_score, 3),
        "underlying_price": price,
        "vwap": round(vwap, 2),
    }
    dedup_blocked = False
    if qualifies:
        dedup_blocked = not should_fire_alert(candidate, dedup_minutes)

    atm_mid = None
    atm_strike = None
    atm_exp = None
    atm_contract = None
    full_config = config or {}
    try:
        from orchestrate import pick_option_contract
        contract = pick_option_contract(
            sym, best_dir, price, options, "0-1 days",
            full_config.get("budget", {}).get("total_usd", 500),
            full_config,
        )
        tier = pick_contract_tier({"recommended_contract": contract})
        if tier:
            atm_mid = _safe_float(tier.get("mid_price"))
            atm_strike = _safe_float(tier.get("strike"))
            atm_exp = tier.get("expiration")
            atm_contract = _contract_tier_snapshot(tier)
    except Exception:
        pass

    last_bar_ts = bars.index[-1]
    if hasattr(last_bar_ts, "isoformat"):
        bar_close_time = last_bar_ts.isoformat()
    else:
        bar_close_time = str(last_bar_ts)

    snap = {
        "record_type": "scan_snapshot",
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "scan_timestamp": scan_timestamp,
        "scan_source": scan_source,
        "minutes_since_open": round(minutes_since_open(), 1),
        "symbol": sym,
        "bar_close_time": bar_close_time,
        "underlying_price": _safe_float(price),
        "prev_close": _safe_float(prev_close),
        "vwap": _safe_float(vwap),
        "vwap_dist_pct": _safe_float(vwap_dist * 100),
        "or_high": _safe_float(or_high),
        "or_low": _safe_float(or_low),
        "rsi": _safe_float(rsi),
        "prev_rsi": _safe_float(prev_rsi),
        "ema9": _safe_float(ema9),
        "ema21": _safe_float(ema21),
        "prev_ema9": _safe_float(prev_ema9),
        "prev_ema21": _safe_float(prev_ema21),
        "relative_volume": rel_vol,
        "call_score": round(call_score, 3),
        "put_score": round(put_score, 3),
        "best_direction": best_dir,
        "best_score": round(best_score, 3),
        "min_score": min_score,
        "qualifies_entry": qualifies,
        "dedup_blocked": dedup_blocked if qualifies else False,
        "would_fire_entry": qualifies and not dedup_blocked,
        "freshness_call": fresh_call,
        "freshness_put": fresh_put,
        "freshness_best_passes": momentum_freshness_passes(fresh_best),
        "anti_chase_call": anti_call,
        "anti_chase_put": anti_put,
        "anti_chase_best_blocks": anti_best["would_block"],
        "atm_option_mid": atm_mid,
        "atm_strike": atm_strike,
        "atm_expiration": atm_exp,
    }
    if scored:
        snap["fired_direction"] = scored.get("direction")
        snap["fired_score"] = scored.get("score")
        snap["alert_fired_this_scan"] = scored.get("direction") in ("call", "put")
        if scored.get("skip_reason"):
            snap["skip_reason"] = scored.get("skip_reason")
        if scored.get("would_have_direction"):
            snap["would_have_direction"] = scored.get("would_have_direction")
        if scored.get("key_signals"):
            snap["key_signals"] = scored.get("key_signals")
        rc = scored.get("recommended_contract")
        if rc:
            tier = pick_contract_tier({"recommended_contract": rc})
            snap["fired_contract"] = _contract_tier_snapshot(tier)

    if tcfg.get("rich_logging", True):
        snap.update(_derived_market_context(
            price, prev_close, vwap, or_high, or_low, ema9, ema21,
        ))
        snap["session_volume"] = vol_so_far
        snap["avg_daily_volume"] = int(avg_daily) if avg_daily else None
        snap["bar_latest"] = _bar_ohlcv(bars, -1)
        snap["bar_prev"] = _bar_ohlcv(bars, -2)
        snap["call_signals"] = call_signals
        snap["put_signals"] = put_signals
        snap["options_flow"] = _options_flow_snapshot(options)
        if atm_contract:
            snap["atm_contract"] = atm_contract
        if full_config:
            snap["contracts_by_direction"] = _pick_contracts_for_directions(
                sym, price, options, full_config,
            )

    return snap


def build_position_mark(
    entry: dict,
    scan_timestamp: str,
    tcfg: dict,
) -> Optional[dict]:
    """Option mark for an open entry — enables premium-stop backtests."""
    contract = pick_contract_tier(entry)
    if not contract:
        return None
    entry_mid = float(contract.get("mid_price") or 0)
    if entry_mid <= 0:
        return None

    strike = float(contract["strike"])
    expiration = str(contract["expiration"])
    symbol = entry.get("symbol", "")
    direction = entry.get("direction", "")

    try:
        mark_date = date.fromisoformat(scan_timestamp[:10])
    except ValueError:
        mark_date = date.today()

    current_mid = option_mid_on_date(symbol, direction, strike, expiration, mark_date)
    if current_mid is None:
        return None

    pnl_pct = round((current_mid - entry_mid) / entry_mid * 100, 2)
    stop_pct = float(tcfg.get("premium_stop_pct", -30.0))

    entry_contract = _contract_tier_snapshot(contract)
    hold_minutes = None
    try:
        from intraday_0dte import minutes_since_entry
        hold_minutes = round(minutes_since_entry(entry), 1)
    except Exception:
        pass

    mark = {
        "record_type": "position_mark",
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "scan_timestamp": scan_timestamp,
        "entry_scan_timestamp": entry.get("scan_timestamp"),
        "symbol": symbol.upper(),
        "direction": direction.lower(),
        "strike": strike,
        "expiration": expiration,
        "entry_option_mid": round(entry_mid, 4),
        "current_option_mid": round(float(current_mid), 4),
        "option_pnl_pct": pnl_pct,
        "premium_stop_pct": stop_pct,
        "premium_stop_would_exit": pnl_pct <= stop_pct,
        "entry_underlying_price": _safe_float(entry.get("underlying_price")),
        "hold_minutes": hold_minutes,
        "entry_contract": entry_contract,
    }
    return mark


def build_exit_eval_record(
    eval_data: dict,
    scan_timestamp: str,
) -> dict:
    """Telemetry record for exit-reversal eval (fired or not)."""
    return {
        "record_type": "exit_eval",
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "scan_timestamp": scan_timestamp,
        **eval_data,
    }


def build_scan_run_record(
    scan_timestamp: str,
    scan_source: str,
    symbols: list[str],
    run_meta: dict,
) -> dict:
    return {
        "record_type": "scan_run",
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "scan_timestamp": scan_timestamp,
        "scan_source": scan_source,
        "symbols": [s.upper() for s in symbols],
        **run_meta,
    }


def log_scan_telemetry(
    config: dict,
    symbols: list[str],
    *,
    scan_source: str = "scan_10m",
    scored_by_symbol: Optional[dict[str, dict]] = None,
    open_positions: Optional[list[dict]] = None,
    exit_evals: Optional[list[dict]] = None,
    run_meta: Optional[dict] = None,
) -> int:
    """Write scan_snapshot (+ optional position_mark) records. Returns count written."""
    tcfg = telemetry_cfg(config)
    if not tcfg.get("enabled", True):
        return 0
    if scan_source == "scan_5m" and not tcfg.get("five_min_snapshots", True):
        return 0
    if scan_source == "scan_10m" and not tcfg.get("snapshot_on_scan", True):
        return 0

    cfg = intraday_cfg(config)
    scan_ts = datetime.now(ET).isoformat()
    dedup = int(cfg.get("dedup_minutes", DEFAULT_DEDUP_MINUTES))
    written = 0

    for symbol in symbols:
        sym = symbol.upper()
        bars = fetch_intraday_bars(sym)
        options = fetch_0dte_options(sym, dte_max=int(cfg.get("dte_max", 1)))
        scored = (scored_by_symbol or {}).get(sym)
        snap = build_scan_snapshot(
            sym, bars, options, cfg, tcfg,
            scan_timestamp=scan_ts,
            scan_source=scan_source,
            scored=scored,
            dedup_minutes=dedup,
            config=config,
        )
        append_telemetry(snap, config)
        written += 1

    if tcfg.get("position_marks_on_scan", True):
        positions = open_positions if open_positions is not None else load_open_positions()
        for entry in positions:
            mark = build_position_mark(entry, scan_ts, tcfg)
            if mark:
                u = fetch_intraday_bars(entry["symbol"].upper())
                if not u.empty:
                    cur_u = _safe_float(u["Close"].iloc[-1])
                    mark["underlying_price"] = cur_u
                    entry_u = mark.get("entry_underlying_price")
                    if entry_u and cur_u and entry.get("direction") == "call":
                        mark["underlying_move_pct"] = _safe_float(
                            (cur_u - entry_u) / entry_u * 100,
                        )
                    elif entry_u and cur_u:
                        mark["underlying_move_pct"] = _safe_float(
                            (entry_u - cur_u) / entry_u * 100,
                        )
                append_telemetry(mark, config)
                written += 1

    if (
        scan_source == "scan_10m"
        and tcfg.get("log_exit_evals", True)
        and exit_evals
    ):
        for ev in exit_evals:
            append_telemetry(build_exit_eval_record(ev, scan_ts), config)
            written += 1

    if tcfg.get("log_scan_run", True) and run_meta is not None:
        append_telemetry(
            build_scan_run_record(scan_ts, scan_source, symbols, run_meta),
            config,
        )
        written += 1
    elif tcfg.get("log_scan_run", True) and scan_source == "scan_5m":
        append_telemetry(
            build_scan_run_record(
                scan_ts,
                scan_source,
                symbols,
                {"symbols_scanned": len(symbols), "entries_fired": 0, "exits_fired": 0},
            ),
            config,
        )
        written += 1

    logger.info(f"Telemetry: wrote {written} record(s) source={scan_source}")
    return written


def run_telemetry_only(
    config: Optional[dict] = None,
    *,
    scan_source: str = "scan_5m",
    symbols: Optional[list[str]] = None,
) -> int:
    """Lightweight snapshot run (no alerts). Used by 5-min cron."""
    config = config or load_config()
    cfg = intraday_cfg(config)
    syms = symbols or cfg.get("symbols", ["SPY", "QQQ", "IWM"])

    if not is_trading_day():
        logger.info("NYSE closed — telemetry skipped")
        return 0
    if not is_market_hours():
        logger.info("Outside market hours — telemetry skipped")
        return 0
    or_wait = int(cfg.get("or_wait_minutes", OR_MINUTES))
    if minutes_since_open() < or_wait:
        logger.info(f"Pre-OR ({or_wait}m) — telemetry skipped")
        return 0

    return log_scan_telemetry(config, [s.upper() for s in syms], scan_source=scan_source)


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Log intraday telemetry snapshots")
    parser.add_argument(
        "--source", default="scan_5m",
        choices=["scan_5m", "scan_10m"],
        help="scan_5m = lightweight cron; scan_10m = same as full scan hook",
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    args = parser.parse_args()
    n = run_telemetry_only(scan_source=args.source, symbols=args.symbols)
    print(f"Wrote {n} telemetry record(s)")


if __name__ == "__main__":
    main()
