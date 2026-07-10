#!/usr/bin/env python3
"""
Replay a day's intraday alerts through the 5 proposed framework filters
and compare P&L vs baseline (actual fired alerts).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

# Project imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import option_outcome as oo
from intraday_0dte import (
    compute_rsi,
    compute_vwap,
    fetch_0dte_options,
    opening_range,
    score_symbol,
    find_exit_for_entry,
)

ET = ZoneInfo("America/New_York")
ALERTS_PATH = Path("/tmp/intraday_0dte_alerts.jsonl")
CONFIG_PATH = ROOT / "config.json"

# Framework parameters (from prior discussion)
ANTI_CHASE_RSI_CALL = 62.0
ANTI_CHASE_RSI_PUT = 38.0
ANTI_CHASE_VWAP_PCT = 0.0015  # 0.15% extended from VWAP
PREMIUM_STOP_PCT = -30.0
DEDUP_ONE_PER_SYMBOL = True


def load_alerts() -> list[dict]:
    return [json.loads(l) for l in ALERTS_PATH.read_text().strip().splitlines() if l.strip()]


def parse_rsi(entry: dict) -> float | None:
    p = entry.get("patterns") or {}
    if p.get("rsi") is not None:
        return float(p["rsi"])
    for sig in entry.get("key_signals") or []:
        m = re.search(r"RSI ([\d.]+)", sig)
        if m:
            return float(m.group(1))
    return None


def entry_mid(entry: dict) -> float | None:
    rc = entry.get("recommended_contract") or {}
    for k in ("atm", "slight_otm", "affordable"):
        t = (rc.get("tiers") or {}).get(k)
        if t and t.get("mid_price"):
            return float(t["mid_price"])
    return None


def reflect_pnl(entry: dict, day_alerts: list[dict], config: dict) -> dict:
    exit_a = find_exit_for_entry(entry, day_alerts)
    scored = oo.evaluate_intraday_alert(entry, config, exit_alert=exit_a)
    return {
        "pnl": scored.get("outcome_option_pnl_pct"),
        "miss": scored.get("miss_type"),
        "had_exit": exit_a is not None,
        "exit_ts": (exit_a or {}).get("scan_timestamp"),
        "entry_mid": scored.get("option_entry_mid") or entry_mid(entry),
    }


def fetch_day_bars(symbol: str, target: date) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="5d", interval="5m", auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET)
    else:
        df.index = df.index.tz_convert(ET)
    return df[df.index.date == target].copy()


def bars_at_time(full: pd.DataFrame, ts: datetime) -> pd.DataFrame:
    return full[full.index <= ts].copy()


def anti_chase_pass(entry: dict) -> tuple[bool, str]:
    direction = entry["direction"].lower()
    rsi = parse_rsi(entry)
    price = float(entry.get("underlying_price") or 0)
    vwap = float(entry.get("vwap") or 0)
    if direction == "call":
        if rsi is not None and rsi > ANTI_CHASE_RSI_CALL:
            return False, f"RSI {rsi:.1f} > {ANTI_CHASE_RSI_CALL}"
        if vwap > 0 and (price - vwap) / vwap > ANTI_CHASE_VWAP_PCT:
            ext = (price - vwap) / vwap * 100
            return False, f"{ext:.2f}% above VWAP (max {ANTI_CHASE_VWAP_PCT*100:.2f}%)"
    else:
        if rsi is not None and rsi < ANTI_CHASE_RSI_PUT:
            return False, f"RSI {rsi:.1f} < {ANTI_CHASE_RSI_PUT}"
        if vwap > 0 and (vwap - price) / vwap > ANTI_CHASE_VWAP_PCT:
            ext = (vwap - price) / vwap * 100
            return False, f"{ext:.2f}% below VWAP"
    return True, "ok"


def momentum_freshness_pass(entry: dict, bars_full: pd.DataFrame) -> tuple[bool, str]:
    ts = datetime.fromisoformat(entry["scan_timestamp"]).astimezone(ET)
    bars = bars_at_time(bars_full, ts)
    if len(bars) < 4:
        return False, "insufficient bars"

    direction = entry["direction"].lower()
    price = float(bars["Close"].iloc[-1])
    prev_close = float(bars["Close"].iloc[-2])
    vwap = compute_vwap(bars)
    or_high, or_low = opening_range(bars)
    rsi = compute_rsi(bars["Close"])
    prev_rsi = compute_rsi(bars["Close"].iloc[:-1])

    if direction == "call":
        fresh_or = or_high > 0 and price > or_high and prev_close <= or_high
        fresh_vwap = price > vwap and prev_close <= vwap
        rsi_rising = rsi is not None and prev_rsi is not None and rsi > prev_rsi + 0.5
        if fresh_or:
            return True, "fresh OR high break"
        if fresh_vwap:
            return True, "fresh VWAP reclaim"
        if rsi_rising and rsi is not None and rsi <= ANTI_CHASE_RSI_CALL:
            return True, f"RSI rising ({prev_rsi:.1f}→{rsi:.1f})"
        return False, "no fresh momentum (extended/chop)"
    else:
        fresh_or = or_low > 0 and price < or_low and prev_close >= or_low
        fresh_vwap = price < vwap and prev_close >= vwap
        rsi_falling = rsi is not None and prev_rsi is not None and rsi < prev_rsi - 0.5
        if fresh_or:
            return True, "fresh OR low break"
        if fresh_vwap:
            return True, "fresh VWAP loss"
        if rsi_falling and rsi is not None and rsi >= ANTI_CHASE_RSI_PUT:
            return True, f"RSI falling ({prev_rsi:.1f}→{rsi:.1f})"
        return False, "no fresh momentum"


def one_per_symbol_filter(entries: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out = []
    for e in sorted(entries, key=lambda x: x["scan_timestamp"]):
        key = (e["symbol"].upper(), e["direction"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def premium_stop_pnl(
    entry: dict,
    actual_pnl: float | None,
    bars_full: pd.DataFrame,
) -> float | None:
    """Simulate -30% hard stop using underlying path + delta proxy."""
    if actual_pnl is None:
        return None
    if actual_pnl >= PREMIUM_STOP_PCT:
        return actual_pnl

    em = entry_mid(entry)
    eu = float(entry.get("underlying_price") or 0)
    if not em or not eu:
        return PREMIUM_STOP_PCT

    ts0 = datetime.fromisoformat(entry["scan_timestamp"]).astimezone(ET)
    direction = entry["direction"].lower()
    bars = bars_full[bars_full.index > ts0]
    if bars.empty:
        return PREMIUM_STOP_PCT

    # Calibrate leverage from actual outcome if available
    if actual_pnl is not None and actual_pnl != 0:
        # find last bar before typical exit (~11:20 for many today)
        last_u = float(bars["Close"].iloc[-1])
        if direction == "call":
            und_pct = (last_u - eu) / eu * 100
        else:
            und_pct = (eu - last_u) / eu * 100
        leverage = abs(actual_pnl / und_pct) if abs(und_pct) > 0.02 else 20.0
        leverage = min(max(leverage, 8.0), 40.0)
    else:
        leverage = 20.0

    for _, row in bars.iterrows():
        u = float(row["Close"])
        if direction == "call":
            und_pct = (u - eu) / eu * 100
        else:
            und_pct = (eu - u) / eu * 100
        est = und_pct * leverage
        if est <= PREMIUM_STOP_PCT:
            return PREMIUM_STOP_PCT
    return PREMIUM_STOP_PCT


def rescore_every_5min(
    target: date,
    symbols: list[str],
    config: dict,
    filters: list[str],
) -> list[dict]:
    """F5: re-run scanner every 5 min; apply optional entry filters."""
    cfg = config.get("intraday_0dte", {})
    min_score = float(cfg.get("min_score", 0.70))
    candidates: list[dict] = []

    open_dt = datetime(target.year, target.month, target.day, 9, 45, tzinfo=ET)
    close_dt = datetime(target.year, target.month, target.day, 15, 50, tzinfo=ET)
    t = open_dt
    bar_cache: dict[str, pd.DataFrame] = {}
    options_cache: dict[str, dict] = {}

    while t <= close_dt:
        for sym in symbols:
            if sym not in bar_cache:
                bar_cache[sym] = fetch_day_bars(sym, target)
                options_cache[sym] = fetch_0dte_options(sym)
            bars = bars_at_time(bar_cache[sym], t)
            if len(bars) < 3:
                continue
            scored = score_symbol(sym, bars, options_cache[sym], cfg)
            if scored.get("direction") == "skip":
                continue
            if scored.get("score", 0) < min_score:
                continue
            scored["scan_timestamp"] = t.isoformat()
            scored["underlying_price"] = float(bars["Close"].iloc[-1])
            scored["vwap"] = round(compute_vwap(bars), 2)
            oh, ol = opening_range(bars)
            scored["or_high"] = round(oh, 2)
            scored["or_low"] = round(ol, 2)
            rsi = compute_rsi(bars["Close"])
            scored["patterns"] = {"rsi": rsi}

            ok = True
            reason = ""
            if "anti_chase" in filters:
                ok, reason = anti_chase_pass(scored)
            if ok and "freshness" in filters:
                ok, reason = momentum_freshness_pass(scored, bar_cache[sym])
            if ok:
                candidates.append({**scored, "_filter_reason": reason or "5m scan"})
        t += timedelta(minutes=5)

    # one per symbol first hit
    if "one_per_symbol" in filters:
        candidates = one_per_symbol_filter(candidates)
    return candidates


def summarize(name: str, entries: list[dict], day_alerts: list[dict], config: dict,
              bars_by_sym: dict[str, pd.DataFrame], use_premium_stop: bool = False) -> dict:
    rows = []
    for e in entries:
        r = reflect_pnl(e, day_alerts, config)
        pnl = r["pnl"]
        if use_premium_stop and pnl is not None:
            sym = e["symbol"].upper()
            pnl = premium_stop_pnl(e, pnl, bars_by_sym.get(sym, pd.DataFrame()))
        rows.append({
            "time": e["scan_timestamp"][11:16],
            "sym": e["symbol"],
            "dir": e["direction"],
            "score": e.get("score"),
            "pnl": pnl,
            "em": r["entry_mid"],
        })

    pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
    dollars = sum(r["em"] * r["pnl"] / 100 for r in rows if r["em"] and r["pnl"] is not None)
    wins = sum(1 for p in pnls if p > 0)
    return {
        "name": name,
        "count": len(rows),
        "wins": wins,
        "win_pct": 100 * wins / len(pnls) if pnls else 0,
        "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
        "total_usd": dollars,
        "rows": rows,
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    alerts = load_alerts()
    # Latest day with entry alerts
    entry_dates = sorted({
        a["scan_timestamp"][:10] for a in alerts
        if a.get("alert_action", "entry") == "entry"
        and a.get("direction") in ("call", "put")
    })
    target = date.fromisoformat(entry_dates[-1])
    target_str = target.isoformat()

    day_alerts = [a for a in alerts if a.get("scan_timestamp", "").startswith(target_str)]
    actual_entries = sorted(
        [a for a in day_alerts if a.get("alert_action", "entry") == "entry"
         and a.get("direction") in ("call", "put")],
        key=lambda x: x["scan_timestamp"],
    )

    symbols = sorted({e["symbol"].upper() for e in actual_entries})
    bars_by_sym = {s: fetch_day_bars(s, target) for s in symbols}

    print(f"=== Framework replay: {target_str} ({len(actual_entries)} actual entry alerts) ===\n")

    scenarios: list[tuple[str, list[dict], bool]] = []

    # Baseline
    scenarios.append(("0. Baseline (actual alerts)", actual_entries, False))

    # F1 Anti-chase only
    f1 = [e for e in actual_entries if anti_chase_pass(e)[0]]
    scenarios.append(("1. Anti-chase filter", f1, False))

    # F2 Momentum freshness only
    f2 = [e for e in actual_entries if momentum_freshness_pass(e, bars_by_sym[e["symbol"].upper()])[0]]
    scenarios.append(("2. Momentum freshness", f2, False))

    # F3 Premium stop on baseline entries
    scenarios.append(("3. Premium stop -30% (same entries)", actual_entries, True))

    # F4 One entry per symbol
    f4 = one_per_symbol_filter(actual_entries)
    scenarios.append(("4. One entry per symbol/day", f4, False))

    # F5 5-min rescan (no extra filters)
    f5_entries = rescore_every_5min(target, symbols, config, filters=[])
    scenarios.append(("5. 5-min rescan (first pass per symbol)", one_per_symbol_filter(f5_entries), False))

    # Combined: all entry filters + premium stop
    combined = [e for e in actual_entries if anti_chase_pass(e)[0]]
    combined = [e for e in combined if momentum_freshness_pass(e, bars_by_sym[e["symbol"].upper()])[0]]
    combined = one_per_symbol_filter(combined)
    scenarios.append(("ALL: F1+F2+F4 + premium stop", combined, True))

    # Combined without premium stop
    scenarios.append(("ALL: F1+F2+F4 (no premium stop)", combined, False))

    results = []
    for name, entries, prem in scenarios:
        results.append(summarize(name, entries, day_alerts, config, bars_by_sym, prem))

    # Print comparison table
    print(f"{'Scenario':<42} {'Alerts':>6} {'Wins':>5} {'Win%':>6} {'Avg P&L':>9} {'Total $':>9}")
    print("-" * 82)
    for r in results:
        print(f"{r['name']:<42} {r['count']:>6} {r['wins']:>5} {r['win_pct']:>5.0f}% "
              f"{r['avg_pnl']:>+8.1f}% ${r['total_usd']:>+8.2f}")

    # Detail: what each filter blocked on baseline
    print("\n=== Baseline entries — filter decisions ===")
    print(f"{'Time':<6} {'Sym':<4} {'Sc':>4} {'Base P&L':>9}  F1  F2  kept in ALL")
    for e in actual_entries:
        r = reflect_pnl(e, day_alerts, config)
        f1_ok, f1_r = anti_chase_pass(e)
        f2_ok, f2_r = momentum_freshness_pass(e, bars_by_sym[e["symbol"].upper()])
        in_all = f1_ok and f2_ok and e in combined
        print(f"{e['scan_timestamp'][11:16]:<6} {e['symbol']:<4} {e['score']:>4.2f} "
              f"{(r['pnl'] if r['pnl'] is not None else 'n/a'):>9}  "
              f"{'✓' if f1_ok else '✗':>2}  {'✓' if f2_ok else '✗':>2}  "
              f"{'✓' if in_all else '✗'}")

    print("\n=== Block reasons (F1 anti-chase failures) ===")
    for e in actual_entries:
        ok, reason = anti_chase_pass(e)
        if not ok:
            print(f"  {e['scan_timestamp'][11:16]} {e['symbol']} score={e['score']:.2f}: {reason}")

    print("\n=== Block reasons (F2 freshness failures) ===")
    for e in actual_entries:
        ok, reason = momentum_freshness_pass(e, bars_by_sym[e["symbol"].upper()])
        if not ok:
            print(f"  {e['scan_timestamp'][11:16]} {e['symbol']} score={e['score']:.2f}: {reason}")

    print("\n=== ALL combined — trades kept ===")
    for e in combined:
        r = reflect_pnl(e, day_alerts, config)
        pnl = premium_stop_pnl(e, r["pnl"], bars_by_sym[e["symbol"].upper()]) if r["pnl"] is not None else r["pnl"]
        print(f"  {e['scan_timestamp'][11:16]} {e['symbol']} {e['direction']} score={e['score']:.2f} "
              f"reflect={r['pnl']} → with stop={pnl}")

    print("\n=== F5 5-min rescan — first qualifying entry per symbol ===")
    for e in one_per_symbol_filter(f5_entries):
        r = reflect_pnl(e, day_alerts, config)
        print(f"  {e['scan_timestamp'][11:16]} {e['symbol']} {e['direction']} score={e['score']:.2f} "
              f"(actual first alert may differ)")


if __name__ == "__main__":
    main()
