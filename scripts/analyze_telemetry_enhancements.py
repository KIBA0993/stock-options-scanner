#!/usr/bin/env python3
"""Analyze intraday telemetry + alerts to suggest win-rate enhancements."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import option_outcome as oo
from intraday_0dte import find_exit_for_entry, load_alerts, load_config

DEFAULT_DAYS = ["2026-06-24", "2026-06-25", "2026-06-26", "2026-06-29", "2026-06-30"]


def find_snap(alert: dict, snapshots: list[dict]) -> dict | None:
    sym = alert.get("symbol", "").upper()
    ts = alert.get("scan_timestamp", "")[:16]
    for s in snapshots:
        if s.get("symbol") == sym and (s.get("scan_timestamp") or "")[:16] == ts:
            return s
    try:
        adt = datetime.fromisoformat(alert["scan_timestamp"])
    except (ValueError, TypeError, KeyError):
        return None
    best, bd = None, 9999.0
    for s in snapshots:
        if s.get("symbol") != sym:
            continue
        try:
            sdt = datetime.fromisoformat(s["scan_timestamp"])
            delta = abs((sdt - adt).total_seconds())
            if delta < bd and delta <= 600:
                bd = delta
                best = s
        except (ValueError, TypeError):
            pass
    return best


def pnl_for_entry(entry: dict, day_alerts: list[dict], config: dict) -> tuple:
    exit_a = find_exit_for_entry(entry, day_alerts)
    if exit_a:
        r = oo.fetch_intraday_exit_outcome(entry, exit_a)
        if r:
            return r.get("outcome_option_pnl_pct"), "exit", r
    r = oo.no_exit_intraday_outcome(entry, config)
    return r.get("outcome_option_pnl_pct"), "no_exit", r


def load_trades(days: list[str], config: dict) -> list[dict]:
    alerts = load_alerts()
    tel = []
    tel_path = ROOT / "data" / "intraday_0dte_telemetry.jsonl"
    if tel_path.exists():
        for line in tel_path.read_text().splitlines():
            if line.strip():
                tel.append(json.loads(line))

    snaps_by_day: dict[str, list] = defaultdict(list)
    marks_by_day: dict[str, list] = defaultdict(list)
    for r in tel:
        d = (r.get("scan_timestamp") or "")[:10]
        if r.get("record_type") == "scan_snapshot":
            snaps_by_day[d].append(r)
        elif r.get("record_type") == "position_mark":
            marks_by_day[d].append(r)

    trades = []
    for day in days:
        day_alerts = [
            a for a in alerts if (a.get("scan_timestamp") or "").startswith(day)
        ]
        entries = [
            a for a in day_alerts
            if a.get("alert_action", "entry") == "entry"
            and a.get("direction") in ("call", "put")
        ]
        seen: set[tuple] = set()
        for e in sorted(entries, key=lambda x: x["scan_timestamp"]):
            k = (e["scan_timestamp"][:19], e["symbol"], e["direction"])
            if k in seen:
                continue
            seen.add(k)
            pnl, reason, _ = pnl_for_entry(e, day_alerts, config)
            rc = e.get("recommended_contract") or {}
            tier = (rc.get("tiers") or {}).get("atm") or {}
            trades.append({
                "day": day,
                "time": e["scan_timestamp"][11:16],
                "entry_ts": e["scan_timestamp"],
                "symbol": e["symbol"],
                "direction": e["direction"],
                "score": e.get("score"),
                "pnl_pct": pnl,
                "exit_reason": reason,
                "won": (pnl or 0) > 0,
                "snap": find_snap(e, snaps_by_day[day]) or {},
                "tier": tier,
            })
    return trades, marks_by_day, alerts


def avg_field(items: list[dict], key: str) -> float | None:
    vals = []
    for t in items:
        v = (t.get("snap") or {}).get(key)
        if v is not None:
            vals.append(float(v))
    return round(mean(vals), 2) if vals else None


def avg_tier(items: list[dict], key: str) -> float | None:
    vals = [
        float(t["tier"][key])
        for t in items
        if t.get("tier", {}).get(key) is not None
    ]
    return round(mean(vals), 2) if vals else None


def avg_opt(items: list[dict], key: str) -> float | None:
    vals = []
    for t in items:
        of = (t.get("snap") or {}).get("options_flow") or {}
        if of.get(key) is not None:
            vals.append(float(of[key]))
    return round(mean(vals), 2) if vals else None


def trade_usd(t: dict, pnl_override: float | None = None) -> float:
    pnl = pnl_override if pnl_override is not None else (t["pnl_pct"] or -100)
    mid = float((t.get("tier") or {}).get("mid_price") or 0)
    return mid * pnl / 100


def filter_backtest(trades: list[dict], name: str, kept: list[dict]) -> None:
    baseline = sum(trade_usd(t) for t in trades)
    total = sum(trade_usd(t) for t in kept)
    wins = sum(1 for t in kept if t["won"])
    print(
        f"  {name:42} trades={len(kept):3}/{len(trades)} "
        f"wins={wins} total=${total:+.2f} (baseline ${baseline:+.2f})"
    )


def prem_stop_trades(trades: list[dict]) -> list[dict]:
    out = []
    for t in trades:
        pnl = t["pnl_pct"] or -100
        if pnl < -30:
            pnl = -30
        out.append({**t, "pnl_pct_adj": pnl})
    return out


def one_per_symbol(trades: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    kept = []
    for t in sorted(trades, key=lambda x: (x["day"], x["entry_ts"])):
        k = (t["day"], t["symbol"])
        if k in seen:
            continue
        seen.add(k)
        kept.append(t)
    return kept


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", default=DEFAULT_DAYS)
    args = parser.parse_args()

    config = load_config()
    trades, marks_by_day, alerts = load_trades(args.days, config)
    wins = [t for t in trades if t["won"]]
    losses = [t for t in trades if not t["won"]]

    print(f"=== TRADE OUTCOMES ({len(args.days)} days) ===")
    print(
        f"Total unique entries: {len(trades)} | Wins: {len(wins)} | "
        f"Losses: {len(losses)} | Win rate: {100 * len(wins) / max(len(trades), 1):.1f}%"
    )
    print("\nWinners:")
    for t in wins:
        s = t["snap"]
        print(
            f"  {t['day']} {t['time']} {t['symbol']} {t['direction']} "
            f"score={t['score']} pnl={t['pnl_pct']:.1f}% "
            f"vwap_dist={s.get('vwap_dist_pct')} rsi={s.get('rsi')} "
            f"rel_vol={s.get('relative_volume')} fresh={s.get('freshness_best_passes')} "
            f"anti={s.get('anti_chase_best_blocks')} min={s.get('minutes_since_open')}"
        )

    print("\n=== WINNERS vs LOSERS (telemetry at entry) ===")
    for key in (
        "score", "vwap_dist_pct", "rsi", "relative_volume",
        "minutes_since_open", "ema_spread_pct", "or_dist_pct",
        "call_score", "put_score",
    ):
        print(f"  {key:22} winners={avg_field(wins, key)}  losers={avg_field(losses, key)}")
    print(
        f"  call_put_ratio          winners={avg_opt(wins, 'call_put_ratio')}  "
        f"losers={avg_opt(losses, 'call_put_ratio')}"
    )
    print(
        f"  spread_pct (contract)   winners={avg_tier(wins, 'spread_pct')}  "
        f"losers={avg_tier(losses, 'spread_pct')}"
    )
    for flag in (
        "freshness_best_passes", "anti_chase_best_blocks",
        "above_vwap", "in_opening_range",
    ):
        wc = sum(1 for t in wins if (t.get("snap") or {}).get(flag))
        lc = sum(1 for t in losses if (t.get("snap") or {}).get(flag))
        print(f"  {flag:24} winners {wc}/{len(wins)}  losers {lc}/{len(losses)}")

    print("\n=== HYPOTHETICAL FILTERS (5-day combined) ===")
    filter_backtest(trades, "baseline (all)", trades)
    ps = prem_stop_trades(trades)
    filter_backtest(
        trades, "F3 premium stop -30%",
        [{**t, "pnl_pct": t["pnl_pct_adj"]} for t in ps],
    )
    ops = one_per_symbol(trades)
    filter_backtest(trades, "F4 one entry/symbol/day", ops)
    filter_backtest(
        trades, "one/sym + premium stop",
        [{**t, "pnl_pct": min(t["pnl_pct"] or -100, -30) if (t["pnl_pct"] or -100) < -30 else t["pnl_pct"]}
         for t in ops],
    )

    filters = [
        ("anti_chase blocks", lambda t: not (t.get("snap") or {}).get("anti_chase_best_blocks")),
        ("freshness passes", lambda t: (t.get("snap") or {}).get("freshness_best_passes")),
        ("score >= 0.80", lambda t: (t.get("score") or 0) >= 0.80),
        ("score >= 0.85", lambda t: (t.get("score") or 0) >= 0.85),
        ("rel_vol >= 1.5", lambda t: (t.get("snap") or {}).get("relative_volume", 0) >= 1.5),
        ("rel_vol >= 2.0", lambda t: (t.get("snap") or {}).get("relative_volume", 0) >= 2.0),
        ("spread_pct <= 15", lambda t: float((t.get("tier") or {}).get("spread_pct") or 999) <= 15),
        ("spread_pct <= 20", lambda t: float((t.get("tier") or {}).get("spread_pct") or 999) <= 20),
        ("minutes <= 120", lambda t: (t.get("snap") or {}).get("minutes_since_open", 999) <= 120),
        ("minutes <= 90", lambda t: (t.get("snap") or {}).get("minutes_since_open", 999) <= 90),
        ("abs(vwap_dist) <= 0.10%", lambda t: abs((t.get("snap") or {}).get("vwap_dist_pct") or 999) <= 0.10),
        ("abs(vwap_dist) <= 0.20%", lambda t: abs((t.get("snap") or {}).get("vwap_dist_pct") or 999) <= 0.20),
        (
            "call + cp_ratio>1.1",
            lambda t: t["direction"] == "call"
            and ((t.get("snap") or {}).get("options_flow") or {}).get("call_put_ratio", 0) > 1.1,
        ),
        (
            "put + cp_ratio<0.9",
            lambda t: t["direction"] == "put"
            and ((t.get("snap") or {}).get("options_flow") or {}).get("call_put_ratio", 999) < 0.9,
        ),
    ]
    combo_filters = [
        (
            "one/sym + spread<=20",
            lambda ts: one_per_symbol([x for x in ts if float((x.get("tier") or {}).get("spread_pct") or 999) <= 20]),
        ),
        (
            "one/sym + min<=120",
            lambda ts: one_per_symbol([x for x in ts if (x.get("snap") or {}).get("minutes_since_open", 999) <= 120]),
        ),
        (
            "one/sym + score>=0.80",
            lambda ts: one_per_symbol([x for x in ts if (x.get("score") or 0) >= 0.80]),
        ),
    ]
    for name, fn in filters:
        kept = [t for t in trades if fn(t)]
        filter_backtest(trades, name, kept)
    for name, fn in combo_filters:
        kept = fn(trades)
        filter_backtest(trades, name, kept)

    print("\n=== EXIT TIMING (peak vs actual on winners) ===")
    for t in wins:
        marks = [
            m for m in marks_by_day.get(t["day"], [])
            if m.get("entry_scan_timestamp") == t["entry_ts"]
        ]
        if not marks:
            continue
        peak = max(m.get("option_pnl_pct", -999) for m in marks)
        final = t["pnl_pct"] or 0
        if peak > final:
            print(
                f"  {t['day']} {t['symbol']} {t['direction']}: "
                f"peak={peak:.1f}% exit={final:.1f}% (left {peak - final:.1f}% on table)"
            )
        else:
            print(f"  {t['day']} {t['symbol']} {t['direction']}: peak={peak:.1f}% exit={final:.1f}%")

    print("\n=== CHURN (2+ entries same symbol/day) ===")
    by_sym_day: dict[tuple, list] = defaultdict(list)
    for t in trades:
        by_sym_day[(t["day"], t["symbol"])].append(t)
    multi = {k: v for k, v in by_sym_day.items() if len(v) > 1}
    churn_loss = sum(sum(trade_usd(x) for x in v) for v in multi.values())
    print(f"Symbol-days with 2+ entries: {len(multi)}")
    print(f"Combined P&L on churned symbol-days: ${churn_loss:+.2f}")

    print("\n=== 5-DAY FRAMEWORK TOTALS ===")
    scenarios = [
        "0_baseline_actual", "3_premium_stop", "4_one_per_symbol_stop",
        "1_anti_chase", "2_momentum_freshness", "5_rescan_5m", "all_f1_f2_f4_stop",
    ]
    totals = {s: 0.0 for s in scenarios}
    for day in args.days:
        p = ROOT / "data" / "framework_backtest" / f"{day}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for sc in d["scenarios"]:
            if sc["name"] in totals:
                totals[sc["name"]] += sc["total_usd"]
    for s in scenarios:
        print(f"  {s:28} ${totals[s]:+.2f}")


if __name__ == "__main__":
    main()
