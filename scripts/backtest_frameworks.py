#!/usr/bin/env python3
"""
backtest_frameworks.py — Daily replay of 5 intraday frameworks.

Uses collected telemetry (preferred) + fired alerts for P&L.

Data sources:
  data/intraday_0dte_telemetry.jsonl  — scan_snapshot, position_mark
  data/intraday_0dte_alerts.jsonl     — actual entries/exits + contracts

Run after market close:
  python scripts/backtest_frameworks.py
  python scripts/backtest_frameworks.py --date 2026-06-24

Reports → data/framework_backtest/YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import option_outcome as oo
from intraday_0dte import ALERTS_PATH, find_exit_for_entry, load_config
from intraday_telemetry import telemetry_cfg, telemetry_path

REPORT_DIR = ROOT / "data" / "framework_backtest"


@dataclass
class VirtualEntry:
    scan_timestamp: str
    symbol: str
    direction: str
    score: float
    entry_mid: Optional[float]
    source: str
    snapshot: dict = field(default_factory=dict)


@dataclass
class TradeResult:
    entry: VirtualEntry
    exit_reason: str
    exit_timestamp: Optional[str]
    pnl_pct: Optional[float]
    dollars: float = 0.0


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_day_records(records: list[dict], day: str) -> list[dict]:
    return [r for r in records if (r.get("scan_timestamp") or "").startswith(day)]


def entry_mid_from_alert(entry: dict) -> Optional[float]:
    rc = entry.get("recommended_contract") or {}
    for k in ("atm", "slight_otm", "affordable"):
        t = (rc.get("tiers") or {}).get(k)
        if t and t.get("mid_price"):
            return float(t["mid_price"])
    return None


def match_alert_entry(
    snap: dict, alerts: list[dict],
) -> Optional[dict]:
    """Find fired alert near snapshot time for contract pricing."""
    sym = snap["symbol"]
    ts = snap["scan_timestamp"]
    for a in alerts:
        if a.get("alert_action", "entry") != "entry":
            continue
        if a.get("symbol", "").upper() != sym:
            continue
        if a.get("scan_timestamp", "")[:16] == ts[:16]:
            return a
    return None


def snap_to_entry(snap: dict, alert: Optional[dict], source: str) -> VirtualEntry:
    direction = snap.get("best_direction") or snap.get("fired_direction") or "call"
    mid = None
    if alert:
        mid = entry_mid_from_alert(alert)
    if mid is None:
        mid = snap.get("atm_option_mid")
    return VirtualEntry(
        scan_timestamp=snap["scan_timestamp"],
        symbol=snap["symbol"],
        direction=direction,
        score=float(snap.get("best_score") or snap.get("fired_score") or 0),
        entry_mid=float(mid) if mid else None,
        source=source,
        snapshot=snap,
    )


def anti_chase_blocks(snap: dict, direction: str, fcfg: dict) -> bool:
    d = direction.lower()
    key = "anti_chase_call" if d == "call" else "anti_chase_put"
    flags = snap.get(key) or {}
    if flags.get("would_block"):
        return True
    return bool(snap.get("anti_chase_best_blocks"))


def resolve_pnl(
    entry: VirtualEntry,
    day_alerts: list[dict],
    marks: list[dict],
    config: dict,
    *,
    use_premium_stop: bool = False,
) -> TradeResult:
    fcfg = telemetry_cfg(config)
    stop_pct = float(fcfg.get("premium_stop_pct", -30.0))

    alert_entry = None
    for a in day_alerts:
        if (
            a.get("alert_action", "entry") == "entry"
            and a.get("scan_timestamp") == entry.scan_timestamp
        ):
            alert_entry = a
            break
    if not alert_entry:
        # synthetic entry — build minimal alert for reflect
        alert_entry = {
            "scan_timestamp": entry.scan_timestamp,
            "symbol": entry.symbol,
            "direction": entry.direction,
            "underlying_price": entry.snapshot.get("underlying_price"),
            "recommended_contract": {
                "tiers": {
                    "atm": {
                        "mid_price": entry.entry_mid,
                        "strike": entry.snapshot.get("atm_strike"),
                        "expiration": entry.snapshot.get("atm_expiration"),
                    }
                }
            } if entry.entry_mid else None,
        }

    entry_marks = sorted(
        [
            m for m in marks
            if m.get("entry_scan_timestamp") == entry.scan_timestamp
        ],
        key=lambda x: x.get("scan_timestamp", ""),
    )

    if use_premium_stop and entry.entry_mid and entry_marks:
        for m in entry_marks:
            if m.get("premium_stop_would_exit"):
                pnl = stop_pct
                return TradeResult(
                    entry=entry,
                    exit_reason="premium_stop",
                    exit_timestamp=m.get("scan_timestamp"),
                    pnl_pct=pnl,
                    dollars=entry.entry_mid * pnl / 100,
                )

    exit_alert = find_exit_for_entry(alert_entry, day_alerts)
    if exit_alert:
        scored = oo.evaluate_intraday_alert(alert_entry, config, exit_alert=exit_alert)
        pnl = scored.get("outcome_option_pnl_pct")
        if use_premium_stop and pnl is not None and pnl < stop_pct:
            pnl = stop_pct
        em = scored.get("option_entry_mid") or entry.entry_mid
        return TradeResult(
            entry=entry,
            exit_reason="reversal_exit",
            exit_timestamp=exit_alert.get("scan_timestamp"),
            pnl_pct=pnl,
            dollars=(em or 0) * (pnl or 0) / 100,
        )

    scored = oo.evaluate_intraday_alert(alert_entry, config, exit_alert=None)
    pnl = scored.get("outcome_option_pnl_pct")
    if use_premium_stop and pnl is not None and pnl < stop_pct:
        pnl = stop_pct
    em = scored.get("option_entry_mid") or entry.entry_mid
    return TradeResult(
        entry=entry,
        exit_reason="no_exit",
        exit_timestamp=None,
        pnl_pct=pnl,
        dollars=(em or 0) * (pnl or 0) / 100 if pnl is not None else 0,
    )


def select_baseline_entries(day_alerts: list[dict]) -> list[VirtualEntry]:
    out = []
    for a in sorted(day_alerts, key=lambda x: x.get("scan_timestamp", "")):
        if a.get("alert_action", "entry") != "entry":
            continue
        if a.get("direction") not in ("call", "put"):
            continue
        out.append(VirtualEntry(
            scan_timestamp=a["scan_timestamp"],
            symbol=a["symbol"].upper(),
            direction=a["direction"].lower(),
            score=float(a.get("score") or 0),
            entry_mid=entry_mid_from_alert(a),
            source="actual_alert",
            snapshot={},
        ))
    return out


def find_snap_for_alert(alert: dict, snapshots: list[dict]) -> Optional[dict]:
    """Nearest telemetry snapshot for an alert (same symbol, same minute)."""
    sym = alert.get("symbol", "").upper()
    ts = alert.get("scan_timestamp", "")[:16]
    candidates = [
        s for s in snapshots
        if s.get("record_type") == "scan_snapshot"
        and s.get("symbol") == sym
        and (s.get("scan_timestamp") or "")[:16] == ts
    ]
    if candidates:
        return candidates[0]
    # fallback: same symbol within 5 min
    try:
        alert_dt = datetime.fromisoformat(alert["scan_timestamp"])
    except (ValueError, TypeError, KeyError):
        return None
    best = None
    best_delta = 9999.0
    for s in snapshots:
        if s.get("record_type") != "scan_snapshot" or s.get("symbol") != sym:
            continue
        try:
            sdt = datetime.fromisoformat(s["scan_timestamp"])
        except (ValueError, TypeError):
            continue
        delta = abs((sdt - alert_dt).total_seconds())
        if delta < best_delta and delta <= 600:
            best_delta = delta
            best = s
    return best


def alert_passes_anti_chase(alert: dict, snap: Optional[dict], fcfg: dict) -> bool:
    direction = alert.get("direction", "call")
    if snap:
        return not anti_chase_blocks(snap, direction, fcfg)
    from intraday_telemetry import compute_anti_chase_flags
    rsi = (alert.get("patterns") or {}).get("rsi")
    return not compute_anti_chase_flags(
        direction,
        float(alert.get("underlying_price") or 0),
        float(alert.get("vwap") or 0),
        float(rsi) if rsi is not None else None,
        fcfg,
    )["would_block"]


def alert_passes_freshness(alert: dict, snap: Optional[dict]) -> bool:
    if snap:
        direction = alert.get("direction", "call").lower()
        if direction == "call":
            return bool(snap.get("freshness_call", {}).get("fresh_or_break")
                        or snap.get("freshness_call", {}).get("fresh_vwap_cross")
                        or snap.get("freshness_call", {}).get("rsi_momentum_ok"))
        return bool(snap.get("freshness_put", {}).get("fresh_or_break")
                    or snap.get("freshness_put", {}).get("fresh_vwap_cross")
                    or snap.get("freshness_put", {}).get("rsi_momentum_ok"))
    return False


def select_from_actual_alerts(
    entries: list[VirtualEntry],
    day_alerts: list[dict],
    snapshots: list[dict],
    fcfg: dict,
    *,
    anti_chase: bool = False,
    freshness: bool = False,
    one_per_symbol: bool = False,
) -> list[VirtualEntry]:
    """Apply framework filters to alerts that actually fired (fair backtest)."""
    chosen: list[VirtualEntry] = []
    seen_symbol: set[str] = set()
    alert_by_ts = {a["scan_timestamp"]: a for a in day_alerts if a.get("alert_action", "entry") == "entry"}

    for entry in entries:
        alert = alert_by_ts.get(entry.scan_timestamp)
        if not alert:
            continue
        snap = find_snap_for_alert(alert, snapshots)
        if anti_chase and not alert_passes_anti_chase(alert, snap, fcfg):
            continue
        if freshness and not alert_passes_freshness(alert, snap):
            continue
        if one_per_symbol:
            if entry.symbol in seen_symbol:
                continue
            seen_symbol.add(entry.symbol)
        enriched = VirtualEntry(
            scan_timestamp=entry.scan_timestamp,
            symbol=entry.symbol,
            direction=entry.direction,
            score=entry.score,
            entry_mid=entry.entry_mid,
            source="actual_alert_filtered",
            snapshot=snap or {},
        )
        chosen.append(enriched)
    return chosen


def select_rescan_5m(
    snapshots: list[dict],
    day_alerts: list[dict],
    fcfg: dict,
) -> list[VirtualEntry]:
    """First qualifying snapshot per symbol — prefers scan_5m, falls back to scan_10m."""
    src_snaps = [s for s in snapshots if s.get("scan_source") == "scan_5m"]
    fallback = not src_snaps
    if fallback:
        src_snaps = [s for s in snapshots if s.get("scan_source") == "scan_10m"]
    return select_from_snapshots(
        src_snaps, day_alerts, fcfg,
        one_per_symbol=True,
        use_would_fire=False,  # counterfactual: first qualify, ignore dedup
    )


def select_from_snapshots(
    snapshots: list[dict],
    day_alerts: list[dict],
    fcfg: dict,
    *,
    anti_chase: bool = False,
    freshness: bool = False,
    one_per_symbol: bool = False,
    scan_source: Optional[str] = None,
    use_would_fire: bool = False,
) -> list[VirtualEntry]:
    snaps = sorted(snapshots, key=lambda x: x.get("scan_timestamp", ""))
    if scan_source:
        snaps = [s for s in snaps if s.get("scan_source") == scan_source]
    chosen: list[VirtualEntry] = []
    seen_symbol: set[str] = set()

    for snap in snaps:
        if snap.get("record_type") != "scan_snapshot":
            continue
        if snap.get("error"):
            continue
        sym = snap["symbol"]
        direction = snap.get("best_direction", "call")
        if use_would_fire:
            if not snap.get("would_fire_entry"):
                if not snap.get("qualifies_entry") or snap.get("dedup_blocked"):
                    continue
        elif not snap.get("qualifies_entry"):
            continue

        if anti_chase and anti_chase_blocks(snap, direction, fcfg):
            continue
        if freshness and not snap.get("freshness_best_passes"):
            continue
        if one_per_symbol:
            if sym in seen_symbol:
                continue
            seen_symbol.add(sym)

        alert = match_alert_entry(snap, day_alerts)
        chosen.append(snap_to_entry(snap, alert, "telemetry"))

    return chosen


def summarize(name: str, trades: list[TradeResult]) -> dict:
    pnls = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    wins = sum(1 for p in pnls if p > 0)
    total_usd = sum(t.dollars for t in trades)
    return {
        "name": name,
        "trades": len(trades),
        "wins": wins,
        "win_pct": round(100 * wins / len(pnls), 1) if pnls else 0,
        "avg_pnl_pct": round(sum(pnls) / len(pnls), 1) if pnls else 0,
        "total_usd": round(total_usd, 2),
        "trade_details": [
            {
                "time": t.entry.scan_timestamp[11:16],
                "symbol": t.entry.symbol,
                "direction": t.entry.direction,
                "score": t.entry.score,
                "pnl_pct": t.pnl_pct,
                "exit": t.exit_reason,
                "exit_time": (t.exit_timestamp or "")[11:16] if t.exit_timestamp else None,
            }
            for t in trades
        ],
    }


def run_backtest(target: date, config: dict) -> dict:
    day = target.isoformat()
    fcfg = telemetry_cfg(config)
    alerts = load_jsonl(ALERTS_PATH)
    telemetry = load_jsonl(telemetry_path(config))
    day_alerts = load_day_records(alerts, day)
    day_snaps = [r for r in load_day_records(telemetry, day) if r.get("record_type") == "scan_snapshot"]
    day_marks = [r for r in load_day_records(telemetry, day) if r.get("record_type") == "position_mark"]

    has_telemetry = len(day_snaps) > 0
    baseline = select_baseline_entries(day_alerts)
    scenarios: list[tuple[str, list[VirtualEntry], bool]] = []

    scenarios.append(("0_baseline_actual", baseline, False))
    scenarios.append(("3_premium_stop", baseline, True))

    if has_telemetry:
        scenarios.extend([
            ("1_anti_chase", select_from_actual_alerts(
                baseline, day_alerts, day_snaps, fcfg, anti_chase=True,
            ), False),
            ("2_momentum_freshness", select_from_actual_alerts(
                baseline, day_alerts, day_snaps, fcfg, freshness=True,
            ), False),
            ("4_one_per_symbol", select_from_actual_alerts(
                baseline, day_alerts, day_snaps, fcfg, one_per_symbol=True,
            ), False),
            ("4_one_per_symbol_stop", select_from_actual_alerts(
                baseline, day_alerts, day_snaps, fcfg, one_per_symbol=True,
            ), True),
            ("5_rescan_5m", select_rescan_5m(day_snaps, day_alerts, fcfg), False),
            ("all_f1_f2_f4", select_from_actual_alerts(
                baseline, day_alerts, day_snaps, fcfg,
                anti_chase=True, freshness=True, one_per_symbol=True,
            ), False),
            ("all_f1_f2_f4_stop", select_from_actual_alerts(
                baseline, day_alerts, day_snaps, fcfg,
                anti_chase=True, freshness=True, one_per_symbol=True,
            ), True),
        ])

    results = []
    for name, entries, prem_stop in scenarios:
        trades = [resolve_pnl(e, day_alerts, day_marks, config, use_premium_stop=prem_stop) for e in entries]
        results.append(summarize(name, trades))

    gaps = data_gap_report(day_alerts, day_snaps, day_marks)

    return {
        "date": day,
        "has_telemetry": has_telemetry,
        "snapshot_count": len(day_snaps),
        "mark_count": len(day_marks),
        "actual_entry_count": len(select_baseline_entries(day_alerts)),
        "data_gaps": gaps,
        "scenarios": results,
        "generated_at": datetime.now().isoformat(),
    }


def data_gap_report(alerts: list, snaps: list, marks: list) -> dict:
    """Document what is / isn't available for each framework."""
    entry_alerts = [a for a in alerts if a.get("alert_action", "entry") == "entry"]
    missing_contract = sum(1 for a in entry_alerts if not entry_mid_from_alert(a))
    missing_prev_rsi = sum(
        1 for s in snaps
        if s.get("prev_rsi") is None and not s.get("error")
    )
    scan_5m = sum(1 for s in snaps if s.get("scan_source") == "scan_5m")
    scan_10m = sum(1 for s in snaps if s.get("scan_source") == "scan_10m")
    return {
        "collection_status": {
            "scan_5m_snapshots": scan_5m,
            "scan_10m_snapshots": scan_10m,
            "position_marks": len(marks),
            "five_min_cron_ok": scan_5m > 0,
        },
        "framework_1_anti_chase": {
            "needs": ["rsi", "vwap_dist_pct", "per-scan anti_chase flags"],
            "from_alerts_only": "partial (fired alerts only, no blocked candidates)",
            "from_telemetry": "full" if snaps else "missing",
        },
        "framework_2_freshness": {
            "needs": ["prev_close", "prev_rsi", "freshness_* flags every scan"],
            "from_alerts_only": "requires yfinance replay (inaccurate timing)",
            "from_telemetry": "full" if snaps else "missing",
        },
        "framework_3_premium_stop": {
            "needs": ["position_mark each scan while open (option_pnl_pct)"],
            "from_alerts_only": "cap at -30% (optimistic) or yfinance proxy",
            "from_telemetry": "full" if marks else "missing marks — use cap heuristic",
            "marks_available": len(marks),
        },
        "framework_4_one_per_symbol": {
            "needs": ["would_fire_entry + dedup_blocked on all qualifying scans"],
            "from_telemetry": "full" if snaps else "derive from alerts (late fires only)",
        },
        "framework_5_rescan_5m": {
            "needs": ["scan_snapshot with scan_source=scan_5m every 5 minutes"],
            "from_telemetry": "full" if scan_5m > 0 else f"missing — only {scan_10m} scan_10m snapshots (5m cron did not run)",
        },
        "alerts_missing_contract": missing_contract,
        "snapshots_missing_prev_rsi": missing_prev_rsi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest 5 intraday frameworks for one day")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today()
    config = load_config()

    report = run_backtest(target, config)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"{target.isoformat()}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\n=== Framework backtest: {target.isoformat()} ===")
    print(f"Telemetry: {'yes' if report['has_telemetry'] else 'NO — limited accuracy'} "
          f"({report['snapshot_count']} snapshots, {report['mark_count']} marks)")
    print(f"Actual entries: {report['actual_entry_count']}\n")
    print(f"{'Scenario':<28} {'Trades':>6} {'Wins':>5} {'Win%':>6} {'Avg P&L':>9} {'Total $':>9}")
    print("-" * 72)
    for s in report["scenarios"]:
        print(f"{s['name']:<28} {s['trades']:>6} {s['wins']:>5} {s['win_pct']:>5.1f}% "
              f"{s['avg_pnl_pct']:>+8.1f}% ${s['total_usd']:>+8.2f}")
    print(f"\nReport → {out_path}")
    if not report["has_telemetry"]:
        print("\n⚠ No telemetry for this date. Deploy NAS 5-min cron + scan hook, then re-run tomorrow.")


if __name__ == "__main__":
    main()
