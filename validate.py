#!/usr/bin/env python3
"""
validate.py — Forward performance validation harness.

Purpose
-------
Independently measure whether the alerts this system surfaces actually predict
the right *direction* (and, when a contract was priced, whether the *option*
would have made money) over the coming weeks — without touching the reflect.py
amendment machinery.

It is a thin, append-only forward-test ledger that reuses the existing
outcome-evaluation code (option_outcome.py, market_calendar.py):

  1. snapshot  — read data/archive/scored-*.json (+ data/alerts.json) and add any
                 new surfaced alert to data/validation/prediction_ledger.jsonl.
                 Immutable, deduped by prediction_id = SYMBOL:DIR:SCAN_DATE.
  2. mark      — for each open prediction, compute forward outcomes at fixed
                 horizons (default 1/3/5 trading days): underlying directional
                 move (always available via yfinance) plus best-effort option
                 P&L (only when the alert captured a live entry contract mid).
                 Idempotent: written to data/validation/prediction_outcomes.json.
  3. report    — aggregate a scorecard: directional hit rate, option P&L,
                 expectancy, and breakdowns by scoring method / score bucket /
                 direction / creator, all vs a 50% coin-flip baseline.

Why underlying-move is the backbone
-----------------------------------
~25% of surfaced alerts are generated before options pricing settles (the 9:43am
scan) and carry no entry contract, and yfinance offers no *historical* option
chains — so retroactive option P&L is unrecoverable for those. Underlying close
prices are always reconstructable, so directional accuracy is the primary,
unbiased signal-quality metric; option P&L is reported as a secondary metric on
the subset where a live entry mid was captured.

Entry/exit convention
----------------------
Entry = underlying close on the scan date. Exit = underlying close N trading days
later. This measures signal quality from an end-of-signal-day entry; it is
honest and fully reconstructable. Run `mark` daily (cron) so option chains are
still live when horizons complete.

Usage
-----
  python validate.py snapshot                 # ingest newly-surfaced alerts
  python validate.py mark                      # compute/refresh forward outcomes
  python validate.py report                    # print scorecard
  python validate.py report --weeks 4          # limit to last 4 weeks of entries
  python validate.py run                        # snapshot + mark + report (cron)
  python validate.py report --json             # machine-readable scorecard
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from market_calendar import is_trading_day, last_trading_day_on_or_before
from option_outcome import (
    add_trading_days,
    fetch_swing_option_outcome,
    underlying_close_on,
)

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path.home() / "trading"
DATA_DIR       = BASE_DIR / "data"
ARCHIVE_DIR    = DATA_DIR / "archive"
ALERTS_PATH    = DATA_DIR / "alerts.json"
CONFIG_PATH    = BASE_DIR / "config.json"
VALID_DIR      = DATA_DIR / "validation"
LEDGER_PATH    = VALID_DIR / "prediction_ledger.jsonl"
OUTCOMES_PATH  = VALID_DIR / "prediction_outcomes.json"
SCORECARD_PATH = VALID_DIR / "scorecard.html"
PAPER_FILLS_PATH = DATA_DIR / "paper" / "paper_fills.json"
LOG_DIR        = BASE_DIR / "logs"

DEFAULT_HORIZONS = [1, 3, 5]          # trading days
DIRECTIONAL_MIN_MOVE = 0.0            # % move to count as a directional hit


# ─── Logging ────────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("validate")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        log.addHandler(h)
        try:
            fh = logging.FileHandler(str(LOG_DIR / "validate.log"))
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
            log.addHandler(fh)
        except Exception:
            pass
    log.setLevel(logging.INFO)
    return log


logger = _setup_logging()


# ─── Config ─────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"config.json unreadable ({exc}); using defaults")
        return {}


def get_horizons(config: dict | None = None) -> list[int]:
    cfg = (config or {}).get("validation", {})
    hz = cfg.get("horizons_days") or DEFAULT_HORIZONS
    try:
        return sorted({int(h) for h in hz if int(h) > 0})
    except Exception:
        return DEFAULT_HORIZONS


# ─── prediction_id ──────────────────────────────────────────────────────────────
def prediction_id(symbol: str, direction: str, scan_date: str) -> str:
    return f"{symbol.upper()}:{direction.lower()}:{scan_date}"


def _scan_date_of(archive: dict) -> str:
    ts = archive.get("scan_timestamp") or ""
    return ts[:10] if ts else date.today().isoformat()


def _entry_contract(alert: dict) -> Optional[dict]:
    """Return the priced tier that would be used for option P&L (ATM preferred)."""
    rc = alert.get("recommended_contract") or {}
    tiers = rc.get("tiers") or {}
    for key in ("atm", "slight_otm", "affordable"):
        t = tiers.get(key)
        if t and float(t.get("mid_price") or 0) > 0:
            return t
    return None


# ─── Ledger I/O (append-only, immutable snapshots) ──────────────────────────────
def load_ledger() -> dict[str, dict]:
    """Return {prediction_id: snapshot}. Later duplicates are ignored."""
    if not LEDGER_PATH.exists():
        return {}
    out: dict[str, dict] = {}
    for line in LEDGER_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = rec.get("prediction_id")
        if pid and pid not in out:
            out[pid] = rec
    return out


def _append_ledger(records: list[dict]) -> None:
    VALID_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def _snapshot_from_alert(alert: dict, scan_date: str, scan_ts: str, source: str) -> dict:
    tier = _entry_contract(alert)
    return {
        "prediction_id":       prediction_id(alert["symbol"], alert["direction"], scan_date),
        "symbol":              alert["symbol"].upper(),
        "direction":           alert["direction"].lower(),
        "scan_date":           scan_date,
        "scan_timestamp":      scan_ts,
        "score":               alert.get("score"),
        "scoring_method":      alert.get("scoring_method", "unknown"),
        "supporting_creators": [c.lstrip("@") for c in (alert.get("supporting_creators") or [])],
        "risk_level":          alert.get("risk_level") or alert.get("conviction"),
        "suggested_dte":       alert.get("suggested_dte"),
        # Immutable entry contract snapshot (may be None if options weren't priced)
        "entry_contract":      tier,
        # Keep the full recommended_contract so option_outcome can re-pick a tier
        "recommended_contract": alert.get("recommended_contract"),
        "source":              source,
        "recorded_at":         datetime.now(timezone.utc).isoformat(),
    }


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Ingest newly-surfaced alerts from archives (+ current alerts.json)."""
    ledger = load_ledger()
    known = set(ledger)
    new: list[dict] = []
    seen_this_run: set[str] = set()

    # Archives are the durable source: one file per scan, alerts preserved with contracts.
    archive_files = sorted(glob.glob(str(ARCHIVE_DIR / "scored-*.json")))
    for fp in archive_files:
        try:
            arch = json.loads(Path(fp).read_text())
        except Exception as exc:
            logger.warning(f"skip unreadable archive {fp}: {exc}")
            continue
        scan_date = _scan_date_of(arch)
        scan_ts = arch.get("scan_timestamp") or ""
        for alert in arch.get("alerts", []):
            if not alert.get("symbol") or alert.get("direction") in (None, "skip"):
                continue
            pid = prediction_id(alert["symbol"], alert["direction"], scan_date)
            if pid in known or pid in seen_this_run:
                continue
            seen_this_run.add(pid)
            new.append(_snapshot_from_alert(alert, scan_date, scan_ts, source="archive"))

    # Current alerts.json may hold an alert not yet archived (dry runs / edge timing).
    if ALERTS_PATH.exists():
        try:
            cur = json.loads(ALERTS_PATH.read_text())
            scan_ts = cur.get("scan_timestamp") or ""
            scan_date = scan_ts[:10] if scan_ts else date.today().isoformat()
            for alert in cur.get("alerts", []):
                if not alert.get("symbol") or alert.get("direction") in (None, "skip"):
                    continue
                pid = prediction_id(alert["symbol"], alert["direction"], scan_date)
                if pid in known or pid in seen_this_run:
                    continue
                seen_this_run.add(pid)
                new.append(_snapshot_from_alert(alert, scan_date, scan_ts, source="alerts"))
        except Exception as exc:
            logger.warning(f"alerts.json ingest skipped: {exc}")

    if new:
        _append_ledger(new)
    withc = sum(1 for r in new if r.get("entry_contract"))
    logger.info(
        f"snapshot: +{len(new)} new prediction(s) "
        f"({withc} with entry contract), ledger total {len(ledger) + len(new)}"
    )
    print(f"snapshot: added {len(new)} new prediction(s); ledger now holds "
          f"{len(ledger) + len(new)}.")
    return 0


# ─── Outcomes I/O (idempotent) ──────────────────────────────────────────────────
def load_outcomes() -> dict[str, dict]:
    if not OUTCOMES_PATH.exists():
        return {}
    try:
        return json.loads(OUTCOMES_PATH.read_text())
    except Exception:
        return {}


def _save_outcomes(outcomes: dict[str, dict]) -> None:
    VALID_DIR.mkdir(parents=True, exist_ok=True)
    OUTCOMES_PATH.write_text(json.dumps(outcomes, indent=2, default=str))


def _directional_correct(direction: str, move_pct: Optional[float]) -> Optional[bool]:
    if move_pct is None:
        return None
    if direction == "call":
        return move_pct > DIRECTIONAL_MIN_MOVE
    if direction == "put":
        return move_pct < -DIRECTIONAL_MIN_MOVE
    return None


def load_paper_fills() -> dict[str, dict]:
    """Real Alpaca paper fills keyed by prediction_id (from paper_broker sync)."""
    if not PAPER_FILLS_PATH.exists():
        return {}
    try:
        return json.loads(PAPER_FILLS_PATH.read_text())
    except Exception:
        return {}


def _apply_fill_override(recommended_contract: Optional[dict], fill_price: float) -> dict:
    """Return a copy of recommended_contract with the priced tier option_outcome
    would pick set to the real paper fill price (so option P&L uses the fill, not
    the assumed mid)."""
    rc = copy.deepcopy(recommended_contract or {})
    tiers = rc.get("tiers") or {}
    for key in ("atm", "slight_otm", "affordable"):
        t = tiers.get(key)
        if t and float(t.get("mid_price") or 0) > 0:
            t["mid_price"] = float(fill_price)
            break
    return rc


def _mark_one(snap: dict, horizons: list[int], config: dict,
              paper_fills: Optional[dict] = None) -> dict:
    """Compute per-horizon outcomes for one prediction. Reconstructable + best-effort."""
    symbol    = snap["symbol"]
    direction = snap["direction"]
    entry_date = date.fromisoformat(snap["scan_date"])
    today = date.today()
    as_of = last_trading_day_on_or_before(today)

    entry_close = underlying_close_on(symbol, entry_date)

    per_h: dict[str, dict] = {}
    all_final = True
    has_contract = bool(snap.get("entry_contract"))

    # Prefer a real Alpaca paper fill over the synthetic mid when one exists.
    fill = (paper_fills or {}).get(snap["prediction_id"])
    entry_source = "mid"
    fill_slippage_pct: Optional[float] = None
    recommended = snap.get("recommended_contract")
    if fill and fill.get("entry_fill") and has_contract:
        recommended = _apply_fill_override(recommended, float(fill["entry_fill"]))
        entry_source = "paper_fill"
        mid = fill.get("entry_mid")
        if mid:
            fill_slippage_pct = round((float(fill["entry_fill"]) - float(mid)) / float(mid) * 100, 2)

    # A real sell-to-close fill closes the round trip: option P&L becomes the
    # realized entry_fill → exit_fill, not a modeled exit mid. Applied to the
    # longest horizon (the held-to-exit bucket).
    realized_pnl: Optional[float] = None
    exit_fill_price: Optional[float] = None
    if (fill and has_contract and fill.get("entry_fill") and fill.get("exit_fill")):
        entry_fill_price = float(fill["entry_fill"])
        exit_fill_price = float(fill["exit_fill"])
        if entry_fill_price > 0:
            realized_pnl = round((exit_fill_price - entry_fill_price) / entry_fill_price * 100, 2)

    # Reconstruct an alert-like dict for option_outcome (uses recommended_contract.tiers)
    alert_like = {
        "symbol": symbol,
        "direction": direction,
        "recommended_contract": recommended,
    }

    for h in horizons:
        target_date = add_trading_days(entry_date, h)
        # Only lock a horizon as final once its target session is strictly in the
        # past — a target landing on *today* would price against an incomplete bar.
        final = target_date < today
        if not final:
            all_final = False
        price_date = min(target_date, as_of)
        exit_close = underlying_close_on(symbol, price_date)

        move_pct: Optional[float] = None
        if entry_close and entry_close > 0 and exit_close is not None:
            move_pct = round((exit_close - entry_close) / entry_close * 100, 2)

        row: dict = {
            "horizon_days":       h,
            "target_date":        target_date.isoformat(),
            "priced_as_of":       price_date.isoformat(),
            "final":              final,
            "entry_close":        round(entry_close, 4) if entry_close else None,
            "exit_close":         round(exit_close, 4) if exit_close is not None else None,
            "underlying_move_pct": move_pct,
            "directional_correct": _directional_correct(direction, move_pct),
            "option_pnl_pct":     None,
        }

        # Best-effort option P&L only when a live entry mid was captured.
        if has_contract:
            try:
                res = fetch_swing_option_outcome(
                    alert_like, entry_date, hold_days=h, as_of=price_date,
                )
                if res:
                    row["option_pnl_pct"] = res.get("outcome_option_pnl_pct")
                    row["option_entry_mid"] = res.get("entry_mid")
                    row["option_exit_mid"] = res.get("exit_mid")
            except Exception as exc:  # pragma: no cover - network/data dependent
                logger.debug(f"option P&L failed {symbol} h{h}: {exc}")

        # Whether option P&L used a real paper fill or the synthetic mid.
        row["entry_source"] = entry_source
        if entry_source == "paper_fill":
            row["paper_fill_price"] = float(fill["entry_fill"])
            row["fill_vs_mid_slippage_pct"] = fill_slippage_pct

        per_h[f"h{h}"] = row

    # Overlay the realized round-trip P&L on the held-to-exit (longest) horizon.
    exit_source: Optional[str] = None
    if realized_pnl is not None and horizons:
        hkey = f"h{max(horizons)}"
        exit_row = per_h.get(hkey)
        if exit_row is not None:
            exit_row["option_pnl_pct"] = realized_pnl
            exit_row["option_pnl_source"] = "round_trip_paper_fill"
            exit_row["exit_source"] = "paper_fill"
            exit_row["paper_exit_price"] = exit_fill_price
            exit_source = "paper_fill"

    return {
        "prediction_id": snap["prediction_id"],
        "symbol":        symbol,
        "direction":     direction,
        "scan_date":     snap["scan_date"],
        "has_contract":  has_contract,
        "horizons":      per_h,
        "all_final":     all_final,
        "exit_source":   exit_source,
        "round_trip_pnl_pct": realized_pnl,
        "marked_at":     datetime.now(timezone.utc).isoformat(),
    }


def cmd_mark(args: argparse.Namespace) -> int:
    """Compute/refresh forward outcomes for every ledger prediction."""
    config   = load_config()
    horizons = get_horizons(config)
    ledger   = load_ledger()
    outcomes = load_outcomes()
    fills    = load_paper_fills()

    if not ledger:
        print("mark: ledger is empty — run `validate.py snapshot` first.")
        return 0

    updated = 0
    skipped_final = 0
    for pid, snap in ledger.items():
        prev = outcomes.get(pid)
        prev_used_fill = bool(prev and any(
            h.get("entry_source") == "paper_fill"
            for h in (prev.get("horizons") or {}).values()
        ))
        prev_used_exit = bool(prev and prev.get("exit_source") == "paper_fill")
        new_has_exit = bool(fills.get(pid, {}).get("exit_fill"))
        # A newly-arrived paper fill re-opens an otherwise-final prediction once,
        # so its option P&L can switch from the assumed mid to the real fill —
        # once for the entry leg, and again when the exit (sell) fill lands.
        needs_fill_update = (pid in fills) and (
            not prev_used_fill or (new_has_exit and not prev_used_exit)
        )
        if prev and prev.get("all_final") and not args.force and not needs_fill_update:
            skipped_final += 1
            continue
        outcomes[pid] = _mark_one(snap, horizons, config, paper_fills=fills)
        updated += 1

    _save_outcomes(outcomes)
    n_with_fills = sum(
        1 for oc in outcomes.values()
        if any(h.get("entry_source") == "paper_fill" for h in oc.get("horizons", {}).values())
    )
    logger.info(f"mark: updated {updated}, already-final {skipped_final}, "
                f"paper-fill entries={n_with_fills}, horizons={horizons}")
    print(f"mark: updated {updated} prediction(s) "
          f"({skipped_final} already final; {n_with_fills} using real paper fills) "
          f"at horizons {horizons} trading days.")
    return 0


# ─── Reporting ──────────────────────────────────────────────────────────────────
def _score_bucket(score: Optional[float]) -> str:
    if score is None:
        return "n/a"
    if score >= 0.8:
        return "0.80+"
    if score >= 0.7:
        return "0.70-0.79"
    if score >= 0.6:
        return "0.60-0.69"
    return "<0.60"


def _agg(rows: list[dict]) -> dict:
    """Aggregate a list of per-horizon outcome rows into summary stats."""
    dir_known = [r for r in rows if r.get("directional_correct") is not None]
    hits = [r for r in dir_known if r["directional_correct"]]
    moves = [r["underlying_move_pct"] for r in dir_known if r.get("underlying_move_pct") is not None]
    opt = [r["option_pnl_pct"] for r in rows if r.get("option_pnl_pct") is not None]
    opt_wins = [p for p in opt if p > 0]

    # "Captured" move = move in the predicted direction (positive when correct).
    captured = []
    for r in dir_known:
        m = r.get("underlying_move_pct")
        if m is None:
            continue
        captured.append(m if r["direction"] == "call" else -m)

    # Real paper fills vs assumed mids
    paper = [r for r in rows if r.get("entry_source") == "paper_fill"]
    slip = [r["fill_vs_mid_slippage_pct"] for r in paper
            if r.get("fill_vs_mid_slippage_pct") is not None]

    # Closed round trips (real entry + real exit fill)
    exited = [r for r in rows if r.get("exit_source") == "paper_fill"]
    rt = [r["option_pnl_pct"] for r in exited if r.get("option_pnl_pct") is not None]
    rt_wins = [p for p in rt if p > 0]

    return {
        "n":                 len(rows),
        "n_directional":     len(dir_known),
        "dir_hit_rate":      round(100 * len(hits) / len(dir_known), 1) if dir_known else None,
        "avg_abs_move":      round(sum(abs(x) for x in moves) / len(moves), 2) if moves else None,
        "avg_captured_move": round(sum(captured) / len(captured), 2) if captured else None,
        "n_option":          len(opt),
        "option_win_rate":   round(100 * len(opt_wins) / len(opt), 1) if opt else None,
        "avg_option_pnl":    round(sum(opt) / len(opt), 1) if opt else None,
        "n_paper_fill":      len(paper),
        "avg_fill_slippage": round(sum(slip) / len(slip), 2) if slip else None,
        "n_round_trip":      len(exited),
        "round_trip_win_rate": round(100 * len(rt_wins) / len(rt), 1) if rt else None,
        "avg_round_trip_pnl": round(sum(rt) / len(rt), 1) if rt else None,
    }


def _collect_rows(
    ledger: dict[str, dict],
    outcomes: dict[str, dict],
    horizon: int,
    since: Optional[date],
    final_only: bool,
) -> list[dict]:
    rows: list[dict] = []
    for pid, snap in ledger.items():
        if since and date.fromisoformat(snap["scan_date"]) < since:
            continue
        oc = outcomes.get(pid)
        if not oc:
            continue
        h = oc.get("horizons", {}).get(f"h{horizon}")
        if not h:
            continue
        if final_only and not h.get("final"):
            continue
        rows.append({
            **h,
            "direction":      snap["direction"],
            "score":          snap.get("score"),
            "scoring_method": snap.get("scoring_method", "unknown"),
            "creators":       snap.get("supporting_creators") or [],
        })
    return rows


def build_scorecard(
    weeks: Optional[int] = None,
    final_only: bool = True,
    config: dict | None = None,
) -> dict:
    config   = config or load_config()
    horizons = get_horizons(config)
    ledger   = load_ledger()
    outcomes = load_outcomes()

    since = None
    if weeks:
        since = date.fromordinal(date.today().toordinal() - weeks * 7)

    scorecard: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizons_days": horizons,
        "final_only": final_only,
        "since": since.isoformat() if since else None,
        "ledger_total": len(ledger),
        "marked_total": len(outcomes),
        "by_horizon": {},
    }

    for h in horizons:
        rows = _collect_rows(ledger, outcomes, h, since, final_only)
        overall = _agg(rows)

        by_method: dict[str, dict] = {}
        for m in sorted({r["scoring_method"] for r in rows}):
            by_method[m] = _agg([r for r in rows if r["scoring_method"] == m])

        by_bucket: dict[str, dict] = {}
        for b in ("0.80+", "0.70-0.79", "0.60-0.69", "<0.60", "n/a"):
            sub = [r for r in rows if _score_bucket(r.get("score")) == b]
            if sub:
                by_bucket[b] = _agg(sub)

        by_direction: dict[str, dict] = {}
        for d in ("call", "put"):
            sub = [r for r in rows if r["direction"] == d]
            if sub:
                by_direction[d] = _agg(sub)

        by_creator: dict[str, dict] = {}
        creators = sorted({c for r in rows for c in r["creators"]})
        for c in creators:
            sub = [r for r in rows if c in r["creators"]]
            if sub:
                by_creator[c] = _agg(sub)

        scorecard["by_horizon"][f"h{h}"] = {
            "overall":      overall,
            "by_method":    by_method,
            "by_score":     by_bucket,
            "by_direction": by_direction,
            "by_creator":   by_creator,
        }

    return scorecard


def _fmt_agg(a: dict) -> str:
    dhr = f"{a['dir_hit_rate']:.0f}%" if a["dir_hit_rate"] is not None else " n/a"
    cap = f"{a['avg_captured_move']:+.2f}%" if a["avg_captured_move"] is not None else "  n/a"
    owr = f"{a['option_win_rate']:.0f}%" if a["option_win_rate"] is not None else " n/a"
    opl = f"{a['avg_option_pnl']:+.0f}%" if a["avg_option_pnl"] is not None else "  n/a"
    return (f"n={a['n']:<3} dir_hit={dhr:>4} capt_move={cap:>7} "
            f"| opt_n={a['n_option']:<3} opt_win={owr:>4} opt_pnl={opl:>6}")


# ─── HTML scorecard (email-friendly, self-contained, light theme) ───────────────
def _pct(x: Optional[float], suffix: str = "%") -> str:
    return f"{x:.0f}{suffix}" if x is not None else "—"


def _signed(x: Optional[float]) -> str:
    return f"{x:+.2f}%" if x is not None else "—"


def _edge_color(hit: Optional[float]) -> str:
    """Green above coin-flip, red below, grey when unknown."""
    if hit is None:
        return "#8695a4"
    if hit >= 55:
        return "#1f9d55"
    if hit <= 45:
        return "#d1495b"
    return "#bd7a10"


def _agg_row_html(label: str, a: dict, is_header: bool = False) -> str:
    hit = a.get("dir_hit_rate")
    cell = "th" if is_header else "td"
    bg = "background:#f4f6f9;" if is_header else ""
    color = _edge_color(hit) if not is_header else "#16202b"
    hitcell = (f"<{cell} style='text-align:right;font-weight:600;color:{color};"
               f"font-variant-numeric:tabular-nums;padding:6px 10px;'>{_pct(hit)}</{cell}>")
    return (
        f"<tr style='{bg}border-bottom:1px solid #e6ebf0;'>"
        f"<{cell} style='text-align:left;padding:6px 10px;color:#16202b;'>{label}</{cell}>"
        f"<{cell} style='text-align:right;padding:6px 10px;color:#566472;font-variant-numeric:tabular-nums;'>{a.get('n', 0)}</{cell}>"
        f"{hitcell}"
        f"<{cell} style='text-align:right;padding:6px 10px;color:#566472;font-variant-numeric:tabular-nums;'>{_signed(a.get('avg_captured_move'))}</{cell}>"
        f"<{cell} style='text-align:right;padding:6px 10px;color:#566472;font-variant-numeric:tabular-nums;'>{a.get('n_option', 0)}</{cell}>"
        f"<{cell} style='text-align:right;padding:6px 10px;color:#566472;font-variant-numeric:tabular-nums;'>{_pct(a.get('option_win_rate'))}</{cell}>"
        f"<{cell} style='text-align:right;padding:6px 10px;color:#566472;font-variant-numeric:tabular-nums;'>{_pct(a.get('avg_option_pnl'))}</{cell}>"
        f"</tr>"
    )


def _breakdown_table(title: str, rows: dict[str, dict]) -> str:
    if not rows:
        return ""
    body = "".join(_agg_row_html(k, v) for k, v in rows.items())
    return (
        f"<div style='margin-top:14px;'>"
        f"<div style='font:600 11px/1 -apple-system,Segoe UI,sans-serif;text-transform:uppercase;"
        f"letter-spacing:.08em;color:#8695a4;margin-bottom:6px;'>{title}</div>"
        f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0' "
        f"style='border-collapse:collapse;font:13px -apple-system,Segoe UI,sans-serif;'>"
        f"{_agg_head()}{body}</table></div>"
    )


def _agg_head() -> str:
    cols = ["", "n", "dir hit", "capt move", "opt n", "opt win", "opt P&L"]
    ths = "".join(
        f"<th style='text-align:{'left' if i == 0 else 'right'};padding:4px 10px;"
        f"font:600 10px/1 -apple-system,Segoe UI,sans-serif;text-transform:uppercase;"
        f"letter-spacing:.06em;color:#8695a4;border-bottom:1px solid #dbe2ea;'>{c}</th>"
        for i, c in enumerate(cols)
    )
    return f"<tr>{ths}</tr>"


def _scorecard_html(scorecard: dict, weeks: Optional[int], include_interim: bool) -> str:
    gen = scorecard["generated_at"][:16].replace("T", " ") + " UTC"
    scope = f"last {weeks} weeks" if weeks else "all time"
    mode = "interim + final" if include_interim else "final outcomes only"

    sections = []
    for hkey, block in scorecard["by_horizon"].items():
        ov = block["overall"]
        if ov["n"] == 0:
            continue
        hit = ov.get("dir_hit_rate")
        edge = (hit - 50.0) if hit is not None else None
        edge_c = _edge_color(hit)
        hold = hkey[1:]

        # Headline strip
        stat = (
            "<table role='presentation' width='100%' cellpadding='0' cellspacing='0'>"
            "<tr>"
            f"{_statcell('Directional hit', _pct(hit), edge_c)}"
            f"{_statcell('Edge vs coin-flip', (f'{edge:+.0f} pts' if edge is not None else '—'), edge_c)}"
            f"{_statcell('Avg captured move', _signed(ov.get('avg_captured_move')), '#16202b')}"
            f"{_statcell('Option win', _pct(ov.get('option_win_rate')), '#16202b')}"
            f"{_statcell('Avg option P&L', _pct(ov.get('avg_option_pnl')), '#16202b')}"
            "</tr></table>"
        )

        breakdowns = (
            _breakdown_table("By scoring method", block["by_method"])
            + _breakdown_table("By score bucket", block["by_score"])
            + _breakdown_table("By direction", block["by_direction"])
            + _breakdown_table("By creator", block["by_creator"])
        )

        sections.append(
            f"<div style='background:#ffffff;border:1px solid #dbe2ea;border-radius:12px;"
            f"padding:18px 20px;margin-bottom:16px;'>"
            f"<div style='font:650 15px/1.2 -apple-system,Segoe UI,sans-serif;color:#16202b;'>"
            f"H{hold} · {hold}-trading-day hold "
            f"<span style='color:#8695a4;font-weight:400;'>({ov['n']} alerts)</span></div>"
            f"<div style='margin-top:14px;'>{stat}</div>"
            f"{breakdowns}</div>"
        )

    if not sections:
        sections.append(
            "<div style='background:#fff;border:1px solid #dbe2ea;border-radius:12px;"
            "padding:24px 20px;color:#566472;font:14px -apple-system,Segoe UI,sans-serif;'>"
            "No resolved outcomes yet — predictions need to age past their 1/3/5-day "
            "horizons. Keep running <code>validate.py mark</code> daily.</div>"
        )

    legend = (
        "<div style='color:#8695a4;font:12px/1.6 -apple-system,Segoe UI,sans-serif;margin-top:8px;'>"
        "<b>dir hit</b> = % of alerts where the underlying moved the predicted way "
        "(green ≥55%, red ≤45% vs a 50% coin-flip). "
        "<b>capt move</b> = avg underlying move in the predicted direction. "
        "<b>opt win / opt P&L</b> = option outcomes on the priced-contract subset only."
        "</div>"
    )

    return (
        "<div style='background:#f4f6f9;padding:24px 12px;'>"
        "<div style='max-width:640px;margin:0 auto;'>"
        "<div style='font:11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;"
        "letter-spacing:.12em;text-transform:uppercase;color:#0a6a65;font-weight:600;'>"
        "Stock Scanner · Forward Validation</div>"
        "<div style='font:700 22px/1.2 -apple-system,Segoe UI,sans-serif;color:#16202b;margin:4px 0 2px;'>"
        "Performance scorecard</div>"
        f"<div style='color:#566472;font:13px -apple-system,Segoe UI,sans-serif;margin-bottom:18px;'>"
        f"{gen} · {scope} · {mode} · "
        f"{scorecard['ledger_total']} alerts tracked, {scorecard['marked_total']} scored</div>"
        + "".join(sections)
        + legend
        + "</div></div>"
    )


def _statcell(label: str, value: str, color: str) -> str:
    return (
        "<td style='vertical-align:top;padding:0 12px 0 0;'>"
        f"<div style='font:600 10px/1 -apple-system,Segoe UI,sans-serif;text-transform:uppercase;"
        f"letter-spacing:.05em;color:#8695a4;'>{label}</div>"
        f"<div style='font:700 20px/1.3 -apple-system,Segoe UI,sans-serif;color:{color};"
        f"font-variant-numeric:tabular-nums;'>{value}</div></td>"
    )


def cmd_report(args: argparse.Namespace) -> int:
    scorecard = build_scorecard(weeks=args.weeks, final_only=not args.include_interim)

    if getattr(args, "html", False):
        out = Path(args.out) if getattr(args, "out", None) else SCORECARD_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        html = _scorecard_html(scorecard, args.weeks, args.include_interim)
        out.write_text("<!doctype html><meta charset='utf-8'>"
                       "<title>Stock Scanner — Validation Scorecard</title>" + html)
        logger.info(f"scorecard HTML written → {out}")
        print(f"HTML scorecard written → {out}")
        return 0

    if args.json:
        print(json.dumps(scorecard, indent=2, default=str))
        return 0

    print(f"\n{'═'*74}")
    print(f"  FORWARD VALIDATION SCORECARD")
    scope = f"last {args.weeks} weeks" if args.weeks else "all time"
    mode = "final outcomes only" if not args.include_interim else "incl. interim"
    print(f"  scope: {scope}  │  {mode}  │  ledger={scorecard['ledger_total']}  "
          f"marked={scorecard['marked_total']}")
    print(f"{'═'*74}")

    any_data = False
    for hkey, block in scorecard["by_horizon"].items():
        ov = block["overall"]
        if ov["n"] == 0:
            continue
        any_data = True
        print(f"\n  ── {hkey.upper()} ({hkey[1:]} trading day hold) "
              f"{'─'*(74-len(hkey)-24)}")
        print(f"     OVERALL   {_fmt_agg(ov)}")
        base = "  (baseline coin-flip = 50%)"
        if ov["dir_hit_rate"] is not None:
            edge = ov["dir_hit_rate"] - 50.0
            print(f"     directional edge vs coin-flip: {edge:+.1f} pts{base}")
        if ov.get("n_paper_fill"):
            slip = ov.get("avg_fill_slippage")
            slip_s = f", avg fill slippage {slip:+.1f}% vs mid" if slip is not None else ""
            print(f"     option P&L on REAL paper fills: {ov['n_paper_fill']}/{ov['n_option']}"
                  f"{slip_s}")
        if ov.get("n_round_trip"):
            wr = ov.get("round_trip_win_rate")
            avg = ov.get("avg_round_trip_pnl")
            wr_s = f", win {wr:.0f}%" if wr is not None else ""
            avg_s = f", avg {avg:+.1f}%" if avg is not None else ""
            print(f"     CLOSED round trips (real entry+exit fills): "
                  f"{ov['n_round_trip']}{wr_s}{avg_s}")

        if len(block["by_method"]) > 1 or block["by_method"]:
            print("     by scoring method:")
            for m, a in block["by_method"].items():
                print(f"       {m:<10} {_fmt_agg(a)}")
        if block["by_score"]:
            print("     by score bucket:")
            for b, a in block["by_score"].items():
                print(f"       {b:<10} {_fmt_agg(a)}")
        if block["by_direction"]:
            print("     by direction:")
            for d, a in block["by_direction"].items():
                print(f"       {d:<10} {_fmt_agg(a)}")
        if block["by_creator"]:
            print("     by creator:")
            for c, a in block["by_creator"].items():
                print(f"       @{c:<9} {_fmt_agg(a)}")

    if not any_data:
        print("\n  No resolved outcomes yet. Predictions need to age past their")
        print("  horizon (1/3/5 trading days). Run `validate.py mark` daily.\n")
    else:
        print(f"\n{'─'*74}")
        print("  dir_hit    = % of alerts where the underlying moved the predicted way")
        print("  capt_move  = avg underlying move IN the predicted direction (edge, signed)")
        print("  opt_win    = % of priced-contract alerts with option P&L > 0 (subset)")
        print(f"{'─'*74}\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """snapshot + mark + report — the cron entry point."""
    cmd_snapshot(args)
    cmd_mark(args)
    return cmd_report(args)


# ─── CLI ────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Forward performance validation harness")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("snapshot", help="Ingest newly-surfaced alerts into the ledger")

    m = sub.add_parser("mark", help="Compute/refresh forward outcomes")
    m.add_argument("--force", action="store_true", help="Re-mark even already-final predictions")

    for name in ("report", "run"):
        r = sub.add_parser(name, help="Print the performance scorecard"
                           if name == "report" else "snapshot + mark + report (cron)")
        r.add_argument("--weeks", type=int, default=None, help="Limit to last N weeks of entries")
        r.add_argument("--include-interim", action="store_true",
                       help="Include not-yet-final horizons")
        r.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
        r.add_argument("--html", action="store_true",
                       help="Write a self-contained HTML scorecard (email-ready)")
        r.add_argument("--out", type=str, default=None,
                       help="Output path for --html (default: data/validation/scorecard.html)")
        if name == "run":
            r.add_argument("--force", action="store_true",
                           help="Re-mark even already-final predictions")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.command:
        build_parser().print_help()
        return 1
    dispatch = {
        "snapshot": cmd_snapshot,
        "mark":     cmd_mark,
        "report":   cmd_report,
        "run":      cmd_run,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
