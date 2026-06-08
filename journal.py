#!/usr/bin/env python3
"""
journal.py — Trade Journal Manager (Week 5)

Append-only log of every paper/live trade. Tracks entries, exits,
P&L in R-multiples, and per-creator attribution.

Data stored in: ~/trading/data/trade_journal.jsonl

Usage:
  journal.py log [--from-alert SYMBOL]  # Log a new trade (pre-fills from alerts.json)
  journal.py close SYMBOL --exit PRICE  # Close an open trade, record P&L
  journal.py status                     # Open trades + this week's budget
  journal.py summary [--weeks N]        # Win rate, R-multiple, per-creator breakdown
  journal.py list [--open] [--closed]   # List all trades
  journal.py verify [SYMBOL ...]        # Cross-check closed trades vs yfinance prices

R-multiple:
  R = (exit_price - entry_price) / (entry_price - stop_price)
  Positive R = winner. Target R ≥ 1.5 average over ≥ 20 trades.
  If no stop recorded, stop defaults to 50% of option premium (entry * 0.5).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional
import logging

from utils import load_budget, save_budget, monday_of_week  # noqa: E402

BASE_DIR     = Path.home() / "trading"
DATA_DIR     = BASE_DIR / "data"
LOG_DIR      = BASE_DIR / "logs"
JOURNAL_PATH = DATA_DIR / "trade_journal.jsonl"
ALERTS_PATH  = DATA_DIR / "alerts.json"
BUDGET_PATH  = DATA_DIR / "budget.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Logging ──────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    fh = TimedRotatingFileHandler(
        str(LOG_DIR / "journal.log"), when="D", backupCount=14
    )
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[ch, fh],
    )
    return logging.getLogger("journal")


logger = _setup_logging()


# ─── Journal I/O ─────────────────────────────────────────────────────────────
def load_journal() -> list[dict]:
    """Load all journal entries (newest last)."""
    if not JOURNAL_PATH.exists():
        return []
    entries = []
    with open(JOURNAL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed journal line: {line[:80]}")
    return entries


def append_entry(entry: dict) -> None:
    """Append a single entry to the journal (atomic append)."""
    with open(JOURNAL_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    logger.info(f"Appended journal entry: {entry['id']}")


def update_entry(entry_id: str, updates: dict) -> bool:
    """Rewrite the journal with updates applied to the matching entry."""
    entries = load_journal()
    found = False
    for e in entries:
        if e["id"] == entry_id:
            e.update(updates)
            found = True
            break
    if not found:
        return False
    JOURNAL_PATH.write_text(
        "\n".join(json.dumps(e, default=str) for e in entries) + "\n"
    )
    logger.info(f"Updated journal entry: {entry_id}")
    return True


# ─── Budget ───────────────────────────────────────────────────────────────────
def _monday_of_week(d: date) -> date:
    return monday_of_week(d)


# ─── Trade ID ─────────────────────────────────────────────────────────────────
def _new_trade_id(symbol: str, direction: str) -> str:
    today_str = date.today().isoformat()
    existing  = [e["id"] for e in load_journal()
                 if e["id"].startswith(f"{today_str}-{symbol.upper()}-")]
    seq = len(existing) + 1
    return f"{today_str}-{symbol.upper()}-{direction.lower()}-{seq:03d}"


# ─── R-Multiple ───────────────────────────────────────────────────────────────
def calc_r_multiple(
    entry_price: float,
    exit_price: float,
    stop_price: Optional[float],
    direction: str,
) -> float:
    """
    R = (exit - entry) / risk_per_share  for calls
    R = (entry - exit) / risk_per_share  for puts
    where risk_per_share = entry - stop  (default stop = 50% of entry)
    """
    if stop_price is None:
        stop_price = entry_price * 0.50   # default: risk 50% of premium

    risk = entry_price - stop_price
    if risk <= 0:
        return 0.0

    if direction == "call":
        return (exit_price - entry_price) / risk
    else:  # put
        return (entry_price - exit_price) / risk


def classify_outcome(r: float) -> str:
    if r >= 1.5:
        return "target_hit"
    elif r >= 0:
        return "partial_win"
    elif r >= -1.0:
        return "stop_hit"
    else:
        return "full_loss"


# ─── Load Alerts ─────────────────────────────────────────────────────────────
def load_alerts() -> list[dict]:
    if not ALERTS_PATH.exists():
        return []
    with open(ALERTS_PATH) as f:
        data = json.load(f)
    return data.get("alerts", [])


def find_alert(symbol: str) -> Optional[dict]:
    alerts = load_alerts()
    symbol = symbol.upper()
    for a in alerts:
        if a["symbol"].upper() == symbol:
            return a
    return None


# ─── Subcommands ─────────────────────────────────────────────────────────────
def cmd_log(args: argparse.Namespace) -> None:
    """Log a new trade entry."""
    alert_data: dict = {}
    symbol    = getattr(args, "from_alert", None)
    if symbol:
        symbol = symbol.upper()
        alert_data = find_alert(symbol) or {}
        if not alert_data:
            print(f"  No alert found for {symbol} in alerts.json. Proceeding manually.")

    print("\n─── Log New Trade ────────────────────────────────────────────────")

    # Symbol
    if not symbol:
        symbol = input("  Symbol (e.g. NVDA): ").strip().upper()
    else:
        print(f"  Symbol: {symbol}")

    # Direction
    default_dir = alert_data.get("direction", "")
    direction_in = input(
        f"  Direction [call/put]{(' ('+default_dir+')') if default_dir else ''}: "
    ).strip().lower() or default_dir
    if direction_in not in ("call", "put"):
        print("  Invalid direction — must be 'call' or 'put'")
        sys.exit(1)
    direction = direction_in

    # Entry date
    today_str = date.today().isoformat()
    entry_date_in = input(f"  Entry date [{today_str}]: ").strip() or today_str

    # Entry price (option premium)
    entry_price_in = input("  Entry price (option premium, e.g. 2.35): ").strip()
    if not entry_price_in:
        print("  Entry price required")
        sys.exit(1)
    entry_price = float(entry_price_in)

    # Stop price
    default_stop = round(entry_price * 0.50, 2)
    stop_in = input(f"  Stop price (default 50% = {default_stop}): ").strip()
    stop_price = float(stop_in) if stop_in else default_stop

    # Target price
    default_target = round(entry_price * 2.0, 2)
    target_in = input(f"  Target price (default 2R = {default_target}): ").strip()
    target_price = float(target_in) if target_in else default_target

    # Strike
    strike_in = input("  Strike price (underlying, e.g. 185.00): ").strip()
    strike = float(strike_in) if strike_in else None

    # Expiration
    default_dte = alert_data.get("suggested_dte", "")
    exp_in = input(
        f"  Expiration date (YYYY-MM-DD){(' hint: '+str(default_dte)) if default_dte else ''}: "
    ).strip()
    expiration = exp_in or None

    # Notes
    notes_in = input("  Notes (optional): ").strip()

    trade_id = _new_trade_id(symbol, direction)
    entry: dict = {
        "id":               trade_id,
        "symbol":           symbol,
        "direction":        direction,
        "alert_score":      alert_data.get("score"),
        "alert_date":       alert_data.get("scan_date", today_str),
        "entry_date":       entry_date_in,
        "entry_price":      entry_price,
        "stop_price":       stop_price,
        "target_price":     target_price,
        "strike":           strike,
        "expiration":       expiration,
        "exit_date":        None,
        "exit_price":       None,
        "outcome":          None,
        "pnl_r":            None,
        "creator_match":    alert_data.get("supporting_creators", []),
        "scoring_method":   alert_data.get("scoring_method", "manual"),
        "notes":            notes_in,
        "logged_at":        datetime.now(timezone.utc).isoformat(),
    }
    append_entry(entry)
    budget = load_budget()
    budget["surfaced_this_week"] = budget.get("surfaced_this_week", 0) + 1
    BUDGET_PATH.write_text(json.dumps(budget, indent=2))

    print(f"\n  ✓ Logged trade: {trade_id}")
    print(f"    Entry: ${entry_price:.2f}  Stop: ${stop_price:.2f}  Target: ${target_price:.2f}")
    print(f"    Budget this week: {budget['surfaced_this_week']}/10\n")


def cmd_close(args: argparse.Namespace) -> None:
    """Close an open trade and compute P&L."""
    symbol    = args.symbol.upper()
    exit_price = float(args.exit)
    exit_date = getattr(args, "exit_date", None) or date.today().isoformat()

    entries = load_journal()
    open_trades = [
        e for e in entries
        if e["symbol"].upper() == symbol and e["exit_date"] is None
    ]
    if not open_trades:
        print(f"\n  No open trades found for {symbol}")
        open_syms = sorted({e["symbol"] for e in entries if e["exit_date"] is None})
        if open_syms:
            print(f"  Open positions: {', '.join(open_syms)}")
        print()
        sys.exit(1)

    if len(open_trades) > 1:
        print(f"\n  Multiple open {symbol} trades:")
        for i, e in enumerate(open_trades):
            print(f"    [{i}] {e['id']}  entry={e['entry_price']}  {e['entry_date']}")
        idx_in = input("  Which to close? [0]: ").strip() or "0"
        entry = open_trades[int(idx_in)]
    else:
        entry = open_trades[0]

    r = calc_r_multiple(
        entry["entry_price"], exit_price, entry.get("stop_price"), entry["direction"]
    )
    outcome = classify_outcome(r)

    updates = {
        "exit_date":  exit_date,
        "exit_price": exit_price,
        "pnl_r":      round(r, 3),
        "outcome":    outcome,
    }
    update_entry(entry["id"], updates)

    icon = "✅" if r >= 0 else "❌"
    print(f"\n  {icon} Closed: {entry['id']}")
    print(f"    Entry: ${entry['entry_price']:.2f}  →  Exit: ${exit_price:.2f}")
    print(f"    R-multiple: {r:+.2f}R  |  Outcome: {outcome}\n")


def cmd_status(args: argparse.Namespace) -> None:
    """Show open trades and weekly budget."""
    entries = load_journal()
    open_trades = [e for e in entries if e["exit_date"] is None]
    budget = load_budget()

    print(f"\n{'─'*64}")
    print(f"  TRADE JOURNAL STATUS")
    print(f"{'─'*64}")
    print(f"  Budget: {budget['surfaced_this_week']}/10 trades this week "
          f"(week of {budget['week_start']})")

    if open_trades:
        print(f"\n  Open Positions ({len(open_trades)}):")
        for e in open_trades:
            icon = "📈" if e["direction"] == "call" else "📉"
            age = (date.today() - date.fromisoformat(e["entry_date"])).days
            print(f"    {icon} {e['symbol']:<8} {e['direction'].upper():<4}  "
                  f"entry=${e['entry_price']:.2f}  stop=${e.get('stop_price') or '?'}  "
                  f"target=${e.get('target_price') or '?'}  "
                  f"({age}d open)")
    else:
        print("\n  No open positions.")

    total_closed = [e for e in entries if e["exit_date"] is not None]
    print(f"\n  Total closed trades: {len(total_closed)}")
    if total_closed:
        rs = [e["pnl_r"] for e in total_closed if e.get("pnl_r") is not None]
        if rs:
            avg_r = sum(rs) / len(rs)
            wins  = sum(1 for r in rs if r >= 0)
            print(f"  Win rate: {wins}/{len(rs)} ({100*wins//len(rs)}%)")
            print(f"  Avg R-multiple: {avg_r:+.2f}R")
    print(f"{'─'*64}\n")


def cmd_summary(args: argparse.Namespace) -> None:
    """Print win rate, R-multiple, and per-creator attribution table."""
    weeks = getattr(args, "weeks", None)
    entries = load_journal()
    closed = [e for e in entries if e.get("exit_date") is not None]

    if weeks:
        cutoff = date.today() - timedelta(weeks=weeks)
        closed = [e for e in closed
                  if date.fromisoformat(e["exit_date"]) >= cutoff]

    if not closed:
        print("\n  No closed trades yet. Use: journal.py close SYMBOL --exit PRICE\n")
        return

    rs = [e["pnl_r"] for e in closed if e.get("pnl_r") is not None]
    wins = sum(1 for r in rs if r >= 0)
    avg_r = sum(rs) / len(rs) if rs else 0

    calls = [e for e in closed if e["direction"] == "call"]
    puts  = [e for e in closed if e["direction"] == "put"]

    label = f"last {weeks} week(s)" if weeks else "all time"
    print(f"\n{'═'*64}")
    print(f"  TRADE JOURNAL SUMMARY — {label.upper()}")
    print(f"{'═'*64}")
    print(f"  Total trades:   {len(closed)}")
    print(f"  Win rate:       {wins}/{len(rs)} ({100*wins//len(rs) if rs else 0}%)")
    print(f"  Avg R-multiple: {avg_r:+.2f}R")
    _gate_check(avg_r, len(rs))

    # By direction
    print(f"\n  By Direction:")
    for label_d, grp in [("CALL", calls), ("PUT", puts)]:
        if not grp:
            continue
        grp_rs = [e["pnl_r"] for e in grp if e.get("pnl_r") is not None]
        grp_wins = sum(1 for r in grp_rs if r >= 0)
        grp_avg = sum(grp_rs) / len(grp_rs) if grp_rs else 0
        print(f"    {label_d:<6}: {len(grp)} trades  "
              f"win={grp_wins}/{len(grp_rs)}  avg={grp_avg:+.2f}R")

    # By outcome
    print(f"\n  By Outcome:")
    outcomes = {}
    for e in closed:
        o = e.get("outcome", "unknown")
        outcomes[o] = outcomes.get(o, 0) + 1
    for outcome, count in sorted(outcomes.items(), key=lambda x: -x[1]):
        print(f"    {outcome:<14}: {count}")

    # Per-creator attribution
    creator_stats: dict[str, dict] = {}
    for e in closed:
        creators = e.get("creator_match") or ["(none)"]
        r = e.get("pnl_r")
        if r is None:
            continue
        for c in creators:
            if c not in creator_stats:
                creator_stats[c] = {"count": 0, "wins": 0, "total_r": 0.0}
            creator_stats[c]["count"] += 1
            creator_stats[c]["wins"]  += 1 if r >= 0 else 0
            creator_stats[c]["total_r"] += r

    if creator_stats:
        print(f"\n  Per-Creator Attribution:")
        print(f"    {'Creator':<20} {'Trades':>6} {'Win%':>6} {'Avg R':>8}")
        print(f"    {'─'*46}")
        for creator, s in sorted(creator_stats.items(),
                                  key=lambda x: x[1]["total_r"] / max(x[1]["count"], 1),
                                  reverse=True):
            avg = s["total_r"] / s["count"]
            win_pct = 100 * s["wins"] // s["count"]
            bar = "★" * min(5, max(0, int(avg * 2 + 2)))
            print(f"    @{creator:<19} {s['count']:>6} {win_pct:>5}% {avg:>+7.2f}R  {bar}")

    print(f"\n{'═'*64}\n")


def _gate_check(avg_r: float, n_trades: int) -> None:
    """Print pipeline validation gate status."""
    print()
    if n_trades < 20:
        remaining = 20 - n_trades
        print(f"  ⏳ GATE: Need {remaining} more closed trades before R-multiple is meaningful.")
    elif avg_r >= 1.5:
        print(f"  ✅ GATE PASSED: Avg R ≥ 1.5 over {n_trades} trades → consider live trading (0.5% risk)")
    elif avg_r < 0.8:
        print(f"  🔴 PIVOT TRIGGER: Avg R < 0.8 after {n_trades} trades → revisit creator selection")
    else:
        print(f"  🟡 KEEP TRACKING: Avg R between 0.8–1.5 — need more trades to confirm edge")


def cmd_list(args: argparse.Namespace) -> None:
    """List all trades, optionally filtered."""
    entries = load_journal()
    show_open   = getattr(args, "open",   False)
    show_closed = getattr(args, "closed", False)

    if show_open:
        entries = [e for e in entries if e["exit_date"] is None]
    elif show_closed:
        entries = [e for e in entries if e["exit_date"] is not None]

    if not entries:
        print("\n  No trades found.\n")
        return

    print(f"\n  {'ID':<38}  {'Dir':<5} {'Entry':>7} {'Exit':>7} {'R':>6} {'Outcome':<14}")
    print(f"  {'─'*82}")
    for e in entries:
        r_str  = f"{e['pnl_r']:+.2f}R" if e.get("pnl_r") is not None else "—"
        ex_str = f"${e['exit_price']:.2f}" if e.get("exit_price") else "—"
        out    = e.get("outcome") or "open"
        print(f"  {e['id']:<38}  {e['direction']:<5} "
              f"${e['entry_price']:>6.2f} {ex_str:>7} {r_str:>6} {out:<14}")
    print()


def cmd_verify(args: argparse.Namespace) -> None:
    """Cross-check closed trades against yfinance price history."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed — run: pip install yfinance")
        sys.exit(1)

    entries = load_journal()
    closed = [e for e in entries if e.get("exit_date") is not None]

    symbols = [s.upper() for s in getattr(args, "symbols", [])]
    if symbols:
        closed = [e for e in closed if e["symbol"].upper() in symbols]

    if not closed:
        print("\n  No closed trades to verify.\n")
        return

    print(f"\n  Verifying {len(closed)} closed trade(s) against yfinance …\n")
    consistent = 0
    inconsistent = 0

    for e in closed:
        try:
            ticker = yf.Ticker(e["symbol"])
            hist   = ticker.history(
                start=e["entry_date"],
                end=e["exit_date"],
                interval="1d",
            )
            if hist.empty:
                print(f"  ⚠ {e['id']}: no price data available")
                continue

            entry_px = hist["Close"].iloc[0]
            exit_px  = hist["Close"].iloc[-1]
            actual_move = (exit_px - entry_px) / entry_px * 100

            expected_sign = "+" if e["direction"] == "call" else "-"
            actual_sign   = "+" if actual_move >= 0 else "-"
            match = (expected_sign == actual_sign) == (e.get("pnl_r", 0) >= 0)

            icon = "✓" if match else "⚠"
            if match:
                consistent += 1
            else:
                inconsistent += 1

            print(f"  {icon} {e['id']}")
            print(f"      Underlying move: {actual_move:+.1f}%  "
                  f"({e['symbol']} {e['entry_date']} → {e['exit_date']})")
            print(f"      Recorded R:      {e.get('pnl_r', '?'):+.2f}  Outcome: {e.get('outcome')}")
            if not match:
                print(f"      ⚠ Inconsistency: underlying moved {actual_sign} "
                      f"but recorded as {'win' if e['pnl_r'] >= 0 else 'loss'}")
            print()

        except Exception as exc:
            print(f"  ⚠ {e['id']}: yfinance error — {exc}\n")

    total = consistent + inconsistent
    if total:
        print(f"  Consistent: {consistent}/{total}  |  Inconsistent: {inconsistent}/{total}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trade Journal Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # log
    p_log = sub.add_parser("log", help="Log a new trade")
    p_log.add_argument("--from-alert", metavar="SYMBOL",
                       help="Pre-fill from latest alerts.json for this symbol")

    # close
    p_close = sub.add_parser("close", help="Close an open trade")
    p_close.add_argument("symbol", help="Symbol to close (e.g. NVDA)")
    p_close.add_argument("--exit", required=True, type=float, dest="exit",
                         help="Exit price (option premium)")
    p_close.add_argument("--exit-date", metavar="YYYY-MM-DD",
                         help="Exit date (default: today)")

    # status
    sub.add_parser("status", help="Open positions + weekly budget")

    # summary
    p_sum = sub.add_parser("summary", help="R-multiple + per-creator breakdown")
    p_sum.add_argument("--weeks", type=int, default=None,
                       help="Limit to last N weeks (default: all time)")

    # list
    p_list = sub.add_parser("list", help="List all trades")
    p_list.add_argument("--open",   action="store_true", help="Only open trades")
    p_list.add_argument("--closed", action="store_true", help="Only closed trades")

    # verify
    p_verify = sub.add_parser("verify",
                               help="Cross-check closed trades vs yfinance prices")
    p_verify.add_argument("symbols", nargs="*",
                          help="Specific symbols to verify (default: all closed)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n  Quick start:")
        print("    journal.py status                        # check open trades + budget")
        print("    journal.py log --from-alert BBCP         # log a trade from latest alert")
        print("    journal.py close BBCP --exit 4.10        # close with exit price")
        print("    journal.py summary                       # win rate + R-multiple")
        print("    journal.py list --open                   # show all open positions\n")
        return

    dispatch = {
        "log":     cmd_log,
        "close":   cmd_close,
        "status":  cmd_status,
        "summary": cmd_summary,
        "list":    cmd_list,
        "verify":  cmd_verify,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
