#!/usr/bin/env python3
"""
paper_broker.py — Alpaca PAPER execution + fill reconciliation (P2-2).

What it does
------------
Submits each surfaced alert as a *paper* option order to Alpaca, captures the
real simulated fill, and reconciles fills back to validate.py's prediction
ledger — so the scorecard can run on ground-truth entry fills instead of
synthetic close-to-close mids.

Safety
------
PAPER ONLY, enforced in code. `_assert_paper()` refuses to build a client unless
the configured endpoint is Alpaca's paper host and `alpaca.paper` is true. There
is no code path here that can reach a live-money account. `enabled` defaults to
false, so nothing is submitted until you opt in via config.json.

Config (config.json)
--------------------
  "alpaca": {
    "enabled": false,
    "paper": true,
    "api_key_id": "PK...",          // paper keys start with PK; add via config, never chat
    "api_secret_key": "...",
    "base_url": "https://paper-api.alpaca.markets",
    "qty": 1                          // contracts per alert
  }

Requires `alpaca-py` (imported lazily) only when actually submitting/syncing.

Exit policy
-----------
Entries are bought-to-open. `exit_positions()` sells-to-close each open paper
position once it has been held `alpaca.hold_trading_days` NYSE sessions (default:
the largest validation horizon, i.e. 5). Exits are market sell-to-close orders so
they fill without a live quote feed; `sync_fills()` then captures the real exit
fill and the round-trip P&L, which `validate.py` folds into the scorecard in
place of the modeled exit mid.

Intraday 0DTE
-------------
`submit_intraday()` / `exit_intraday()` mirror the intraday alert stream into the
same paper account: buy on each intraday entry alert, sell on its matching exit
alert (reversal / premium-stop / flip) and force-close at EOD so no 0DTE is held
overnight. Gated by `alpaca.intraday_enabled` (default true) on top of
`alpaca.enabled`. Keyed by the entry timestamp, so intraday round trips are
tracked separately from swing and reported via `intraday-report`.

CLI
---
  python paper_broker.py submit [--dry-run]           # swing: submit alerts.json as paper orders
  python paper_broker.py exit   [--dry-run]           # swing: sell-to-close past the hold window
  python paper_broker.py intraday-submit [--dry-run]  # intraday: buy-to-open today's entry alerts
  python paper_broker.py intraday-exit   [--dry-run]  # intraday: sell-to-close (exit alert / EOD)
  python paper_broker.py intraday-report              # intraday: today's closed round trips
  python paper_broker.py sync                          # refresh entry+exit fills from Alpaca
  python paper_broker.py status                        # show tracked paper orders + fills
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from market_calendar import is_trading_day

BASE_DIR      = Path.home() / "trading"
DATA_DIR      = BASE_DIR / "data"
ALERTS_PATH   = DATA_DIR / "alerts.json"
INTRADAY_ALERTS_PATH = DATA_DIR / "intraday_0dte_alerts.jsonl"
CONFIG_PATH   = BASE_DIR / "config.json"
PAPER_DIR     = DATA_DIR / "paper"
ORDERS_PATH   = PAPER_DIR / "paper_orders.jsonl"
FILLS_PATH    = PAPER_DIR / "paper_fills.json"
LOG_DIR       = BASE_DIR / "logs"

PAPER_HOST = "paper-api.alpaca.markets"
ET = ZoneInfo("America/New_York")
DEFAULT_EOD_EXIT_TIME = "15:45"          # matches intraday_0dte.DEFAULT_EOD_EXIT_TIME


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("paper_broker")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


logger = _setup_logging()


# ─── Config + gating ────────────────────────────────────────────────────────────
def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def alpaca_cfg(config: Optional[dict] = None) -> dict:
    return (config or load_config()).get("alpaca", {})


def is_enabled(config: Optional[dict] = None) -> bool:
    return bool(alpaca_cfg(config).get("enabled"))


def intraday_enabled(config: Optional[dict] = None) -> bool:
    """Intraday 0DTE paper execution — gated by alpaca.enabled AND an opt-out
    sub-flag (default on) so swing paper trading can run without intraday."""
    cfg = alpaca_cfg(config)
    return bool(cfg.get("enabled")) and bool(cfg.get("intraday_enabled", True))


def _assert_paper(cfg: dict) -> None:
    """Hard guard: refuse anything but Alpaca's paper endpoint."""
    base = str(cfg.get("base_url", "")).lower()
    if not cfg.get("paper", True) or PAPER_HOST not in base:
        raise RuntimeError(
            "paper_broker refuses to run against a non-paper endpoint. "
            f"Set alpaca.paper=true and base_url to https://{PAPER_HOST}."
        )


# ─── OCC option symbol ──────────────────────────────────────────────────────────
def build_occ_symbol(underlying: str, expiration: str, direction: str, strike: float) -> str:
    """
    Build an OCC-21 option symbol, e.g. SPY + 2026-07-31 + call + 750
    → 'SPY260731C00750000'.

    Format: <root><YYMMDD><C|P><strike*1000 zero-padded to 8 digits>.
    """
    d = date.fromisoformat(str(expiration))
    cp = "C" if direction.lower() == "call" else "P"
    strike_int = int(round(float(strike) * 1000))
    return f"{underlying.upper()}{d:%y%m%d}{cp}{strike_int:08d}"


def _pick_tier(alert: dict) -> Optional[dict]:
    rc = alert.get("recommended_contract") or {}
    tiers = rc.get("tiers") or {}
    for key in ("atm", "slight_otm", "affordable"):
        t = tiers.get(key)
        if t and float(t.get("mid_price") or 0) > 0:
            return {**t, "_tier": key}
    return None


def prediction_id(symbol: str, direction: str, scan_date: str) -> str:
    """Match validate.py's key so fills reconcile to the ledger."""
    return f"{symbol.upper()}:{direction.lower()}:{scan_date}"


# ─── Alert → order request (pure) ───────────────────────────────────────────────
def alert_to_order(
    alert: dict,
    scan_date: Optional[str] = None,
    qty: int = 1,
    limit: bool = True,
) -> Optional[dict]:
    """
    Map a surfaced alert to a broker-agnostic order request dict, or None if the
    alert has no priced contract to trade.

    We buy-to-open the contract. A limit at the ask models a realistic fill
    without chasing; market is available for immediacy.
    """
    tier = _pick_tier(alert)
    if not tier:
        return None
    sym = alert["symbol"].upper()
    direction = alert["direction"].lower()
    occ = build_occ_symbol(sym, tier["expiration"], direction, tier["strike"])
    # Buy at the ask (marketable limit) so paper fills are realistic, not optimistic.
    limit_price = round(float(tier.get("ask") or tier.get("mid_price")), 2)

    order = {
        "occ_symbol":     occ,
        "underlying":     sym,
        "direction":      direction,
        "tier":           tier["_tier"],
        "strike":         float(tier["strike"]),
        "expiration":     str(tier["expiration"]),
        "qty":            int(qty),
        "side":           "buy",
        "type":           "limit" if limit else "market",
        "limit_price":    limit_price if limit else None,
        "time_in_force":  "day",
        "entry_mid":      float(tier.get("mid_price") or 0),
        "score":          alert.get("score"),
    }
    if scan_date:
        order["prediction_id"] = prediction_id(sym, direction, scan_date)
    return order


def build_exit_order(entry_order: dict, qty: Optional[int] = None) -> dict:
    """
    Map a tracked *entry* order to a sell-to-close order request (pure).

    We sell-to-close at market so the paper fill lands without needing a live
    option quote feed — realistic (fills near the bid) and dependency-free. The
    order carries `closes_prediction_id` so sync_fills can reconcile the exit leg
    back to the same ledger prediction as the entry.
    """
    q = int(qty if qty is not None else entry_order.get("qty", 1))
    return {
        "occ_symbol":          entry_order["occ_symbol"],
        "underlying":          entry_order.get("underlying"),
        "direction":           entry_order.get("direction"),
        "strike":              entry_order.get("strike"),
        "expiration":          entry_order.get("expiration"),
        "qty":                 q,
        "side":                "sell",
        "type":                "market",
        "limit_price":         None,
        "time_in_force":       "day",
        "entry_mid":           entry_order.get("entry_mid"),
        "closes_prediction_id": entry_order.get("prediction_id"),
        "entry_order_id":      entry_order.get("order_id"),
    }


# ─── Storage ────────────────────────────────────────────────────────────────────
def load_orders() -> list[dict]:
    if not ORDERS_PATH.exists():
        return []
    out = []
    for line in ORDERS_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _submitted_occ_today() -> set[str]:
    """OCC symbols already submitted today — avoids double-submitting on reruns."""
    today = date.today().isoformat()
    return {
        o["occ_symbol"] for o in load_orders()
        if o.get("occ_symbol") and str(o.get("submitted_at", "")).startswith(today)
    }


def _append_orders(records: list[dict]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    with open(ORDERS_PATH, "a") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


# ─── Hold window / exit selection ────────────────────────────────────────────────
def _add_trading_days(start: date, n: int) -> date:
    """The session `n` NYSE trading days after `start` (matches option_outcome)."""
    d = start
    added = 0
    while added < n:
        d += timedelta(days=1)
        if is_trading_day(d):
            added += 1
    return d


def _hold_trading_days(config: Optional[dict] = None) -> int:
    """How many sessions to hold before selling to close. `alpaca.hold_trading_days`
    if set, else the largest validation horizon (default 5)."""
    config = config or load_config()
    explicit = alpaca_cfg(config).get("hold_trading_days")
    if explicit:
        return int(explicit)
    horizons = config.get("validation", {}).get("horizons_days") or [5]
    return int(max(horizons))


def _entry_date_of(order: dict) -> Optional[date]:
    """Scan/entry date for a tracked entry order — from the prediction_id suffix,
    falling back to the submit timestamp."""
    pid = str(order.get("prediction_id") or "")
    parts = pid.split(":")
    ds = parts[-1] if len(parts) == 3 else str(order.get("submitted_at", ""))[:10]
    try:
        return date.fromisoformat(ds)
    except ValueError:
        return None


def _closed_prediction_ids() -> set[str]:
    """Predictions that already have a sell-to-close order recorded — for dedup."""
    return {
        o["closes_prediction_id"] for o in load_orders()
        if str(o.get("side")) == "sell" and o.get("closes_prediction_id")
    }


def _open_positions(client) -> dict[str, int]:
    """{occ_symbol: qty} of currently-open Alpaca paper positions."""
    try:
        positions = client.get_all_positions()
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning(f"positions lookup failed: {exc}")
        return {}
    out: dict[str, int] = {}
    for p in positions:
        sym = getattr(p, "symbol", None)
        qty = getattr(p, "qty", None)
        if sym:
            try:
                out[sym] = int(float(qty or 0))
            except (TypeError, ValueError):
                out[sym] = 0
    return out


# ─── Alpaca client (lazy) ───────────────────────────────────────────────────────
def _client(cfg: dict):
    _assert_paper(cfg)
    try:
        from alpaca.trading.client import TradingClient
    except ImportError as exc:
        raise RuntimeError("alpaca-py not installed — run: pip install alpaca-py") from exc
    return TradingClient(
        api_key=cfg["api_key_id"],
        secret_key=cfg["api_secret_key"],
        paper=True,
    )


def _to_alpaca_request(order: dict):
    """Convert our order dict to an alpaca-py MarketOrderRequest/LimitOrderRequest."""
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    side = OrderSide.SELL if str(order.get("side")).lower() == "sell" else OrderSide.BUY
    common = dict(
        symbol=order["occ_symbol"],
        qty=order["qty"],
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    if order["type"] == "limit":
        return LimitOrderRequest(limit_price=order["limit_price"], **common)
    return MarketOrderRequest(**common)


# ─── Submit ─────────────────────────────────────────────────────────────────────
def submit_alerts(config: Optional[dict] = None, dry_run: bool = False) -> dict:
    """
    Submit today's alerts.json as paper option orders. Idempotent per OCC/day.
    Returns a summary dict.
    """
    config = config or load_config()
    if not is_enabled(config):
        logger.info("alpaca.enabled is false — not submitting")
        return {"enabled": False, "submitted": 0, "skipped": 0}

    cfg = alpaca_cfg(config)
    _assert_paper(cfg)

    doc = json.loads(ALERTS_PATH.read_text()) if ALERTS_PATH.exists() else {}
    alerts = doc.get("alerts", [])
    scan_date = (doc.get("scan_timestamp") or "")[:10] or date.today().isoformat()
    qty = int(cfg.get("qty", 1))

    already = _submitted_occ_today()
    client = None if dry_run else _client(cfg)
    submitted, skipped, records = 0, 0, []

    for alert in alerts:
        order = alert_to_order(alert, scan_date=scan_date, qty=qty)
        if not order:
            skipped += 1
            logger.info(f"skip {alert.get('symbol')} — no priced contract")
            continue
        if order["occ_symbol"] in already:
            skipped += 1
            logger.info(f"skip {order['occ_symbol']} — already submitted today")
            continue

        rec = {**order, "submitted_at": _now(), "status": "dry_run" if dry_run else "submitted"}
        if not dry_run:
            try:
                resp = client.submit_order(_to_alpaca_request(order))
                rec["order_id"] = str(getattr(resp, "id", "") or "")
                rec["status"] = str(getattr(resp, "status", "submitted"))
            except Exception as exc:
                rec["status"] = "error"
                rec["error"] = str(exc)
                logger.error(f"submit failed {order['occ_symbol']}: {exc}")
        records.append(rec)
        submitted += 1
        logger.info(f"{'[dry-run] ' if dry_run else ''}order {order['occ_symbol']} "
                    f"{order['qty']}x @ {order['type']} {order.get('limit_price','')}")

    if records and not dry_run:
        _append_orders(records)

    return {"enabled": True, "submitted": submitted, "skipped": skipped,
            "dry_run": dry_run, "orders": records}


# ─── Exit (sell-to-close past the hold window) ───────────────────────────────────
def exit_positions(config: Optional[dict] = None, dry_run: bool = False) -> dict:
    """
    Sell-to-close every tracked entry that has been held `hold_trading_days`
    NYSE sessions and is still open in the paper account. Idempotent: an entry
    that already has a recorded sell order is skipped.
    """
    config = config or load_config()
    if not is_enabled(config):
        logger.info("alpaca.enabled is false — not exiting")
        return {"enabled": False, "exited": 0, "skipped": 0}

    cfg = alpaca_cfg(config)
    _assert_paper(cfg)
    hold_days = _hold_trading_days(config)
    today = date.today()
    closed = _closed_prediction_ids()

    client = None if dry_run else _client(cfg)
    positions = _open_positions(client) if client is not None else {}

    exited, skipped, records = 0, 0, []
    for o in load_orders():
        if str(o.get("side", "buy")) != "buy":       # only close entry legs
            continue
        occ = o.get("occ_symbol")
        pid = o.get("prediction_id")
        if not occ:
            continue
        if pid and pid in closed:
            skipped += 1
            continue                                  # already has a sell order
        entry_date = _entry_date_of(o)
        if not entry_date or today < _add_trading_days(entry_date, hold_days):
            skipped += 1
            continue                                  # inside the hold window
        held_qty = positions.get(occ)
        if not dry_run and not held_qty:
            skipped += 1
            logger.info(f"skip exit {occ} — no open position")
            continue                                  # never filled / already gone

        qty = int(held_qty) if held_qty else int(o.get("qty", 1))
        exit_order = build_exit_order(o, qty=qty)
        rec = {**exit_order, "submitted_at": _now(),
               "status": "dry_run" if dry_run else "submitted"}
        if not dry_run:
            try:
                resp = client.submit_order(_to_alpaca_request(exit_order))
                rec["order_id"] = str(getattr(resp, "id", "") or "")
                rec["status"] = str(getattr(resp, "status", "submitted"))
            except Exception as exc:
                rec["status"] = "error"
                rec["error"] = str(exc)
                logger.error(f"exit failed {occ}: {exc}")
        records.append(rec)
        exited += 1
        if pid:
            closed.add(pid)                           # guard duplicates within this run
        logger.info(f"{'[dry-run] ' if dry_run else ''}exit {occ} {qty}x SELL market "
                    f"(held ≥ {hold_days} sessions)")

    if records and not dry_run:
        _append_orders(records)

    return {"enabled": True, "exited": exited, "skipped": skipped,
            "dry_run": dry_run, "orders": records}


# ─── Intraday 0DTE paper execution ───────────────────────────────────────────────
# The intraday scanner already emits entry alerts and exit alerts (reversal /
# premium-stop / EOD / flip) linked by `exit_for_entry_ts`. Paper execution simply
# mirrors that stream: buy-to-open on each entry alert, sell-to-close on its
# matching exit alert, and force-close anything still open at EOD (0DTE must never
# be held overnight). Keyed by the entry's scan_timestamp so re-entries on the same
# symbol/direction later in the day are tracked as separate trades.

def load_intraday_alerts() -> list[dict]:
    """All intraday 0DTE alert records (entries + exits), newest last."""
    if not INTRADAY_ALERTS_PATH.exists():
        return []
    out = []
    for line in INTRADAY_ALERTS_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def intraday_prediction_id(alert: dict) -> str:
    """Unique per intraday entry — includes the scan timestamp (not just the day),
    so multiple same-symbol re-entries don't collide."""
    return (f"{alert['symbol'].upper()}:{alert['direction'].lower()}:"
            f"{alert['scan_timestamp']}")


def intraday_entries_on(alerts: list[dict], day_str: str) -> list[dict]:
    """Entry alerts (call/put, action=entry) scanned on `day_str`."""
    return [
        a for a in alerts
        if str(a.get("scan_timestamp", "")).startswith(day_str)
        and a.get("alert_action", "entry") == "entry"
        and str(a.get("direction", "")).lower() in ("call", "put")
    ]


def intraday_order_from_alert(alert: dict, qty: int = 1) -> Optional[dict]:
    """Buy-to-open order for an intraday entry alert, or None if unpriced.
    Tagged strategy=intraday and keyed by the entry timestamp."""
    order = alert_to_order(alert, scan_date=None, qty=qty)
    if not order:
        return None
    order["strategy"] = "intraday"
    order["entry_ref_ts"] = alert["scan_timestamp"]
    order["prediction_id"] = intraday_prediction_id(alert)
    return order


def _submitted_intraday_refs() -> set[str]:
    """Entry timestamps already submitted as intraday paper orders — for dedup."""
    return {
        o["entry_ref_ts"] for o in load_orders()
        if o.get("strategy") == "intraday" and o.get("entry_ref_ts")
    }


def select_intraday_exits(
    entry_orders: list[dict],
    alerts_pool: list[dict],
    closed_pids: set[str],
    eod: bool,
) -> list[dict]:
    """Pure: intraday entry orders due to close — because a matching exit alert
    fired, or because it is past EOD (0DTE force-close). Skips already-closed."""
    due = []
    for o in entry_orders:
        if o.get("strategy") != "intraday" or str(o.get("side", "buy")) != "buy":
            continue
        pid = o.get("prediction_id")
        if pid and pid in closed_pids:
            continue
        ref = o.get("entry_ref_ts")
        has_exit_alert = any(
            r.get("alert_action") == "exit" and r.get("exit_for_entry_ts") == ref
            for r in alerts_pool
        )
        if eod or has_exit_alert:
            due.append(o)
    return due


def _is_past_eod_exit(intraday_cfg: dict, now: Optional[datetime] = None) -> bool:
    """True once the ET wall-clock is at/after the intraday EOD exit time."""
    if not intraday_cfg.get("eod_exit_enabled", True):
        return False
    now = now or datetime.now(ET)
    hhmm = str(intraday_cfg.get("eod_exit_time", DEFAULT_EOD_EXIT_TIME))
    try:
        h, m = (int(x) for x in hhmm.split(":")[:2])
    except ValueError:
        h, m = 15, 45
    return now.timetz().replace(tzinfo=None) >= dtime(h, m)


def submit_intraday(config: Optional[dict] = None, dry_run: bool = False) -> dict:
    """Buy-to-open each of today's intraday entry alerts once. Idempotent per
    entry timestamp."""
    config = config or load_config()
    if not intraday_enabled(config):
        logger.info("intraday paper disabled — not submitting")
        return {"enabled": False, "submitted": 0, "skipped": 0}

    cfg = alpaca_cfg(config)
    _assert_paper(cfg)
    qty = int(cfg.get("qty", 1))
    day_str = date.today().isoformat()
    entries = intraday_entries_on(load_intraday_alerts(), day_str)
    already = _submitted_intraday_refs()

    client = None if dry_run else _client(cfg)
    submitted, skipped, records = 0, 0, []
    for a in entries:
        ref = a.get("scan_timestamp")
        if not ref or ref in already:
            skipped += 1
            continue
        order = intraday_order_from_alert(a, qty=qty)
        if not order:
            skipped += 1
            logger.info(f"skip intraday {a.get('symbol')} — no priced contract")
            continue
        already.add(ref)
        rec = {**order, "submitted_at": _now(),
               "status": "dry_run" if dry_run else "submitted"}
        if not dry_run:
            try:
                resp = client.submit_order(_to_alpaca_request(order))
                rec["order_id"] = str(getattr(resp, "id", "") or "")
                rec["status"] = str(getattr(resp, "status", "submitted"))
            except Exception as exc:
                rec["status"] = "error"
                rec["error"] = str(exc)
                logger.error(f"intraday submit failed {order['occ_symbol']}: {exc}")
        records.append(rec)
        submitted += 1
        logger.info(f"{'[dry-run] ' if dry_run else ''}intraday order {order['occ_symbol']} "
                    f"{order['qty']}x @ {order['type']} {order.get('limit_price','')}")

    if records and not dry_run:
        _append_orders(records)
    return {"enabled": True, "submitted": submitted, "skipped": skipped,
            "dry_run": dry_run, "orders": records}


def exit_intraday(config: Optional[dict] = None, dry_run: bool = False) -> dict:
    """Sell-to-close intraday positions whose exit alert fired or that are past
    EOD. Idempotent per prediction."""
    config = config or load_config()
    if not intraday_enabled(config):
        logger.info("intraday paper disabled — not exiting")
        return {"enabled": False, "exited": 0, "skipped": 0}

    cfg = alpaca_cfg(config)
    _assert_paper(cfg)
    icfg = config.get("intraday_0dte", {})
    eod = _is_past_eod_exit(icfg)
    day_str = date.today().isoformat()
    pool = [a for a in load_intraday_alerts()
            if str(a.get("scan_timestamp", "")).startswith(day_str)]
    closed = _closed_prediction_ids()
    entries = [o for o in load_orders()
               if o.get("strategy") == "intraday" and str(o.get("side", "buy")) == "buy"]
    due = select_intraday_exits(entries, pool, closed, eod)

    client = None if dry_run else _client(cfg)
    positions = _open_positions(client) if client is not None else {}

    exited, skipped, records = 0, 0, []
    for o in due:
        occ = o["occ_symbol"]
        held_qty = positions.get(occ)
        if not dry_run and not held_qty:
            skipped += 1
            logger.info(f"skip intraday exit {occ} — no open position")
            continue
        qty = min(int(o.get("qty", 1)), int(held_qty)) if held_qty else int(o.get("qty", 1))
        exit_order = build_exit_order(o, qty=qty)
        exit_order["strategy"] = "intraday"
        rec = {**exit_order, "submitted_at": _now(),
               "status": "dry_run" if dry_run else "submitted"}
        if not dry_run:
            try:
                resp = client.submit_order(_to_alpaca_request(exit_order))
                rec["order_id"] = str(getattr(resp, "id", "") or "")
                rec["status"] = str(getattr(resp, "status", "submitted"))
            except Exception as exc:
                rec["status"] = "error"
                rec["error"] = str(exc)
                logger.error(f"intraday exit failed {occ}: {exc}")
        records.append(rec)
        exited += 1
        if o.get("prediction_id"):
            closed.add(o["prediction_id"])
        logger.info(f"{'[dry-run] ' if dry_run else ''}intraday exit {occ} {qty}x SELL "
                    f"market ({'EOD' if eod else 'exit-alert'})")

    if records and not dry_run:
        _append_orders(records)
    return {"enabled": True, "exited": exited, "skipped": skipped,
            "dry_run": dry_run, "orders": records}


# ─── Sync fills + reconcile to validator ────────────────────────────────────────
def sync_fills(config: Optional[dict] = None) -> dict:
    """Refresh entry+exit fill status/price for tracked orders and write
    paper_fills.json keyed by prediction_id for validate.py to consume."""
    config = config or load_config()
    cfg = alpaca_cfg(config)
    _assert_paper(cfg)
    client = _client(cfg)

    orders = load_orders()
    order_status: dict[str, dict] = {}
    for o in orders:
        oid = o.get("order_id")
        if not oid or oid in order_status:
            continue
        try:
            resp = client.get_order_by_id(oid)
        except Exception as exc:
            logger.warning(f"order lookup failed {oid}: {exc}")
            continue
        order_status[oid] = {
            "filled_avg_price": getattr(resp, "filled_avg_price", None),
            "status":           str(getattr(resp, "status", "")),
            "filled_at":        str(getattr(resp, "filled_at", "") or ""),
        }

    fills = reconcile(orders, order_status)
    if fills:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        FILLS_PATH.write_text(json.dumps(fills, indent=2, default=str))
    n_entry = sum(1 for f in fills.values() if f.get("entry_fill") is not None)
    n_exit = sum(1 for f in fills.values() if f.get("exit_fill") is not None)
    return {"orders": len(orders), "filled": n_entry, "exits": n_exit, "fills": fills}


def reconcile(orders: list[dict], order_status: dict[str, dict]) -> dict[str, dict]:
    """
    Pure reconciliation: given tracked orders and a {order_id: {filled_avg_price,
    status, filled_at}} map (however obtained), return {prediction_id: fill_record}
    carrying the entry leg and, once sold, the exit leg + round-trip P&L.
    Separated from the network call so it is fully testable offline.
    """
    fills: dict[str, dict] = {}
    # Entry legs (buy-to-open).
    for o in orders:
        if str(o.get("side", "buy")) != "buy":
            continue
        oid = o.get("order_id")
        pid = o.get("prediction_id")
        st = order_status.get(oid or "")
        if not (oid and pid and st):
            continue
        price = st.get("filled_avg_price")
        if price is None:
            continue
        fills[pid] = {
            "occ_symbol": o.get("occ_symbol"),
            "order_id":   oid,
            "strategy":   o.get("strategy", "swing"),
            "entry_fill": float(price),
            "entry_mid":  o.get("entry_mid"),
            "qty":        o.get("qty"),
            "slippage_vs_mid": (round(float(price) - float(o["entry_mid"]), 2)
                                if o.get("entry_mid") else None),
            "status":     st.get("status"),
            "filled_at":  st.get("filled_at"),
        }
    # Exit legs (sell-to-close) — attach to the prediction they close.
    for o in orders:
        if str(o.get("side")) != "sell":
            continue
        oid = o.get("order_id")
        pid = o.get("closes_prediction_id")
        st = order_status.get(oid or "")
        if not (oid and pid and st):
            continue
        price = st.get("filled_avg_price")
        if price is None:
            continue
        rec = fills.get(pid, {"occ_symbol": o.get("occ_symbol"),
                              "strategy": o.get("strategy", "swing")})
        rec["exit_fill"] = float(price)
        rec["exit_order_id"] = oid
        rec["exit_status"] = st.get("status")
        rec["exit_filled_at"] = st.get("filled_at")
        entry = rec.get("entry_fill")
        if entry:
            rec["round_trip_pnl_pct"] = round((float(price) - float(entry)) / float(entry) * 100, 2)
        fills[pid] = rec
    return fills


def intraday_report(config: Optional[dict] = None) -> dict:
    """Summarize today's closed intraday paper round trips from paper_fills.json."""
    fills = {}
    if FILLS_PATH.exists():
        try:
            fills = json.loads(FILLS_PATH.read_text())
        except Exception:
            fills = {}
    rows = [f for f in fills.values() if f.get("strategy") == "intraday"]
    closed = [f for f in rows if f.get("round_trip_pnl_pct") is not None]
    open_legs = [f for f in rows if f.get("entry_fill") and f.get("exit_fill") is None]
    pnl = [f["round_trip_pnl_pct"] for f in closed]
    wins = [p for p in pnl if p > 0]
    return {
        "intraday_orders": len(rows),
        "closed_round_trips": len(closed),
        "still_open": len(open_legs),
        "win_rate": round(100 * len(wins) / len(pnl), 1) if pnl else None,
        "avg_round_trip_pnl": round(sum(pnl) / len(pnl), 1) if pnl else None,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── CLI ────────────────────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Alpaca paper execution + reconciliation")
    sub = p.add_subparsers(dest="command")
    s = sub.add_parser("submit", help="Submit today's alerts as paper orders")
    s.add_argument("--dry-run", action="store_true")
    e = sub.add_parser("exit", help="Sell-to-close positions past their hold window")
    e.add_argument("--dry-run", action="store_true")
    isub = sub.add_parser("intraday-submit", help="Buy-to-open today's intraday 0DTE entry alerts")
    isub.add_argument("--dry-run", action="store_true")
    ix = sub.add_parser("intraday-exit", help="Sell-to-close intraday positions (exit alert / EOD)")
    ix.add_argument("--dry-run", action="store_true")
    sub.add_parser("intraday-report", help="Summarize today's intraday paper round trips")
    sub.add_parser("sync", help="Refresh entry+exit fills from Alpaca")
    sub.add_parser("status", help="Show tracked paper orders")
    args = p.parse_args(argv)

    if args.command == "submit":
        r = submit_alerts(dry_run=args.dry_run)
        if not r.get("enabled"):
            print("alpaca.enabled is false — add paper keys to config.json first.")
        else:
            tag = "[dry-run] " if r.get("dry_run") else ""
            print(f"{tag}submitted {r['submitted']}, skipped {r['skipped']}.")
        return 0
    if args.command == "exit":
        r = exit_positions(dry_run=args.dry_run)
        if not r.get("enabled"):
            print("alpaca.enabled is false — add paper keys to config.json first.")
        else:
            tag = "[dry-run] " if r.get("dry_run") else ""
            print(f"{tag}exited {r['exited']}, skipped {r['skipped']}.")
        return 0
    if args.command == "intraday-submit":
        r = submit_intraday(dry_run=args.dry_run)
        if not r.get("enabled"):
            print("intraday paper disabled (alpaca.enabled / alpaca.intraday_enabled).")
        else:
            tag = "[dry-run] " if r.get("dry_run") else ""
            print(f"{tag}intraday submitted {r['submitted']}, skipped {r['skipped']}.")
        return 0
    if args.command == "intraday-exit":
        r = exit_intraday(dry_run=args.dry_run)
        if not r.get("enabled"):
            print("intraday paper disabled (alpaca.enabled / alpaca.intraday_enabled).")
        else:
            tag = "[dry-run] " if r.get("dry_run") else ""
            print(f"{tag}intraday exited {r['exited']}, skipped {r['skipped']}.")
        return 0
    if args.command == "intraday-report":
        r = intraday_report()
        print(f"intraday paper: {r['closed_round_trips']} closed round trip(s), "
              f"{r['still_open']} open"
              + (f", win {r['win_rate']:.0f}%, avg {r['avg_round_trip_pnl']:+.1f}%"
                 if r['win_rate'] is not None else ""))
        return 0
    if args.command == "sync":
        r = sync_fills()
        print(f"synced {r['filled']} entry + {r['exits']} exit fill(s) "
              f"across {r['orders']} order(s).")
        return 0
    if args.command == "status":
        orders = load_orders()
        print(f"{len(orders)} tracked paper order(s):")
        for o in orders[-20:]:
            print(f"  {o.get('submitted_at','')[:16]}  {o['occ_symbol']:<22} "
                  f"{o.get('qty')}x {o.get('type')}  status={o.get('status')}")
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
