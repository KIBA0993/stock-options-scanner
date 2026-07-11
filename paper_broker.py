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

CLI
---
  python paper_broker.py submit [--dry-run]   # submit today's alerts.json as paper orders
  python paper_broker.py sync                  # refresh fill status from Alpaca
  python paper_broker.py status                # show tracked paper orders + fills
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR      = Path.home() / "trading"
DATA_DIR      = BASE_DIR / "data"
ALERTS_PATH   = DATA_DIR / "alerts.json"
CONFIG_PATH   = BASE_DIR / "config.json"
PAPER_DIR     = DATA_DIR / "paper"
ORDERS_PATH   = PAPER_DIR / "paper_orders.jsonl"
FILLS_PATH    = PAPER_DIR / "paper_fills.json"
LOG_DIR       = BASE_DIR / "logs"

PAPER_HOST = "paper-api.alpaca.markets"


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
    common = dict(
        symbol=order["occ_symbol"],
        qty=order["qty"],
        side=OrderSide.BUY,
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


# ─── Sync fills + reconcile to validator ────────────────────────────────────────
def sync_fills(config: Optional[dict] = None) -> dict:
    """Refresh fill status/price for tracked orders and write paper_fills.json
    keyed by prediction_id for validate.py to consume."""
    config = config or load_config()
    cfg = alpaca_cfg(config)
    _assert_paper(cfg)
    client = _client(cfg)

    fills: dict[str, dict] = {}
    updated = 0
    for o in load_orders():
        oid = o.get("order_id")
        pid = o.get("prediction_id")
        if not oid:
            continue
        try:
            resp = client.get_order_by_id(oid)
        except Exception as exc:
            logger.warning(f"order lookup failed {oid}: {exc}")
            continue
        fill_price = getattr(resp, "filled_avg_price", None)
        status = str(getattr(resp, "status", ""))
        if fill_price and pid:
            fills[pid] = {
                "occ_symbol":  o["occ_symbol"],
                "order_id":    oid,
                "entry_fill":  float(fill_price),
                "entry_mid":   o.get("entry_mid"),
                "qty":         o.get("qty"),
                "status":      status,
                "filled_at":   str(getattr(resp, "filled_at", "") or ""),
            }
            updated += 1

    if fills:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        FILLS_PATH.write_text(json.dumps(fills, indent=2, default=str))
    return {"orders": len(load_orders()), "filled": updated, "fills": fills}


def reconcile(orders: list[dict], order_status: dict[str, dict]) -> dict[str, dict]:
    """
    Pure reconciliation: given tracked orders and a {order_id: {filled_avg_price,
    status, filled_at}} map (however obtained), return {prediction_id: fill_record}.
    Separated from the network call so it is fully testable offline.
    """
    fills: dict[str, dict] = {}
    for o in orders:
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
            "entry_fill": float(price),
            "entry_mid":  o.get("entry_mid"),
            "slippage_vs_mid": (round(float(price) - float(o["entry_mid"]), 2)
                                if o.get("entry_mid") else None),
            "status":     st.get("status"),
            "filled_at":  st.get("filled_at"),
        }
    return fills


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── CLI ────────────────────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Alpaca paper execution + reconciliation")
    sub = p.add_subparsers(dest="command")
    s = sub.add_parser("submit", help="Submit today's alerts as paper orders")
    s.add_argument("--dry-run", action="store_true")
    sub.add_parser("sync", help="Refresh fills from Alpaca")
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
    if args.command == "sync":
        r = sync_fills()
        print(f"synced {r['filled']}/{r['orders']} order(s) with fills.")
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
