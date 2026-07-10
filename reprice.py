#!/usr/bin/env python3
"""
reprice.py — Backfill option contracts for alerts surfaced before pricing settled.

Why this exists
---------------
The morning swing scan runs ~9:43 AM ET, before option quotes are live. So a
chunk of alerts get written to alerts.json (and archived) with EMPTY contract
tiers — no strike, no entry mid. That's unrecoverable later (yfinance has no
historical option chains), which blanks out option-P&L for validate.py and
leaves notify.py with no contract card.

This module re-fetches live option contracts for exactly those alerts a few
minutes after the open, and writes them back into:
  1. data/alerts.json                — so notify.py sends a real contract card
  2. data/archive/scored-*.json      — so validate.py snapshots a priced entry

It only touches alerts that lack a priced tier, and it's idempotent: once an
alert has a contract, re-running skips it.

Ordering note
-------------
Run this AFTER the scan and BEFORE the daily validate.py snapshot (e.g. cron at
9:52 and 10:02), so the archive carries contracts by the time the validator
ingests it. Repricing an archive the validator has already snapshotted won't
retro-fill the ledger (snapshot keeps the first-seen entry).

Usage
-----
  python reprice.py                 # reprice today's alerts.json + its archive
  python reprice.py --dry-run       # show what would change, write nothing
  python reprice.py --verbose
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from orchestrate import pick_option_contract

BASE_DIR      = Path.home() / "trading"
DATA_DIR      = BASE_DIR / "data"
ARCHIVE_DIR   = DATA_DIR / "archive"
ALERTS_PATH   = DATA_DIR / "alerts.json"
ALL_DATA_PATH = DATA_DIR / "all_data.json"
CONFIG_PATH   = BASE_DIR / "config.json"
LOG_DIR       = BASE_DIR / "logs"


# ─── Logging ────────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("reprice")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        log.addHandler(h)
        try:
            fh = logging.FileHandler(str(LOG_DIR / "reprice.log"))
            fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
            log.addHandler(fh)
        except Exception:
            pass
    log.setLevel(logging.INFO)
    return log


logger = _setup_logging()


# ─── Helpers ────────────────────────────────────────────────────────────────────
def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning(f"could not read {path}: {exc}")
        return None


def has_priced_tier(alert: dict) -> bool:
    """True if the alert already carries at least one tier with a live mid price."""
    rc = alert.get("recommended_contract") or {}
    tiers = rc.get("tiers") or {}
    for key in ("atm", "slight_otm", "affordable"):
        t = tiers.get(key)
        if t and float(t.get("mid_price") or 0) > 0:
            return True
    return False


def _price_and_chain(all_data: Optional[dict]) -> tuple[dict, dict]:
    """Build {symbol: price} and {symbol: options_chain} lookups from all_data.json."""
    price_by, chain_by = {}, {}
    for t in (all_data or {}).get("tickers", []):
        sym = t.get("symbol")
        if not sym:
            continue
        price_by[sym] = t.get("price") or 0
        chain_by[sym] = t.get("options_chain", {})
    return price_by, chain_by


def reprice_alert(
    alert: dict,
    price_by: dict,
    chain_by: dict,
    config: dict,
) -> Optional[dict]:
    """
    Re-fetch a live contract for one alert. Returns the new recommended_contract
    dict if a priced tier was found, else None. pick_option_contract falls back to
    a live yfinance fetch when the stored chain has no live quotes.
    """
    sym = alert["symbol"]
    price = price_by.get(sym) or 0
    try:
        contract = pick_option_contract(
            symbol        = sym,
            direction     = alert["direction"],
            current_price = price,
            options_chain = chain_by.get(sym, {}),
            dte_hint      = alert.get("suggested_dte"),
            budget        = config.get("budget", {}).get("total_usd", 500),
            config        = config,
        )
    except Exception as exc:
        logger.warning(f"reprice fetch failed for {sym}: {exc}")
        return None

    tiers = (contract or {}).get("tiers") or {}
    priced = any(
        tiers.get(k) and float((tiers.get(k) or {}).get("mid_price") or 0) > 0
        for k in ("atm", "slight_otm", "affordable")
    )
    return contract if priced else None


def _update_archives(scan_ts: str, filled: dict[tuple, dict], dry_run: bool) -> int:
    """
    Write repriced contracts into the archive file(s) whose scan_timestamp matches
    the alerts. Keyed by (symbol, direction). Returns number of archive alerts updated.
    """
    if not filled:
        return 0
    updated = 0
    for fp in sorted(glob.glob(str(ARCHIVE_DIR / "scored-*.json"))):
        arch = load_json(Path(fp))
        if not arch or arch.get("scan_timestamp") != scan_ts:
            continue
        changed = False
        for a in arch.get("alerts", []):
            key = (a.get("symbol"), a.get("direction"))
            if key in filled and not has_priced_tier(a):
                a["recommended_contract"] = filled[key]
                updated += 1
                changed = True
        if changed and not dry_run:
            Path(fp).write_text(json.dumps(arch, indent=2, default=str))
            logger.info(f"archive updated: {Path(fp).name}")
    return updated


def reprice(dry_run: bool = False) -> dict:
    """Reprice empty-contract alerts in alerts.json and the matching archive."""
    config = load_json(CONFIG_PATH) or {}
    alerts_doc = load_json(ALERTS_PATH)
    if not alerts_doc or not alerts_doc.get("alerts"):
        logger.info("no alerts.json / no alerts to reprice")
        return {"candidates": 0, "filled": 0, "archive_updated": 0}

    all_data = load_json(ALL_DATA_PATH)
    price_by, chain_by = _price_and_chain(all_data)
    scan_ts = alerts_doc.get("scan_timestamp")

    needing = [a for a in alerts_doc["alerts"] if not has_priced_tier(a)]
    logger.info(
        f"{len(needing)}/{len(alerts_doc['alerts'])} alert(s) lack a priced contract"
    )

    filled: dict[tuple, dict] = {}
    for alert in needing:
        new_contract = reprice_alert(alert, price_by, chain_by, config)
        if new_contract:
            alert["recommended_contract"] = new_contract
            filled[(alert["symbol"], alert["direction"])] = new_contract
            tier = next((new_contract["tiers"][k] for k in ("atm", "slight_otm", "affordable")
                         if new_contract["tiers"].get(k)), None)
            logger.info(f"  filled {alert['symbol']} {alert['direction']}: "
                        f"{(tier or {}).get('label', 'contract found')}")
        else:
            logger.info(f"  still no live contract for {alert['symbol']} "
                        f"(market closed or illiquid)")

    if filled and not dry_run:
        alerts_doc["repriced_at"] = datetime.now(timezone.utc).isoformat()
        ALERTS_PATH.write_text(json.dumps(alerts_doc, indent=2, default=str))
        logger.info(f"alerts.json updated ({len(filled)} contract(s) filled)")

    archive_updated = _update_archives(scan_ts, filled, dry_run)

    return {
        "candidates": len(needing),
        "filled": len(filled),
        "archive_updated": archive_updated,
        "dry_run": dry_run,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Backfill option contracts for empty-tier alerts")
    p.add_argument("--dry-run", action="store_true", help="Show changes, write nothing")
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    args = p.parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    result = reprice(dry_run=args.dry_run)
    if result.get("dry_run"):
        print(f"[dry-run] reprice: would fill {result['filled']}/{result['candidates']} "
              f"empty alert(s), would update {result['archive_updated']} archive entr(ies). "
              f"Nothing written.")
    else:
        print(f"reprice: {result['filled']}/{result['candidates']} empty alert(s) filled, "
              f"{result['archive_updated']} archive entr(ies) updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
