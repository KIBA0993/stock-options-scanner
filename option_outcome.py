#!/usr/bin/env python3
"""
option_outcome.py — Option P&L evaluation for reflect (swing + intraday).

Uses archived contract tiers (ATM preferred) and yfinance for exit mids.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import yfinance as yf

from market_calendar import is_trading_day, last_trading_day_on_or_before

logger = logging.getLogger(__name__)


def pick_contract_tier(alert: dict, prefer: str = "atm") -> Optional[dict]:
    """ATM tier by default; falls back to slight_otm then affordable."""
    rc = alert.get("recommended_contract") or {}
    tiers = rc.get("tiers") or {}
    order = (prefer, "slight_otm", "affordable") if prefer == "atm" else (prefer, "atm", "affordable")
    for key in order:
        tier = tiers.get(key)
        if tier and float(tier.get("mid_price") or 0) > 0:
            return tier
    return None


def underlying_close_on(symbol: str, on_date: date) -> Optional[float]:
    try:
        end = on_date + timedelta(days=5)
        hist = yf.Ticker(symbol).history(
            start=on_date.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        if hist is None or hist.empty:
            return None
        day_rows = hist[hist.index.date == on_date]
        if day_rows.empty:
            day_rows = hist.head(1)
        return float(day_rows["Close"].iloc[-1])
    except Exception as exc:
        logger.warning(f"Underlying close failed for {symbol} on {on_date}: {exc}")
        return None


def underlying_move_pct(symbol: str, entry_date: date, exit_date: date) -> Optional[float]:
    entry = underlying_close_on(symbol, entry_date)
    exit_ = underlying_close_on(symbol, exit_date)
    if entry is None or exit_ is None or entry <= 0:
        return None
    return round((exit_ - entry) / entry * 100, 2)


def add_trading_days(start: date, n: int) -> date:
    """Return calendar date after `n` NYSE trading sessions."""
    d = start
    count = 0
    while count < n:
        d += timedelta(days=1)
        if is_trading_day(d):
            count += 1
    return d


def _intrinsic_mid(direction: str, strike: float, underlying: float) -> float:
    if direction == "call":
        return round(max(0.0, underlying - strike), 2)
    return round(max(0.0, strike - underlying), 2)


def option_mid_on_date(
    symbol: str,
    direction: str,
    strike: float,
    expiration: str,
    on_date: date,
) -> Optional[float]:
    """Option mid on `on_date` — chain quote if live, else intrinsic at expiry."""
    exp_date = date.fromisoformat(expiration)
    uclose = underlying_close_on(symbol, on_date)
    if uclose is None:
        return None

    if on_date >= exp_date:
        u_exp = underlying_close_on(symbol, exp_date) or uclose
        return _intrinsic_mid(direction, strike, u_exp)

    try:
        chain = yf.Ticker(symbol).option_chain(expiration)
        df = chain.calls if direction == "call" else chain.puts
        row = df[df["strike"] == strike]
        if row.empty:
            row = df.iloc[(df["strike"] - strike).abs().argsort()[:1]]
        bid = float(row.iloc[0].get("bid") or 0)
        ask = float(row.iloc[0].get("ask") or 0)
        if bid > 0.01 and ask > 0.01:
            return round((bid + ask) / 2, 2)
        last = float(row.iloc[0].get("lastPrice") or 0)
        if last > 0:
            return round(last, 2)
    except Exception as exc:
        logger.warning(f"Option chain failed {symbol} {expiration} on {on_date}: {exc}")

    return _intrinsic_mid(direction, strike, uclose)


def _outcome_result(
    symbol: str,
    direction: str,
    contract: dict,
    entry_date: date,
    exit_date: date,
    underlying_entry: Optional[float] = None,
) -> Optional[dict]:
    entry_mid = float(contract.get("mid_price") or 0)
    if entry_mid <= 0:
        return None

    strike = float(contract["strike"])
    expiration = str(contract["expiration"])
    exit_mid = option_mid_on_date(symbol, direction, strike, expiration, exit_date)
    if exit_mid is None:
        return None

    pnl_pct = round((exit_mid - entry_mid) / entry_mid * 100, 2)
    und_pct = underlying_move_pct(symbol, entry_date, exit_date)
    if und_pct is None and underlying_entry and underlying_entry > 0:
        u_exit = underlying_close_on(symbol, exit_date)
        if u_exit:
            und_pct = round((u_exit - underlying_entry) / underlying_entry * 100, 2)

    return {
        "outcome_option_pnl_pct": pnl_pct,
        "outcome_underlying_pct": und_pct,
        "entry_mid": entry_mid,
        "exit_mid": exit_mid,
        "exit_date": exit_date.isoformat(),
        "strike": strike,
        "expiration": expiration,
        "contract_label": contract.get(
            "label", f"${strike:.0f}{direction[0].upper()} {expiration}",
        ),
    }


def fetch_intraday_option_outcome(alert: dict) -> Optional[dict]:
    """Legacy same-day-to-close P&L (prefer fetch_intraday_exit_outcome)."""
    contract = pick_contract_tier(alert)
    if not contract:
        return None
    try:
        alert_date = date.fromisoformat((alert.get("scan_timestamp") or "")[:10])
    except ValueError:
        return None
    return _outcome_result(
        alert.get("symbol", ""),
        alert.get("direction", ""),
        contract,
        alert_date,
        alert_date,
        float(alert.get("underlying_price") or 0) or None,
    )


def fetch_intraday_exit_outcome(
    entry: dict,
    exit_alert: dict,
) -> Optional[dict]:
    """Option P&L from entry contract mid to exit alert (exit mid or underlying)."""
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

    exit_mid = exit_alert.get("exit_option_mid")
    if exit_mid is not None:
        exit_mid = float(exit_mid)
    else:
        try:
            exit_date = date.fromisoformat((exit_alert.get("scan_timestamp") or "")[:10])
        except ValueError:
            exit_date = date.today()
        exit_mid = option_mid_on_date(symbol, direction, strike, expiration, exit_date)

    if exit_mid is None:
        exit_u = float(exit_alert.get("underlying_price") or 0)
        if exit_u > 0:
            exit_mid = _intrinsic_mid(direction, strike, exit_u)
        else:
            return None

    pnl_pct = round((exit_mid - entry_mid) / entry_mid * 100, 2)
    entry_u = float(entry.get("underlying_price") or 0)
    exit_u = float(exit_alert.get("underlying_price") or 0)
    und_pct = None
    if entry_u > 0 and exit_u > 0:
        if direction == "call":
            und_pct = round((exit_u - entry_u) / entry_u * 100, 2)
        else:
            und_pct = round((entry_u - exit_u) / entry_u * 100, 2)

    try:
        exit_date_str = (exit_alert.get("scan_timestamp") or "")[:10]
    except ValueError:
        exit_date_str = date.today().isoformat()

    return {
        "outcome_option_pnl_pct": pnl_pct,
        "outcome_underlying_pct": und_pct,
        "entry_mid": entry_mid,
        "exit_mid": round(float(exit_mid), 2),
        "exit_date": exit_date_str,
        "strike": strike,
        "expiration": expiration,
        "contract_label": contract.get(
            "label", f"${strike:.0f}{direction[0].upper()} {expiration}",
        ),
        "outcome_exit_alert": True,
        "exit_timestamp": exit_alert.get("scan_timestamp"),
    }


def no_exit_intraday_outcome(entry: dict, config: dict | None = None) -> dict:
    """Entry with no exit alert before daily reflect → full premium loss."""
    cfg = (config or {}).get("intraday_0dte", {})
    loss_pct = float(cfg.get("no_exit_loss_pct", -100))
    contract = pick_contract_tier(entry)
    entry_mid = float(contract.get("mid_price") or 0) if contract else None
    try:
        entry_date = (entry.get("scan_timestamp") or "")[:10]
    except ValueError:
        entry_date = date.today().isoformat()
    return {
        "outcome_option_pnl_pct": loss_pct,
        "outcome_underlying_pct": None,
        "entry_mid": entry_mid,
        "exit_mid": 0.0 if loss_pct <= -99 else None,
        "exit_date": entry_date,
        "contract_label": contract.get("label") if contract else None,
        "outcome_no_exit": True,
        "outcome_exit_alert": False,
    }


def fetch_swing_option_outcome(
    alert: dict,
    entry_date: date,
    hold_days: int = 5,
    as_of: date | None = None,
) -> Optional[dict]:
    """
    Swing option P&L from entry mid to mid on `as_of` (default: latest session).

    If the full hold window has not elapsed, marks `outcome_interim=True` using
    the most recent trading day's option mid instead of waiting.
    """
    contract = pick_contract_tier(alert)
    if not contract:
        return None

    as_of = as_of or last_trading_day_on_or_before()
    target_exit = add_trading_days(entry_date, hold_days)
    price_date = min(target_exit, as_of)
    if price_date < entry_date:
        return None

    result = _outcome_result(
        alert.get("symbol", ""),
        alert.get("direction", ""),
        contract,
        entry_date,
        price_date,
        None,
    )
    if not result:
        return None

    result["outcome_interim"] = price_date < target_exit
    result["target_exit_date"] = target_exit.isoformat()
    result["outcome_as_of"] = price_date.isoformat()
    return result


def evaluate_swing_alert(
    alert: dict,
    entry_date: date,
    config: dict | None = None,
    as_of: date | None = None,
) -> dict:
    cfg = (config or {}).get("swing_reflect", {})
    hold_days = int(cfg.get("hold_days", 5))
    min_pnl = float(cfg.get("min_option_pnl_pct", 0))

    result = fetch_swing_option_outcome(
        alert, entry_date, hold_days=hold_days, as_of=as_of,
    )
    if not result:
        still_open = exit_date_pending(entry_date, hold_days)
        return {
            "miss_type": "false_take",
            "outcome_5d_pct": None,
            "outcome_option_pnl_pct": None,
            "outcome_underlying_pct": None,
            "outcome_pending": still_open,
            "outcome_final": False,
            "outcome_interim": still_open,
        }

    pnl = result["outcome_option_pnl_pct"]
    interim = bool(result.get("outcome_interim"))
    miss_type = "correct_take" if pnl > min_pnl else "false_take"
    return {
        "miss_type": miss_type,
        "outcome_5d_pct": pnl,
        "outcome_option_pnl_pct": pnl,
        "outcome_underlying_pct": result.get("outcome_underlying_pct"),
        "option_entry_mid": result.get("entry_mid"),
        "option_exit_mid": result.get("exit_mid"),
        "option_exit_date": result.get("exit_date"),
        "option_target_exit_date": result.get("target_exit_date"),
        "outcome_as_of": result.get("outcome_as_of"),
        "contract_label": result.get("contract_label"),
        "outcome_pending": interim,
        "outcome_final": not interim,
        "outcome_interim": interim,
    }


def evaluate_intraday_alert(
    entry: dict,
    config: dict | None = None,
    exit_alert: dict | None = None,
) -> dict:
    """
    Score a 0DTE entry for daily reflect.

    - With exit_alert: option P&L from entry → exit alert pricing.
    - Without exit_alert: full loss (no_exit_loss_pct, default -100%).
    """
    cfg = (config or {}).get("intraday_0dte", {})
    min_pnl = float(cfg.get("min_option_pnl_pct", 0))

    if exit_alert:
        result = fetch_intraday_exit_outcome(entry, exit_alert)
    else:
        result = no_exit_intraday_outcome(entry, config)

    if not result or result.get("outcome_option_pnl_pct") is None:
        return {
            "miss_type": "false_take",
            "outcome_5d_pct": None,
            "outcome_option_pnl_pct": None,
            "outcome_underlying_pct": None,
            "outcome_pending": False,
            "outcome_no_exit": exit_alert is None,
            "outcome_exit_alert": bool(exit_alert),
        }

    pnl = result["outcome_option_pnl_pct"]
    miss_type = "correct_take" if pnl > min_pnl else "false_take"
    return {
        "miss_type": miss_type,
        "outcome_5d_pct": pnl,
        "outcome_option_pnl_pct": pnl,
        "outcome_underlying_pct": result.get("outcome_underlying_pct"),
        "option_entry_mid": result.get("entry_mid"),
        "option_exit_mid": result.get("exit_mid"),
        "option_exit_date": result.get("exit_date"),
        "exit_timestamp": result.get("exit_timestamp"),
        "contract_label": result.get("contract_label"),
        "outcome_pending": False,
        "outcome_no_exit": bool(result.get("outcome_no_exit")),
        "outcome_exit_alert": bool(result.get("outcome_exit_alert")),
    }


def exit_date_pending(entry_date: date, hold_days: int) -> bool:
    return add_trading_days(entry_date, hold_days) > date.today()
