#!/usr/bin/env python3
"""
backtest.py — Historical Signal Quality Validator (Week 5)

Fetches 1 year of daily OHLCV data via yfinance and tests whether simple
momentum / mean-reversion strategies have historically worked on the same
stocks our scanner flags. This cross-checks whether we're fishing in
productive waters — it does NOT prove individual signal quality.

Strategies tested:
  ema_cross      — EMA20 crosses above EMA50 → hold for 14 days
  rsi_oversold   — RSI drops below 30 → hold for 14 days
  supertrend     — price crosses above 10-period ATR-based supertrend line
  combined       — all three must agree (high-conviction filter)

Metrics reported:
  win_rate       — % of signals that produced a positive return
  avg_return_pct — average % gain/loss per signal
  sharpe_ratio   — risk-adjusted return (annualised)
  verdict        — ROBUST / MODERATE / WEAK based on combined metrics

Usage:
  backtest.py NVDA AAPL TSLA             # validate specific tickers
  backtest.py --from-alerts              # validate current alerts.json tickers
  backtest.py --from-journal             # validate all tickers in trade_journal.jsonl
  backtest.py NVDA --period 2y           # extend lookback (default: 1y)
  backtest.py NVDA --strategy rsi_oversold  # single strategy
  backtest.py --report                   # summary table for all tickers
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional
import logging

BASE_DIR      = Path.home() / "trading"
DATA_DIR      = BASE_DIR / "data"
LOG_DIR       = BASE_DIR / "logs"
ALERTS_PATH   = DATA_DIR / "alerts.json"
JOURNAL_PATH  = DATA_DIR / "trade_journal.jsonl"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Logging ──────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    fh = TimedRotatingFileHandler(
        str(LOG_DIR / "backtest.log"), when="D", backupCount=14
    )
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[ch, fh],
    )
    return logging.getLogger("backtest")


logger = _setup_logging()

STRATEGIES = ("ema_cross", "rsi_oversold", "supertrend", "combined")
HOLD_DAYS  = 14   # how many calendar days to hold after a signal
MIN_SIGNALS = 5   # minimum signals needed to report a verdict


# ─── Technical Indicators ─────────────────────────────────────────────────────
def _ema(series: list[float], period: int) -> list[Optional[float]]:
    """Exponential Moving Average (EMA)."""
    result: list[Optional[float]] = [None] * len(series)
    if len(series) < period:
        return result
    k = 2 / (period + 1)
    ema = sum(series[:period]) / period
    result[period - 1] = ema
    for i in range(period, len(series)):
        ema = series[i] * k + ema * (1 - k)
        result[i] = ema
    return result


def _rsi(closes: list[float], period: int = 14) -> list[Optional[float]]:
    """Relative Strength Index (RSI)."""
    result: list[Optional[float]] = [None] * len(closes)
    if len(closes) < period + 1:
        return result
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(closes)):
        j = i - period
        avg_gain = (avg_gain * (period - 1) + gains[j]) / period
        avg_loss = (avg_loss * (period - 1) + losses[j]) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))
    return result


def _atr(highs: list[float], lows: list[float], closes: list[float],
          period: int = 10) -> list[Optional[float]]:
    """Average True Range (ATR)."""
    result: list[Optional[float]] = [None] * len(closes)
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return result
    atr_val = sum(trs[:period]) / period
    result[period] = atr_val
    for i in range(period + 1, len(closes)):
        atr_val = (atr_val * (period - 1) + trs[i - 1]) / period
        result[i] = atr_val
    return result


def _supertrend_signals(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 10,
    multiplier: float = 3.0,
) -> list[bool]:
    """
    Returns a boolean list: True where price crosses above the Supertrend line
    (bullish crossover = potential buy signal for calls).
    """
    atrs = _atr(highs, lows, closes, period)
    upper_band = [None] * len(closes)
    in_uptrend = [False] * len(closes)
    signals    = [False] * len(closes)

    for i in range(period, len(closes)):
        atr = atrs[i]
        if atr is None:
            continue
        mid = (highs[i] + lows[i]) / 2
        upper = mid + multiplier * atr

        if upper_band[i - 1] is None:
            upper_band[i] = upper
        else:
            upper_band[i] = min(upper, upper_band[i - 1]) if closes[i - 1] > upper_band[i - 1] else upper

        if closes[i] > (upper_band[i] or 0):
            in_uptrend[i] = True
        else:
            in_uptrend[i] = False

        if in_uptrend[i] and not in_uptrend[i - 1]:
            signals[i] = True   # crossover → buy signal

    return signals


# ─── Signal Generators ────────────────────────────────────────────────────────
def _ema_cross_signals(closes: list[float]) -> list[bool]:
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    signals = [False] * len(closes)
    for i in range(1, len(closes)):
        if (ema20[i] is not None and ema50[i] is not None and
                ema20[i - 1] is not None and ema50[i - 1] is not None):
            # Golden cross: EMA20 crosses above EMA50
            if ema20[i - 1] <= ema50[i - 1] and ema20[i] > ema50[i]:
                signals[i] = True
    return signals


def _rsi_oversold_signals(closes: list[float], threshold: float = 30.0) -> list[bool]:
    rsis = _rsi(closes)
    signals = [False] * len(closes)
    for i in range(1, len(closes)):
        if rsis[i] is not None and rsis[i - 1] is not None:
            # RSI crosses back above threshold from oversold
            if rsis[i - 1] < threshold and rsis[i] >= threshold:
                signals[i] = True
    return signals


# ─── Backtest Engine ──────────────────────────────────────────────────────────
def run_strategy(
    dates:   list[str],
    closes:  list[float],
    highs:   list[float],
    lows:    list[float],
    strategy: str,
    hold_days: int = HOLD_DAYS,
) -> dict:
    """
    Run one strategy over the price series. Returns a result dict.
    hold_days is in trading days (approximate — we use index offset).
    """
    n = len(closes)
    if strategy == "ema_cross":
        signals = _ema_cross_signals(closes)
    elif strategy == "rsi_oversold":
        signals = _rsi_oversold_signals(closes)
    elif strategy == "supertrend":
        signals = _supertrend_signals(highs, lows, closes)
    elif strategy == "combined":
        ema_s = _ema_cross_signals(closes)
        rsi_s = _rsi_oversold_signals(closes)
        sup_s = _supertrend_signals(highs, lows, closes)
        signals = [a or b or c for a, b, c in zip(ema_s, rsi_s, sup_s)]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    trades: list[dict] = []
    i = 0
    while i < n:
        if signals[i]:
            exit_idx = min(i + hold_days, n - 1)
            entry_px = closes[i]
            exit_px  = closes[exit_idx]
            ret_pct  = (exit_px - entry_px) / entry_px * 100
            trades.append({
                "entry_date": dates[i],
                "exit_date":  dates[exit_idx],
                "entry_px":   entry_px,
                "exit_px":    exit_px,
                "return_pct": ret_pct,
                "win":        ret_pct > 0,
            })
            i = exit_idx + 1  # no overlapping trades
        else:
            i += 1

    if not trades:
        return {"strategy": strategy, "signal_count": 0, "verdict": "NO_SIGNALS"}

    returns = [t["return_pct"] for t in trades]
    wins    = sum(1 for t in trades if t["win"])
    avg_ret = sum(returns) / len(returns)
    win_rate = wins / len(trades)

    # Simplified Sharpe (daily returns proxy)
    if len(returns) > 1:
        mean = avg_ret
        std  = (sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
        sharpe = (mean / std * (252 / hold_days) ** 0.5) if std > 0 else 0.0
    else:
        sharpe = 0.0

    verdict = _verdict(win_rate, avg_ret, sharpe, len(trades))

    return {
        "strategy":     strategy,
        "signal_count": len(trades),
        "win_rate":     round(win_rate, 3),
        "avg_return":   round(avg_ret, 2),
        "sharpe":       round(sharpe, 2),
        "verdict":      verdict,
        "trades":       trades,
    }


def _verdict(win_rate: float, avg_ret: float, sharpe: float, n: int) -> str:
    if n < MIN_SIGNALS:
        return "INSUFFICIENT_DATA"
    if win_rate >= 0.60 and avg_ret >= 2.0 and sharpe >= 1.0:
        return "ROBUST"
    elif win_rate >= 0.50 and avg_ret >= 0.5:
        return "MODERATE"
    else:
        return "WEAK"


# ─── Ticker Backtest ──────────────────────────────────────────────────────────
def backtest_ticker(
    symbol:   str,
    period:   str = "1y",
    strategy: str = "combined",
    verbose:  bool = True,
) -> dict:
    """Fetch price data and run all or one strategy."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed — run: pip install yfinance")
        sys.exit(1)

    ticker = yf.Ticker(symbol)
    hist   = ticker.history(period=period, interval="1d", auto_adjust=True)

    if hist.empty:
        if verbose:
            print(f"  ⚠ {symbol}: no price data available")
        return {"symbol": symbol, "error": "no_data"}

    dates  = [str(d.date()) for d in hist.index]
    closes = hist["Close"].tolist()
    highs  = hist["High"].tolist()
    lows   = hist["Low"].tolist()

    strategies_to_run = STRATEGIES if strategy == "combined" else [strategy]
    results = {}
    for strat in strategies_to_run:
        try:
            r = run_strategy(dates, closes, highs, lows, strat)
            results[strat] = r
        except Exception as exc:
            logger.warning(f"{symbol}/{strat} error: {exc}")
            results[strat] = {"strategy": strat, "error": str(exc)}

    if verbose:
        _print_ticker_results(symbol, results, len(dates))

    return {"symbol": symbol, "bar_count": len(dates), "strategies": results}


def _print_ticker_results(symbol: str, results: dict, bar_count: int) -> None:
    print(f"\n  ── {symbol} ({bar_count} trading days) ─────────────────────────")
    print(f"  {'Strategy':<18} {'Signals':>7} {'Win%':>6} {'Avg Ret':>8} {'Sharpe':>7} {'Verdict'}")
    print(f"  {'─'*62}")
    for strat, r in results.items():
        if "error" in r:
            print(f"  {strat:<18} ERROR: {r['error']}")
            continue
        if r.get("signal_count", 0) == 0:
            print(f"  {strat:<18} {'0':>7}   {'—':>6} {'—':>8} {'—':>7} NO_SIGNALS")
            continue
        v       = r["verdict"]
        v_icon  = {"ROBUST": "✅", "MODERATE": "🟡", "WEAK": "❌",
                   "INSUFFICIENT_DATA": "⏳"}.get(v, "?")
        print(f"  {strat:<18} {r['signal_count']:>7} "
              f"{r['win_rate']*100:>5.0f}% "
              f"{r['avg_return']:>+7.1f}% "
              f"{r['sharpe']:>7.2f} "
              f"{v_icon} {v}")


def _overall_verdict(results: dict) -> str:
    verdicts = [r.get("verdict", "WEAK") for r in results.values() if "verdict" in r]
    if not verdicts:
        return "UNKNOWN"
    if "ROBUST" in verdicts:
        return "ROBUST"
    elif "MODERATE" in verdicts:
        return "MODERATE"
    else:
        return "WEAK"


# ─── Source loaders ────────────────────────────────────────────────────────────
def symbols_from_alerts() -> list[str]:
    if not ALERTS_PATH.exists():
        print("  No alerts.json found. Run: trade-scan")
        return []
    with open(ALERTS_PATH) as f:
        data = json.load(f)
    return [a["symbol"] for a in data.get("alerts", [])]


def symbols_from_journal() -> list[str]:
    if not JOURNAL_PATH.exists():
        return []
    seen: set[str] = set()
    with open(JOURNAL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    seen.add(e["symbol"].upper())
                except Exception:
                    pass
    return sorted(seen)


# ─── Report ────────────────────────────────────────────────────────────────────
def print_report(all_results: list[dict]) -> None:
    """Summary table across all tickers."""
    print(f"\n{'═'*72}")
    print(f"  BACKTEST REPORT — {len(all_results)} ticker(s)")
    print(f"{'═'*72}")
    print(f"  {'Symbol':<8} {'Bars':>5} {'EMA Cross':>12} {'RSI OS':>12} {'SuperTrend':>12} {'Overall'}")
    print(f"  {'─'*70}")

    for r in all_results:
        if "error" in r:
            print(f"  {r['symbol']:<8} ERROR: {r.get('error')}")
            continue
        sym = r["symbol"]
        bars = r.get("bar_count", "?")
        strats = r.get("strategies", {})

        def _v(key: str) -> str:
            s = strats.get(key, {})
            if not s or "error" in s:
                return "  ERR"
            v = s.get("verdict", "?")
            icons = {"ROBUST": "✅ ROB", "MODERATE": "🟡 MOD",
                     "WEAK": "❌ WEK", "INSUFFICIENT_DATA": "⏳ LOW",
                     "NO_SIGNALS": " —    "}
            return icons.get(v, f"  {v[:5]}")

        overall_v = _overall_verdict(strats)
        ov_icon = {"ROBUST": "✅", "MODERATE": "🟡", "WEAK": "❌"}.get(overall_v, "?")

        print(f"  {sym:<8} {bars:>5} {_v('ema_cross'):>12} {_v('rsi_oversold'):>12} "
              f"{_v('supertrend'):>12} {ov_icon} {overall_v}")

    print(f"\n  Legend: ✅ ROBUST (win≥60%, ret≥2%, Sharpe≥1)  "
          f"🟡 MODERATE  ❌ WEAK  ⏳ <{MIN_SIGNALS} signals")
    print(f"\n  Interpretation:")
    print(f"    ROBUST/MODERATE → historically strong TA signals on this ticker")
    print(f"    WEAK → TA momentum strategies don't reliably predict moves here")
    print(f"    Use this to re-weight scanner candidates — prefer ROBUST tickers")
    print(f"{'═'*72}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Historical Signal Quality Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("symbols", nargs="*",
                        help="Ticker symbols to validate (e.g. NVDA AAPL)")
    parser.add_argument("--from-alerts",  action="store_true",
                        help="Load symbols from current alerts.json")
    parser.add_argument("--from-journal", action="store_true",
                        help="Load symbols from trade_journal.jsonl")
    parser.add_argument("--strategy",
                        choices=list(STRATEGIES) + ["all"],
                        default="all",
                        help="Strategy to test (default: all)")
    parser.add_argument("--period",
                        choices=["3mo", "6mo", "1y", "2y"],
                        default="1y",
                        help="Lookback period (default: 1y)")
    parser.add_argument("--report", action="store_true",
                        help="Print summary table after all tickers")
    args = parser.parse_args()

    # Build symbol list
    symbols: list[str] = [s.upper() for s in args.symbols]
    if args.from_alerts:
        symbols += symbols_from_alerts()
    if args.from_journal:
        symbols += symbols_from_journal()

    # Deduplicate, preserve order
    seen: set[str] = set()
    unique: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    symbols = unique

    if not symbols:
        parser.print_help()
        print("\n  Example: backtest.py NVDA AAPL --period 1y")
        print("           backtest.py --from-alerts --report\n")
        return

    strat_arg = args.strategy if args.strategy != "all" else "combined"

    print(f"\n  Backtesting {len(symbols)} symbol(s)  "
          f"period={args.period}  hold={HOLD_DAYS}d\n")

    all_results = []
    for sym in symbols:
        result = backtest_ticker(sym, period=args.period, strategy=strat_arg)
        all_results.append(result)

    if args.report or len(symbols) > 1:
        print_report(all_results)


if __name__ == "__main__":
    main()
