"""
trendlines.py — automatic trendline detection from OHLCV data.

Uses swing highs/lows to fit support and resistance trendlines,
classifies the chart pattern, and flags proximity/breakout conditions.

Input:  list of OHLCV dicts  [{"date":…,"open":…,"high":…,"low":…,"close":…,"volume":…}, …]
Output: dict with trendline metrics the LLM can reason about.
"""

from __future__ import annotations

import math
from typing import Optional


def _safe_float(v) -> Optional[float]:
    """Return float or None if value is None/NaN/invalid."""
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


# ── helpers ────────────────────────────────────────────────────────────────

def _find_swings(
    highs: list[float],
    lows:  list[float],
    window: int = 2,
) -> tuple[list[int], list[int]]:
    """
    Return (swing_high_indices, swing_low_indices).
    A swing high at i: high[i] is the max in [i-window … i+window].
    A swing low  at i: low[i]  is the min in [i-window … i+window].
    """
    n = len(highs)
    sh, sl = [], []
    for i in range(window, n - window):
        window_highs = highs[i - window : i + window + 1]
        window_lows  = lows[i - window : i + window + 1]
        if highs[i] == max(window_highs):
            sh.append(i)
        if lows[i] == min(window_lows):
            sl.append(i)
    return sh, sl


def _fit_line(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Ordinary least-squares. Returns (slope, intercept).
    Pure Python — no numpy dependency.
    """
    n = len(x)
    if n < 2:
        return 0.0, y[0] if y else 0.0
    sx  = sum(x)
    sy  = sum(y)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    sx2 = sum(xi * xi for xi in x)
    d   = n * sx2 - sx * sx
    if d == 0:
        return 0.0, sy / n
    slope     = (n * sxy - sx * sy) / d
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _direction(slope_pct: float) -> str:
    if slope_pct > 0.15:
        return "rising"
    if slope_pct < -0.15:
        return "falling"
    return "flat"


# ── main API ───────────────────────────────────────────────────────────────

def analyze_trendlines(ohlcv: list[dict]) -> dict:
    """
    Detect support / resistance trendlines from 30-day OHLCV data.

    Returns a dict (empty dict if data is insufficient):
    {
        "support_trendline":    { current_level, slope_pct_per_day, direction,
                                   price_distance_pct, touches },
        "resistance_trendline": { … },
        "pattern":              "ascending_channel" | "descending_channel" |
                                "ascending_triangle" | "descending_triangle" |
                                "converging_wedge" | "expanding_wedge" |
                                "horizontal_range" | "mixed",
        "near_support":              bool,   # price within 2.5% above support TL
        "near_resistance":           bool,   # price within 2% below resistance TL
        "broke_above_resistance":    bool,   # price > resistance TL by >0.5%
        "broke_below_support":       bool,   # price < support TL
        "trendline_summary":         str,    # human-readable one-liner for LLM
    }
    """
    if not ohlcv or len(ohlcv) < 8:
        return {}

    highs  = [_safe_float(b.get("high",  b.get("close", 0))) or 0.0 for b in ohlcv]
    lows   = [_safe_float(b.get("low",   b.get("close", 0))) or 0.0 for b in ohlcv]
    closes = [_safe_float(b.get("close", 0)) or 0.0 for b in ohlcv]

    # Drop trailing zero bars (NaN rows from yfinance)
    while closes and closes[-1] == 0.0:
        closes.pop(); highs.pop(); lows.pop()

    if len(closes) < 8:
        return {}

    current  = closes[-1]
    if current == 0.0:
        return {}

    n_bars   = len(closes)
    last_idx = float(n_bars - 1)

    swing_highs, swing_lows = _find_swings(highs, lows, window=2)

    result: dict = {}

    # ── Resistance trendline ────────────────────────────────────────────────
    if len(swing_highs) >= 2:
        sh_idx = swing_highs[-5:]
        sh_y   = [highs[i] for i in sh_idx]
        slope, intercept = _fit_line([float(i) for i in sh_idx], sh_y)
        projected = slope * last_idx + intercept
        proj_safe = _safe_float(projected)
        if proj_safe and proj_safe > 0:
            dist_pct  = round((current - proj_safe) / proj_safe * 100, 2)
            slope_pct = round(slope / proj_safe * 100, 3)
            result["resistance_trendline"] = {
                "current_level":      round(proj_safe, 2),
                "slope_pct_per_day":  slope_pct,
                "direction":          _direction(slope_pct),
                "price_distance_pct": dist_pct,
                "touches":            len(sh_idx),
            }

    # ── Support trendline ───────────────────────────────────────────────────
    if len(swing_lows) >= 2:
        sl_idx = swing_lows[-5:]
        sl_y   = [lows[i] for i in sl_idx]
        slope, intercept = _fit_line([float(i) for i in sl_idx], sl_y)
        projected = slope * last_idx + intercept
        proj_safe = _safe_float(projected)
        if proj_safe and proj_safe > 0:
            dist_pct  = round((current - proj_safe) / proj_safe * 100, 2)
            slope_pct = round(slope / proj_safe * 100, 3)
            result["support_trendline"] = {
                "current_level":      round(proj_safe, 2),
                "slope_pct_per_day":  slope_pct,
                "direction":          _direction(slope_pct),
                "price_distance_pct": dist_pct,
                "touches":            len(sl_idx),
            }

    if not result:
        return {}

    # ── Pattern ─────────────────────────────────────────────────────────────
    rt = result.get("resistance_trendline", {})
    st = result.get("support_trendline", {})
    if rt and st:
        key = (rt["direction"], st["direction"])
        pattern = {
            ("rising",  "rising"):  "ascending_channel",
            ("falling", "falling"): "descending_channel",
            ("flat",    "rising"):  "ascending_triangle",
            ("falling", "flat"):    "descending_triangle",
            ("falling", "rising"):  "converging_wedge",
            ("rising",  "falling"): "expanding_wedge",
            ("flat",    "flat"):    "horizontal_range",
            ("rising",  "flat"):    "rising_support_flat_resistance",
            ("flat",    "falling"): "falling_support_flat_resistance",
        }.get(key, "mixed")
        result["pattern"] = pattern
    elif rt:
        result["pattern"] = f"resistance_{rt['direction']}_only"
    elif st:
        result["pattern"] = f"support_{st['direction']}_only"

    # ── Proximity / breakout flags ──────────────────────────────────────────
    if rt:
        d = rt["price_distance_pct"]
        result["near_resistance"]        = -2.0 <= d <= 0.5
        result["broke_above_resistance"] = d > 0.5
    else:
        result["near_resistance"]        = False
        result["broke_above_resistance"] = False

    if st:
        d = st["price_distance_pct"]
        result["near_support"]        = 0.0 <= d <= 2.5
        result["broke_below_support"] = d < 0.0
    else:
        result["near_support"]        = False
        result["broke_below_support"] = False

    # ── Human-readable summary for LLM ─────────────────────────────────────
    parts = []
    if result.get("pattern"):
        parts.append(f"pattern={result['pattern']}")
    if st:
        lvl = st["current_level"]
        dist = st["price_distance_pct"]
        tag  = "NEAR" if result.get("near_support") else ("BROKE" if result.get("broke_below_support") else "above")
        parts.append(f"support_TL=${lvl:.2f} ({dist:+.1f}%,{tag})")
    if rt:
        lvl  = rt["current_level"]
        dist = rt["price_distance_pct"]
        tag  = "NEAR" if result.get("near_resistance") else ("BROKE_ABOVE" if result.get("broke_above_resistance") else "below")
        parts.append(f"resist_TL=${lvl:.2f} ({dist:+.1f}%,{tag})")
    result["trendline_summary"] = " | ".join(parts) if parts else "insufficient data"

    return result
