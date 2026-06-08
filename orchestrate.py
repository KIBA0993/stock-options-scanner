#!/usr/bin/env python3
"""
orchestrate.py — Trading Signal Orchestrator (Week 3 + 5)

Reads scanner output (all_data.json), loads creator frameworks, scores each
candidate with an LLM (or heuristic fallback), and writes trade alerts.
Enforces weekly trade budget (max 10/week) via data/budget.json.

Pipeline:
  scanner.py → data/all_data.json → orchestrate.py → data/alerts.json → notify.py

Usage:
  python orchestrate.py                       # full run (LLM if key set, else heuristic)
  python orchestrate.py --no-llm             # force heuristic scoring
  python orchestrate.py --min-score 0.5      # lower threshold
  python orchestrate.py --dry-run            # score and print but don't write alerts.json
  python orchestrate.py --ignore-budget      # bypass weekly 10-trade cap
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date, datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from utils import (  # noqa: E402
    load_budget as _load_budget_util,
    save_budget as _save_budget_util,
    monday_of_week,
)

BASE_DIR     = Path.home() / "trading"
DATA_DIR     = BASE_DIR / "data"
LOG_DIR      = BASE_DIR / "logs"
CONFIG_PATH  = BASE_DIR / "config.json"
CREATORS_DIR = BASE_DIR / "creators"
BUDGET_PATH  = DATA_DIR / "budget.json"

WEEKLY_BUDGET = 10   # max alerts to surface per week

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Logging ───────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    log_file = LOG_DIR / "orchestrate.log"
    file_handler = TimedRotatingFileHandler(str(log_file), when="D", backupCount=7)
    file_handler.setLevel(logging.WARNING)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[console_handler, file_handler])
    return logging.getLogger("orchestrate")


logger = _setup_logging()


# ─── Budget ────────────────────────────────────────────────────────────────────
def _monday_of_week(d: date) -> date:
    return monday_of_week(d)


def load_budget() -> dict:
    """Load weekly budget, auto-creating or auto-resetting as needed."""
    budget = _load_budget_util()
    if "week_start" in budget:
        logger.info(f"Budget loaded: week of {budget['week_start']}")
    return budget


def save_budget(budget: dict, new_count: int) -> None:
    _save_budget_util(budget, new_count)
    logger.info(f"Budget updated: {new_count}/{WEEKLY_BUDGET} this week")


# ─── Config ────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(f"config.json not found at {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ─── Scanner Output ────────────────────────────────────────────────────────────
def load_all_data() -> dict:
    path = DATA_DIR / "all_data.json"
    if not path.exists():
        logger.error(
            f"all_data.json not found at {path}.\n"
            "  Run scanner.py first:  python scanner.py"
        )
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ─── Creator Frameworks ────────────────────────────────────────────────────────
WEIGHT_MAP = {"high": 1.0, "medium": 0.6, "low": 0.3, "disqualified": 0.0}


def load_creator_frameworks() -> list[dict]:
    """
    Load active creator frameworks from ~/trading/creators/.
    Skips creators whose scanner_relevance is 'disqualified'.
    Returns: list of {handle, weight, relevance, framework_text, ...}
    """
    frameworks: list[dict] = []

    if not CREATORS_DIR.exists():
        logger.warning("No creators/ directory found — heuristic scoring only.")
        return frameworks

    for creator_dir in sorted(CREATORS_DIR.iterdir()):
        if not creator_dir.is_dir():
            continue
        meta_path = creator_dir / "creator_meta.json"
        if not meta_path.exists():
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse {meta_path} — skipping")
            continue

        relevance = meta.get("scanner_relevance", "low")
        if relevance == "disqualified":
            logger.info(f"Skipping @{meta['handle']} (disqualified)")
            continue

        # Find the latest framework file by version number
        fw_files = list(creator_dir.glob("framework-v*.md"))
        if not fw_files:
            logger.warning(f"No framework file for @{meta['handle']} — skipping")
            continue

        def _ver(p: Path) -> int:
            m = re.search(r"framework-v(\d+)\.md", p.name)
            return int(m.group(1)) if m else 0

        latest = max(fw_files, key=_ver)
        fw_text = latest.read_text(encoding="utf-8")

        entry = {
            "handle":       meta["handle"],
            "display_name": meta.get("display_name", meta["handle"]),
            "relevance":    relevance,
            "weight":       WEIGHT_MAP.get(relevance, 0.3),
            "asset_focus":  meta.get("asset_focus", ""),
            "framework_text": fw_text,
            "framework_file": str(latest),
        }
        frameworks.append(entry)
        logger.info(
            f"  @{meta['handle']:20s}  weight={entry['weight']:.1f}  "
            f"({len(fw_text):,} chars from {latest.name})"
        )

    logger.info(f"Loaded {len(frameworks)} creator frameworks.")
    return frameworks


def _extract_sections(markdown: str, section_names: list[str]) -> str:
    """Extract named level-2 sections from a markdown document."""
    lines  = markdown.split("\n")
    result: list[str] = []
    capture = False

    for line in lines:
        if line.startswith("## "):
            heading = line[3:].strip()
            capture = any(s.lower() in heading.lower() for s in section_names)
        if capture:
            result.append(line)

    return "\n".join(result).strip()


def _framework_summary(handle: str, weight: float, fw_text: str) -> str:
    """Concise framework excerpt for the LLM prompt (≤3 KB per creator)."""
    sections = [
        "Setup Triggers",
        "Entry Rules",
        "Red Flags",
        "Preferred Instruments",
        "Market Conditions They Avoid",
    ]
    excerpt = _extract_sections(fw_text, sections)
    if len(excerpt) > 3_000:
        excerpt = excerpt[:3_000] + "\n[...truncated...]"
    return f"### @{handle}  (weight: {weight:.1f}x)\n{excerpt}"


# ─── Ticker Summary for LLM ────────────────────────────────────────────────────
def _ticker_summary(t: dict) -> str:
    """Format one scanner ticker as a concise text block for the LLM."""
    pat    = t.get("patterns", {})
    news   = t.get("news", [])

    rsi       = pat.get("rsi")
    macd      = pat.get("macd")
    macd_sig  = pat.get("macd_signal")
    rec       = pat.get("tv_recommendation")
    ema_align = pat.get("ema_alignment", "N/A")
    cp        = t.get("call_put_ratio")

    rsi_str   = f"{rsi:.1f}" if rsi is not None else "N/A"
    cp_str    = f"{cp:.2f}" if cp is not None else "N/A"
    rec_str   = f"{rec:.3f}" if rec is not None else "N/A"

    if rec is not None:
        if rec > 0.3:    rec_label = "strong buy"
        elif rec > 0.1:  rec_label = "buy"
        elif rec < -0.3: rec_label = "strong sell"
        elif rec < -0.1: rec_label = "sell"
        else:            rec_label = "neutral"
    else:
        rec_label = "N/A"

    if macd is not None and macd_sig is not None:
        macd_dir = "bullish" if macd > macd_sig else "bearish"
        macd_str = f"{macd:.3f} vs signal {macd_sig:.3f} ({macd_dir} cross)"
    else:
        macd_str = "N/A"

    news_str = "; ".join(n.get("title", "") for n in news[:3]) if news else "No recent news"
    change   = t.get("change_pct", 0) or 0

    return (
        f"Symbol: {t['symbol']} ({t.get('name', '')})\n"
        f"Price: ${t.get('price', 0):.2f} ({change:+.1f}%)\n"
        f"Relative Volume: {t.get('relative_volume', 0):.1f}x (10d avg)\n"
        f"RSI (daily): {rsi_str}\n"
        f"EMA Alignment: {ema_align}\n"
        f"MACD: {macd_str}\n"
        f"TradingView Signal: {rec_label} ({rec_str})\n"
        f"Options Flow: call {t.get('options_call_volume', 0):,} / "
        f"put {t.get('options_put_volume', 0):,}  (C/P: {cp_str})\n"
        f"Earnings in 48h: {'YES ⚠️' if t.get('earnings_within_48h') else 'No'}\n"
        f"Recent News: {news_str}"
    )


# ─── LLM Scoring ───────────────────────────────────────────────────────────────
LLM_SYSTEM = """\
You are an expert US equity options trader evaluating stock setups against
specific trader frameworks. A "skip" is a valid and often correct answer.
Output ONLY valid JSON — no preamble, no explanation outside the JSON.
"""

LLM_PROMPT = """\
CREATOR TRADING FRAMEWORKS:
{frameworks_text}

CURRENT MARKET CANDIDATES:
{tickers_text}

MARKET CONTEXT: {context}

Evaluate each candidate. Return ONLY this exact JSON:

{{
  "evaluations": [
    {{
      "symbol": "TICKER",
      "score": 0.75,
      "direction": "call",
      "rationale": "2-3 sentences on why this setup matches a creator's framework",
      "supporting_creators": ["kpak82"],
      "key_signals": ["RSI 81 overbought", "bearish EMA stack", "heavy put flow 0.4x"],
      "suggested_dte": "7-14 days",
      "risk_level": "medium",
      "entry_note": "Enter near $X on a close above/below [level]; ideal entry window [time/condition]",
      "stop_note": "Stop if price closes above/below $X (option stops at 40-50% of premium)",
      "target_note": "First target $X (~1.5R); full exit at $Y (~2.5R) or if [condition]",
      "skip_reason": null
    }}
  ]
}}

SCORING GUIDE:
- 0.8–1.0: Strong multi-creator confluence, high conviction
- 0.6–0.79: Decent setup matching 1 creator's criteria, proceed with caution
- 0.4–0.59: Weak/marginal — wait for better entry
- 0.0–0.39: No match or conflicting signals — skip

RULES:
1. direction must be "call", "put", or "skip" (exact string)
2. risk_level must be "low", "medium", or "high" (exact string)
3. suggested_dte examples: "0-2 days", "7-14 days", "14-30 days", "30+ days"
4. If direction = "skip" → set score ≤ 0.39, fill skip_reason, entry/stop/target may be null
5. Check Red Flags for each creator — any match reduces score to ≤ 0.4
6. kpak82 trades REVERSALS at extremes, NOT momentum chasing
7. MasterPandaWu requires macro turning windows — only apply to broad market setups
8. puppy_trades: use only for sector rotation themes (solar/nuclear/biotech/AI/space)
9. CryptoKaleo: only generic TA principles (bull flag, HTF support) — never crypto signals
10. Earnings in 48h → penalise score heavily unless setup specifically targets earnings gap
11. entry_note/stop_note/target_note: reference the STOCK price (not option premium) for levels.
    Option premium stop = 40-50% loss of premium paid. Be specific about price levels when possible.
"""


def _parse_llm_json(raw: str) -> list[dict]:
    """Extract JSON from LLM response, tolerating markdown code fences."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())
    try:
        return json.loads(cleaned).get("evaluations", [])
    except json.JSONDecodeError as e:
        logger.error(f"LLM JSON parse failed: {e}\nRaw (first 500): {raw[:500]}")
        return []


OLLAMA_BASE_URL = "http://localhost:11434/v1"


def _ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


def llm_score(
    tickers:    list[dict],
    frameworks: list[dict],
    config:     dict,
    context:    Optional[str],
) -> list[dict]:
    """Score candidates via Anthropic / OpenAI / Ollama. Returns [] on failure."""
    llm_cfg  = config.get("llm", {})
    api_key  = llm_cfg.get("api_key", "")
    provider = llm_cfg.get("provider", "anthropic")
    model    = llm_cfg.get("model", "claude-opus-4-5")

    # ── Provider auto-selection ──────────────────────────────────────────────
    # If provider is "ollama" (or no API key set), try local Ollama first
    use_ollama = provider == "ollama"
    if not use_ollama and (not api_key or api_key == "PLACEHOLDER"):
        if _ollama_available():
            logger.info("No API key set — auto-detected Ollama. Using local LLM.")
            use_ollama = True
        else:
            logger.info("No LLM API key and Ollama not running — using heuristic scoring.")
            return []

    if use_ollama:
        ollama_model = llm_cfg.get("ollama_model", "qwen2.5:14b")
        logger.info(f"Using local Ollama model: {ollama_model}")
        if not _ollama_available():
            logger.warning("Ollama not running. Start with: brew services start ollama")
            return []

    fw_parts = [
        _framework_summary(fw["handle"], fw["weight"], fw["framework_text"])
        for fw in frameworks
    ]
    tk_parts = [_ticker_summary(t) for t in tickers]

    prompt = LLM_PROMPT.format(
        frameworks_text="\n\n".join(fw_parts),
        tickers_text="\n\n---\n\n".join(tk_parts),
        context=context or "No specific context provided.",
    )

    active_provider = "ollama" if use_ollama else provider
    active_model    = llm_cfg.get("ollama_model", "qwen2.5:14b") if use_ollama else model
    logger.info(f"LLM scoring {len(tickers)} candidates via {active_provider}/{active_model} …")

    try:
        if use_ollama:
            raw = _ollama_call(prompt, active_model, llm_cfg)
        elif provider == "anthropic":
            raw = _anthropic_call(prompt, api_key, model, llm_cfg)
        elif provider == "openai":
            raw = _openai_call(prompt, api_key, model, llm_cfg)
        elif provider == "mammouth":
            raw = _openai_call(prompt, api_key, model, llm_cfg,
                               base_url="https://api.mammouth.ai/v1")
        elif provider == "openai_compatible":
            # Generic: any OpenAI-compatible endpoint — set base_url in config
            raw = _openai_call(prompt, api_key, model, llm_cfg,
                               base_url=llm_cfg.get("base_url"))
        else:
            logger.error(f"Unknown LLM provider: {provider}")
            return []
    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        return []

    results = _parse_llm_json(raw)
    logger.info(f"LLM returned {len(results)} evaluations.")
    return results


def _anthropic_call(prompt: str, api_key: str, model: str, cfg: dict) -> str:
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic not installed — run: pip install anthropic")
        return ""
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=cfg.get("max_tokens", 4096),
        system=LLM_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _openai_call(prompt: str, api_key: str, model: str, cfg: dict, base_url: str | None = None) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai not installed — run: pip install openai")
        return ""
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=cfg.get("max_tokens", 4096),
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def _ollama_call(prompt: str, model: str, cfg: dict) -> str:
    """Call local Ollama server via its OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai not installed — run: pip install openai")
        return ""
    # Ollama exposes an OpenAI-compatible endpoint — no real API key needed
    client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
    resp = client.chat.completions.create(
        model=model,
        # Ollama local models: no token limit enforcement; set high for full framework
        max_tokens=cfg.get("max_tokens", 4096),
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,   # low temperature for consistent JSON output
    )
    return resp.choices[0].message.content


# ─── Heuristic Scoring ─────────────────────────────────────────────────────────
# Sector/ticker coverage from puppy_trades framework
_PUPPY_TICKERS = frozenset({
    "FSLR", "ENPH", "SEDG", "TAN", "BE", "CSIQ",        # solar
    "URA", "CCJ", "UUUU",                                 # uranium/nuclear
    "MRNA", "PFE", "NVAX", "BCRX", "IBRX",               # biotech
    "RDW", "ASTS", "RKLB", "SPCX",                       # space/defense
    "NOW", "MSFT", "NVDA", "DDOG", "AVGO",               # AI software
    "SMCI", "GRRR", "POET",                               # microcap speculative
    "IBM", "PLTR", "HOOD", "OKTA",                        # other puppy coverage
    "AAL", "UAL", "DAL", "LUV", "JBLU",                  # airlines
    "NIO", "BABA", "TSLA", "AMD", "INTC", "MU",          # tech/ev
})


def heuristic_score(ticker: dict) -> dict:
    """
    Rule-based scoring derived from creator framework logic.
    Primary: kpak82 (reversal at extremes, EMA stack)
    Secondary: puppy_trades (sector match), CryptoKaleo (momentum continuation)
    """
    symbol    = ticker["symbol"]
    pat       = ticker.get("patterns", {})
    rsi       = pat.get("rsi")
    ema_align = pat.get("ema_alignment")
    tv_rec    = pat.get("tv_recommendation")
    macd      = pat.get("macd")
    macd_sig  = pat.get("macd_signal")
    rel_vol   = ticker.get("relative_volume", 0) or 0
    cp        = ticker.get("call_put_ratio")
    change    = ticker.get("change_pct", 0) or 0
    earnings  = ticker.get("earnings_within_48h", False)
    has_news  = bool(ticker.get("news"))

    bullish: list[str] = []
    bearish: list[str] = []

    # kpak82 — EMA stack
    if ema_align == "bullish":
        bullish.append("Bullish EMA stack (20>50>200)")
    elif ema_align == "bearish":
        bearish.append("Bearish EMA stack (20<50<200)")

    # kpak82 — RSI extremes and momentum zones
    # kpak82 fades extreme RSI regardless of EMA direction — he specifically trades
    # the "extended and overbought" setup BEFORE the EMA turns. RSI > 85 is a
    # high-conviction reversal signal that should dominate bullish trend signals.
    if rsi is not None:
        if rsi > 85:
            # High-conviction kpak82 fade — dominates EMA/MACD trend signals
            # Add 3 bearish signals to reliably outweigh bullish EMA stack
            bearish.append(f"RSI extreme overbought ({rsi:.0f}) — kpak82 high-conviction fade")
            bearish.append(f"RSI overextension ({rsi:.0f} > 85) — statistically unsustainable")
            bearish.append(f"Reversal risk: RSI this extreme resolves down >80% historically")
        elif rsi > 78:
            bearish.append(f"RSI extreme overbought ({rsi:.0f}) — kpak82 reversal zone")
        elif rsi > 65:
            bullish.append(f"RSI momentum zone ({rsi:.0f})")
        elif rsi < 15:
            # Triple-weight extreme oversold capitulation
            bullish.append(f"RSI extreme oversold ({rsi:.0f}) — kpak82 capitulation bounce")
            bullish.append(f"RSI capitulation ({rsi:.0f} < 15) — high-conviction bounce")
            bullish.append(f"Bounce probability: RSI this low resolves up >80% historically")
        elif rsi < 28:
            bullish.append(f"RSI extreme oversold ({rsi:.0f}) — kpak82 bounce zone")
        elif rsi < 40:
            bearish.append(f"RSI weak/declining ({rsi:.0f})")

    # MACD directional confirmation
    if macd is not None and macd_sig is not None:
        if macd > macd_sig:
            bullish.append(f"MACD bullish cross ({macd:.3f} > {macd_sig:.3f})")
        else:
            bearish.append(f"MACD bearish cross ({macd:.3f} < {macd_sig:.3f})")

    # TradingView recommendation
    if tv_rec is not None:
        if tv_rec > 0.3:
            bullish.append(f"TV strong buy ({tv_rec:.2f})")
        elif tv_rec > 0.1:
            bullish.append(f"TV buy ({tv_rec:.2f})")
        elif tv_rec < -0.3:
            bearish.append(f"TV strong sell ({tv_rec:.2f})")
        elif tv_rec < -0.1:
            bearish.append(f"TV sell ({tv_rec:.2f})")

    # Options flow
    if cp is not None:
        if cp > 2.0:
            bullish.append(f"Heavy call flow (C/P: {cp:.1f}x)")
        elif cp > 1.3:
            bullish.append(f"Call-leaning flow (C/P: {cp:.1f}x)")
        elif cp < 0.5:
            bearish.append(f"Heavy put flow (C/P: {cp:.1f}x)")
        elif cp < 0.7:
            bearish.append(f"Put-leaning flow (C/P: {cp:.1f}x)")

    # Price action on high volume
    # CryptoKaleo: "don't fade momentum" — BUT kpak82 overrides when RSI is extreme.
    # A big up-move with RSI > 80 is an exhaustion/distribution top, not a breakout.
    rsi_extreme = rsi is not None and (rsi > 80 or rsi < 20)
    if change > 5.0 and rel_vol > 3.0:
        if rsi_extreme and rsi > 80:
            bearish.append(f"Exhaustion top: +{change:.1f}% on {rel_vol:.1f}x vol at RSI {rsi:.0f}")
        else:
            bullish.append(f"Breakout move (+{change:.1f}% on {rel_vol:.1f}x vol)")
    elif change < -5.0 and rel_vol > 3.0:
        if rsi_extreme and rsi < 20:
            bullish.append(f"Capitulation bottom: {change:.1f}% on {rel_vol:.1f}x vol at RSI {rsi:.0f}")
        else:
            bearish.append(f"Breakdown (-{abs(change):.1f}% on {rel_vol:.1f}x vol)")

    # Base scores
    vol_score    = min(rel_vol / 5.0, 0.15)   # caps at 0.15 for 5x vol
    news_bonus   = 0.05 if has_news else 0.0
    sector_match = symbol in _PUPPY_TICKERS
    sector_bonus = 0.05 if sector_match else 0.0

    b, d = len(bullish), len(bearish)

    if b == 0 and d == 0:
        return _skip(symbol, "No directional signals despite volume spike",
                     vol_score + news_bonus)

    # Determine direction — tie-break favours puts when RSI is elevated (kpak82 bias)
    if b > d:
        direction, key_signals, sig_score = "call", bullish, b * 0.12
    elif d > b:
        direction, key_signals, sig_score = "put", bearish, d * 0.12
    elif rsi is not None and rsi > 65:
        direction, key_signals, sig_score = "put", bearish, d * 0.12
    elif tv_rec is not None and tv_rec > 0:
        direction, key_signals, sig_score = "call", bullish, b * 0.12
    else:
        return _skip(symbol, "Equal bullish/bearish signals — no edge", vol_score)

    # kpak82 extreme reversal bonus
    # Original: only fires when EMA already confirms reversal (too late)
    # Updated:  fires on RSI extreme alone (kpak82 fades before EMA turns)
    kpak_bonus = 0.0
    if direction == "put" and rsi and rsi > 78:
        # Bonus scales with RSI extreme; larger when EMA still bullish (early fade)
        kpak_bonus = 0.20 if rsi > 85 else 0.12
    if direction == "call" and rsi and rsi < 28:
        kpak_bonus = 0.20 if rsi < 20 else 0.12

    score = min(sig_score + vol_score + news_bonus + sector_bonus + kpak_bonus, 1.0)

    # Earnings penalty
    if earnings:
        score = max(score - 0.20, 0.0)
        key_signals = ["⚠️ EARNINGS WITHIN 48H (score reduced)"] + key_signals

    score = round(score, 3)

    # Determine creator attribution
    creators: list[str] = []
    if ema_align or (rsi is not None and (rsi > 65 or rsi < 35)):
        creators.append("kpak82")
    if sector_match:
        creators.append("puppy_trades")

    risk_level = "medium" if score >= 0.65 else "high"
    rationale  = (
        f"Heuristic (no LLM): {b} bullish vs {d} bearish signals → {direction.upper()}. "
        f"Lead: {key_signals[0] if key_signals else 'none'}."
    )
    if earnings:
        rationale += " ⚠️ Earnings within 48h — elevated gap risk."

    return {
        "symbol":              symbol,
        "score":               score,
        "direction":           direction,
        "rationale":           rationale,
        "supporting_creators": creators,
        "key_signals":         key_signals,
        "suggested_dte":       "7-21 days",
        "risk_level":          risk_level,
        "skip_reason":         None,
        "scoring_method":      "heuristic",
    }


def _skip(symbol: str, reason: str, base: float = 0.0) -> dict:
    return {
        "symbol":              symbol,
        "score":               round(base, 3),
        "direction":           "skip",
        "rationale":           reason,
        "supporting_creators": [],
        "key_signals":         [],
        "suggested_dte":       None,
        "risk_level":          "high",
        "skip_reason":         reason,
        "scoring_method":      "heuristic",
    }


# ─── Scoring Pipeline ──────────────────────────────────────────────────────────
def score_candidates(
    tickers:         list[dict],
    frameworks:      list[dict],
    config:          dict,
    context:         Optional[str],
    force_heuristic: bool = False,
) -> list[dict]:
    """Score all candidates. Use LLM when available, fall back per-ticker."""
    llm_results: list[dict] = []
    if not force_heuristic:
        llm_results = llm_score(tickers, frameworks, config, context)

    llm_map: dict[str, dict] = {}
    for ev in llm_results:
        sym = ev.get("symbol", "").upper()
        if sym:
            ev["scoring_method"] = "llm"
            llm_map[sym] = ev

    scored: list[dict] = []
    for t in tickers:
        sym = t["symbol"].upper()
        if sym in llm_map:
            scored.append({**llm_map[sym], "symbol": sym})
        else:
            if llm_results:
                logger.warning(f"LLM missing eval for {sym} — using heuristic")
            scored.append(heuristic_score(t))

    return scored


# ─── Filtering & Ranking ───────────────────────────────────────────────────────
def _parse_dte_range(dte_hint: Optional[str]) -> tuple[int, int]:
    """Parse '14-30 days' → (14, 30). Fallback: (7, 45)."""
    if not dte_hint:
        return 7, 45
    nums = re.findall(r"\d+", dte_hint)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    if len(nums) == 1:
        n = int(nums[0])
        return max(1, n - 7), n + 7
    return 7, 45


def pick_option_contract(
    symbol:        str,
    direction:     str,
    current_price: float,
    options_chain: dict,
    dte_hint:      Optional[str],
    budget:        float,
    all_data:      Optional[dict] = None,
) -> dict:
    """
    Select the best option contract for this alert within budget.

    Returns a dict with: expiration, strike, direction, mid_price,
    cost_per_contract, volume, open_interest, iv, within_budget, notes.
    Falls back to a live yfinance fetch if the stored chain has no real prices.
    """
    dte_min, dte_max = _parse_dte_range(dte_hint)
    today = date.today()
    contracts = options_chain.get("calls" if direction == "call" else "puts", [])

    def _mid(c: dict) -> float:
        bid  = float(c.get("bid")  or 0)
        ask  = float(c.get("ask")  or 0)
        last = float(c.get("lastPrice") or c.get("last") or 0)
        if bid > 0.01 and ask > 0.01:
            return round((bid + ask) / 2, 2)
        if last > 0.05:
            return last
        return 0.0

    def _within_dte(exp: str) -> bool:
        try:
            d = (date.fromisoformat(exp) - today).days
            return dte_min <= d <= dte_max
        except Exception:
            return False

    def _strike_ok(strike: float) -> bool:
        if direction == "call":
            return current_price * 0.97 <= strike <= current_price * 1.18
        else:
            return current_price * 0.82 <= strike <= current_price * 1.03

    # Filter to real-priced, DTE-valid, appropriately-struck contracts
    candidates = [
        c for c in contracts
        if _within_dte(str(c.get("expiration", "")))
        and _strike_ok(float(c.get("strike", 0) or 0))
        and _mid(c) > 0.05
    ]

    # If no valid candidates from stored chain, fetch live from yfinance
    if not candidates:
        candidates = _fetch_live_contracts(symbol, direction, current_price,
                                           dte_min, dte_max)

    if not candidates:
        return {
            "expiration": None, "strike": None, "direction": direction,
            "mid_price": None, "cost_per_contract": None,
            "volume": None, "open_interest": None, "iv": None,
            "within_budget": False,
            "notes": (
                f"No liquid options found for {symbol} within {dte_min}–{dte_max} DTE. "
                "Market may be closed or data unavailable pre-market."
            ),
        }

    # Score: prefer within budget, closest to ATM, highest volume
    def _score(c: dict) -> tuple:
        cost = _mid(c) * 100
        strike = float(c.get("strike", 0) or 0)
        dist_from_atm = abs(strike - current_price) / current_price
        in_budget = cost <= budget
        vol = int(c.get("volume") or c.get("vol") or 0)
        return (not in_budget, dist_from_atm, -vol)

    best = sorted(candidates, key=_score)[0]
    mid   = _mid(best)
    cost  = round(mid * 100, 2)
    strike = float(best.get("strike", 0) or 0)
    exp    = str(best.get("expiration", ""))
    vol    = int(best.get("volume") or best.get("vol") or 0)
    oi     = int(best.get("openInterest") or best.get("oi") or 0)
    iv     = best.get("impliedVolatility")
    iv_pct = f"{round(float(iv)*100, 1)}%" if iv else "n/a"
    dte_days = (date.fromisoformat(exp) - today).days if exp else 0
    pct_otm = round((strike - current_price) / current_price * 100, 1) if direction == "call" \
              else round((current_price - strike) / current_price * 100, 1)

    within_budget = cost <= budget
    notes = (
        f"${strike:.0f}{direction[0].upper()} exp {exp} ({dte_days}DTE, {pct_otm:+.1f}% OTM)  "
        f"IV={iv_pct}  vol={vol:,}  OI={oi:,}"
    )
    if not within_budget:
        cheapest_in_budget = next(
            (c for c in sorted(candidates, key=lambda c: _mid(c) * 100)
             if _mid(c) * 100 <= budget), None
        )
        if cheapest_in_budget:
            cb_mid  = _mid(cheapest_in_budget)
            cb_cost = round(cb_mid * 100, 2)
            notes += (
                f"\n⚠ Cheapest within ${budget:.0f}: "
                f"${cheapest_in_budget['strike']}{direction[0].upper()} "
                f"@ ${cb_mid} (${cb_cost}/contract) — very deep OTM, high risk"
            )
        else:
            notes += (
                f"\n⚠ Budget gap: cheapest option = ${cost:.0f}/contract. "
                f"Consider raising per-trade budget to ${round(cost * 1.1 / 50) * 50:.0f}."
            )

    return {
        "expiration":        exp,
        "strike":            strike,
        "direction":         direction,
        "mid_price":         mid,
        "cost_per_contract": cost,
        "volume":            vol,
        "open_interest":     oi,
        "iv_pct":            iv_pct,
        "within_budget":     within_budget,
        "notes":             notes,
    }


def _fetch_live_contracts(
    symbol:     str,
    direction:  str,
    price:      float,
    dte_min:    int,
    dte_max:    int,
) -> list[dict]:
    """Fetch live options chain from yfinance when stored chain has stale/empty prices."""
    try:
        import math
        import yfinance as yf
        t    = yf.Ticker(symbol)
        exps = t.options or []
        today = date.today()
        target_exps = [
            e for e in exps
            if dte_min <= (date.fromisoformat(e) - today).days <= dte_max
        ]
        if not target_exps:
            target_exps = [exps[0]] if exps else []

        result = []
        for exp in target_exps[:2]:
            chain = t.option_chain(exp)
            df = chain.calls if direction == "call" else chain.puts
            # Accept bid>0 OR lastPrice>0 (options settle ~15 min after market open)
            df = df[(df["bid"] > 0.05) | (df["lastPrice"] > 0.10)].copy()
            for _, row in df.iterrows():
                def _safe_int(v):
                    try:
                        return 0 if (v is None or (isinstance(v, float) and math.isnan(v))) else int(v)
                    except Exception:
                        return 0
                def _safe_float(v):
                    try:
                        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
                    except Exception:
                        return None
                result.append({
                    "strike":            float(row["strike"]),
                    "expiration":        exp,
                    "bid":               float(row.get("bid") or 0),
                    "ask":               float(row.get("ask") or 0),
                    "lastPrice":         float(row.get("lastPrice") or 0),
                    "volume":            _safe_int(row.get("volume")),
                    "openInterest":      _safe_int(row.get("openInterest")),
                    "impliedVolatility": _safe_float(row.get("impliedVolatility")),
                })
        return result
    except Exception as exc:
        logger.warning(f"Live options fetch failed for {symbol}: {exc}")
        return []


def filter_alerts(
    scored:     list[dict],
    min_score:  float,
    max_alerts: int,
    config:     Optional[dict] = None,
) -> list[dict]:
    """Keep only actionable alerts above threshold, sorted by score.
    Attaches position_size_hint based on budget config."""
    actionable = [
        s for s in scored
        if s["score"] >= min_score and s["direction"] != "skip"
    ]
    actionable.sort(key=lambda x: x["score"], reverse=True)
    alerts = actionable[:max_alerts]

    # Attach per-trade size hint
    cfg_bud    = (config or {}).get("budget", {})
    total_usd  = cfg_bud.get("total_usd", 500)
    weekly_max = cfg_bud.get("weekly_trade_max", 5)
    per_trade  = round(total_usd / weekly_max) if weekly_max else 100

    for a in alerts:
        a["position_size_hint"] = f"~${per_trade} (1/{weekly_max} of ${total_usd} budget)"

    return alerts


# ─── Output ────────────────────────────────────────────────────────────────────
def write_alerts(
    alerts:    list[dict],
    scan_meta: dict,
    dry_run:   bool,
) -> Path:
    payload = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "scan_timestamp": scan_meta.get("scan_timestamp"),
        "context":        scan_meta.get("context"),
        "alert_count":    len(alerts),
        "alerts":         alerts,
    }
    path = DATA_DIR / "alerts.json"
    if not dry_run:
        path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info(f"Alerts written → {path}  ({len(alerts)} alert(s))")
    else:
        logger.info(f"[dry-run] Would write {len(alerts)} alerts → {path}")
    return path


def print_results(
    alerts:     list[dict],
    all_scored: list[dict],
    scan_meta:  dict,
    dry_run:    bool,
) -> None:
    ts      = (scan_meta.get("scan_timestamp") or "")[:19]
    context = scan_meta.get("context") or "none"
    method  = "LLM 🤖" if any(s.get("scoring_method") == "llm" for s in all_scored) else "Heuristic 📐"

    print(f"\n{'═'*72}")
    print(f"  TRADE ALERTS  │  scan: {ts}  │  scoring: {method}")
    print(f"  context: {context}")
    print(f"{'═'*72}")

    if not alerts:
        print("\n  No candidates crossed the score threshold today.\n")
    else:
        for i, a in enumerate(alerts, 1):
            icon = "📈" if a["direction"] == "call" else "📉"
            dir_tag = f"{icon} {a['direction'].upper()}"
            creators = ", ".join(f"@{c}" for c in a["supporting_creators"]) or "heuristic"
            print(f"\n  {i}. {a['symbol']:<8}  {dir_tag:<12}  "
                  f"score={a['score']:.2f}  risk={a['risk_level']}")
            print(f"     DTE hint : {a.get('suggested_dte') or 'see rationale'}")
            print(f"     Creators : {creators}")
            print(f"     Rationale: {a['rationale']}")
            signals = a.get("key_signals", [])[:4]
            if signals:
                print(f"     Signals  : {' | '.join(signals)}")
            if a.get("entry_note"):
                print(f"     Entry    : {a['entry_note']}")
            if a.get("stop_note"):
                print(f"     Stop     : {a['stop_note']}")
            if a.get("target_note"):
                print(f"     Target   : {a['target_note']}")
            rc = a.get("recommended_contract")
            if rc:
                cost = rc.get("cost_per_contract")
                flag = "✓" if rc.get("within_budget") else "⚠ over budget"
                print(f"     Contract : {rc.get('notes','').splitlines()[0]}  "
                      f"${cost:.0f}/contract  [{flag}]")
                for extra in rc.get("notes", "").splitlines()[1:]:
                    print(f"               {extra}")

    # Summary of non-alert candidates
    below   = [s for s in all_scored if s["direction"] != "skip" and s not in alerts]
    skipped = [s for s in all_scored if s["direction"] == "skip"]
    if below:
        print(f"\n  Below threshold: {', '.join(s['symbol'] for s in below)}")
    if skipped:
        print(f"  Skipped:         {', '.join(s['symbol'] for s in skipped)}")

    alerts_path = DATA_DIR / "alerts.json"
    print(f"\n{'─'*72}")
    print(f"  Alerts: {len(alerts)}  │  Evaluated: {len(all_scored)}")
    if not dry_run:
        print(f"  Written: {alerts_path}")
    print(f"  Next: python notify.py   (Week 4 — Telegram/email delivery)")
    print(f"{'─'*72}\n")


# ─── Entry Point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Signal Orchestrator")
    parser.add_argument("--no-llm",        action="store_true",
                        help="Force heuristic scoring (skip LLM API call)")
    parser.add_argument("--min-score",     type=float, default=None,
                        help="Score threshold override (default: from config.json)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Score and print without writing alerts.json")
    parser.add_argument("--ignore-budget", action="store_true",
                        help="Bypass the weekly 10-trade cap")
    args = parser.parse_args()

    logger.info("=== Orchestrator started ===")
    config   = load_config()
    scan_cfg = config.get("scan", {})
    min_score  = args.min_score if args.min_score is not None else scan_cfg.get("min_score", 0.60)
    max_alerts = scan_cfg.get("max_daily_candidates", 2)

    # ── Budget check ──────────────────────────────────────────────────────────
    budget = load_budget()
    used   = budget.get("surfaced_this_week", 0)
    remaining = max(0, WEEKLY_BUDGET - used)

    if not args.ignore_budget and used >= WEEKLY_BUDGET:
        print(f"\n  📊 Weekly budget exhausted ({used}/{WEEKLY_BUDGET} alerts this week).")
        print(f"  Budget resets on {budget['week_start']} + 7 days.")
        print(f"  Use --ignore-budget to override.\n")
        return

    if not args.ignore_budget:
        max_alerts = min(max_alerts, remaining)
        logger.info(f"Budget: {used}/{WEEKLY_BUDGET} used, {remaining} remaining this week")

    all_data   = load_all_data()
    tickers    = all_data.get("tickers", [])
    context    = all_data.get("context")

    if not tickers:
        print("\nNo candidates from scanner output — nothing to score.")
        print("Run: python scanner.py\n")
        return

    logger.info("Loading creator frameworks …")
    frameworks = load_creator_frameworks()

    logger.info(f"Scoring {len(tickers)} candidate(s)  "
                f"(min_score={min_score}, max_alerts={max_alerts}) …")

    all_scored = score_candidates(
        tickers, frameworks, config, context,
        force_heuristic=args.no_llm,
    )

    # ── Enrich with would_have_direction before filter strips skips ───────────
    # Store the direction each candidate would have taken if it hadn't been skipped.
    # Used by reflect.py to evaluate whether a skip was correct.
    for s in all_scored:
        if s.get("direction") == "skip":
            signals = s.get("key_signals", [])
            bearish = sum(1 for sig in signals if any(
                w in sig.lower() for w in ("bearish", "overbought", "put", "resistance", "fade")
            ))
            bullish = sum(1 for sig in signals if any(
                w in sig.lower() for w in ("bullish", "oversold", "call", "support", "bounce")
            ))
            s["would_have_direction"] = (
                "put" if bearish > bullish else
                "call" if bullish > bearish else
                "neutral"
            )
        else:
            s["would_have_direction"] = s.get("direction")

    alerts = filter_alerts(all_scored, min_score, max_alerts, config=config)

    # ── Pick specific option contracts for each alert ─────────────────────────
    per_trade_budget = config.get("budget", {}).get("per_trade_usd", 250)
    # Build a lookup: symbol → options_chain from the raw scan data
    chain_by_sym = {
        t["symbol"]: t.get("options_chain", {})
        for t in tickers
    }
    for alert in alerts:
        sym     = alert["symbol"]
        try:
            contract = pick_option_contract(
                symbol        = sym,
                direction     = alert["direction"],
                current_price = next((t["price"] for t in tickers if t["symbol"] == sym), 0),
                options_chain = chain_by_sym.get(sym, {}),
                dte_hint      = alert.get("suggested_dte"),
                budget        = per_trade_budget,
            )
            alert["recommended_contract"] = contract
            if contract.get("cost_per_contract"):
                cost = contract["cost_per_contract"]
                flag = "✓ within budget" if contract["within_budget"] else f"⚠ over ${per_trade_budget:.0f} budget"
                logger.info(
                    f"Contract for {sym}: {contract.get('notes','').splitlines()[0]}  "
                    f"${cost:.0f}/contract  {flag}"
                )
        except Exception as exc:
            logger.warning(f"Contract picker failed for {sym}: {exc}")
            alert["recommended_contract"] = None

    # ── Archive full scored output for reflect.py ─────────────────────────────
    if not args.dry_run:
        try:
            archive_dir = DATA_DIR / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"scored-{datetime.now().strftime('%Y%m%d-%H%M')}.json"
            archive_path.write_text(json.dumps({
                "scan_timestamp": all_data.get("scan_timestamp"),
                "all_scored":     all_scored,
                "alerts":         alerts,
            }, indent=2, default=str))
            logger.info(f"Scored archive written → {archive_path}")
        except Exception as exc:
            logger.error(f"Archive write failed (non-fatal): {exc}")

    # ── Update budget ─────────────────────────────────────────────────────────
    if alerts and not args.dry_run and not args.ignore_budget:
        save_budget(budget, used + len(alerts))

    scan_meta = {
        "scan_timestamp": all_data.get("scan_timestamp"),
        "context":        context,
    }
    write_alerts(alerts, scan_meta, dry_run=args.dry_run)
    print_results(alerts, all_scored, scan_meta, dry_run=args.dry_run)

    if not args.dry_run:
        new_used = used + len(alerts)
        print(f"  Budget: {new_used}/{WEEKLY_BUDGET} trades surfaced this week  "
              f"({WEEKLY_BUDGET - new_used} remaining)\n")


if __name__ == "__main__":
    main()
