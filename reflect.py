#!/usr/bin/env python3
"""
reflect.py — Weekly Self-Reflection & Framework Evolution Loop (Week 6)

Runs every Friday at 5pm (via launchd) to:
  1. Load this week's closed trades + archived scored candidates (incl. skips)
  2. Fetch outcomes via yfinance for non-journaled skips
  3. Append records to reflect_history.jsonl (pattern ledger)
  4. Detect patterns: same 3-of-3 feature signature 3+ times = amendment candidate
  5. Write draft amendment files for human review

Usage:
  reflect report                            # This week's performance summary
  reflect patterns                          # Accumulated pattern ledger
  reflect amendments                        # List pending draft amendments
  reflect apply <handle> [--date DATE]      # Merge amendment into framework-v{n+1}.md
  reflect reject <handle> --date DATE       # Mark amendment rejected
  reflect --auto                            # Non-interactive full run (launchd mode)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from utils import (
    monday_of_week,
    rsi_bucket,
    momentum_bucket,
    rel_vol_bucket,
    load_week_sent_alerts,
    find_archive_alert,
    find_archive_alert_on_date,
)

try:
    from intraday_0dte import (
        find_exit_for_entry,
        load_alerts_for_date,
        load_entry_alerts_for_date,
        load_week_intraday_alerts,
    )
    from option_outcome import evaluate_intraday_alert, evaluate_swing_alert
except ImportError:
    load_week_intraday_alerts = None  # type: ignore
    load_entry_alerts_for_date = None  # type: ignore
    load_alerts_for_date = None  # type: ignore
    find_exit_for_entry = None  # type: ignore
    evaluate_intraday_alert = None  # type: ignore
    evaluate_swing_alert = None  # type: ignore

BASE_DIR      = Path.home() / "trading"
DATA_DIR      = BASE_DIR / "data"
ARCHIVE_DIR   = DATA_DIR / "archive"
LOG_DIR       = BASE_DIR / "logs"
CREATORS_DIR  = BASE_DIR / "creators"
JOURNAL_PATH  = DATA_DIR / "trade_journal.jsonl"
HISTORY_PATH  = DATA_DIR / "reflect_history.jsonl"

MISS_THRESHOLD    = 3     # occurrences before an amendment draft is generated
OUTCOME_WINDOW    = 5     # trading days to check for outcome
OUTCOME_MIN_MOVE  = 5.0   # % move in predicted direction = false skip
ARCHIVE_KEEP_DAYS = 14    # delete archive files older than this


# ─── Logging ───────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = TimedRotatingFileHandler(
        str(LOG_DIR / "reflect.log"), when="D", backupCount=14
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    log = logging.getLogger("reflect")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        log.addHandler(fh)
    return log


logger = _setup_logging()


# ─── History I/O ───────────────────────────────────────────────────────────────
def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    records = []
    with open(HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def append_history(records: list[dict]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def upsert_sent_alert_records(records: list[dict]) -> None:
    """Replace sent_alert rows for the same week/symbol/direction (interim refresh)."""
    if not records:
        return
    keys = {
        (r["week_start"], r["symbol"].upper(), r["direction"].lower())
        for r in records
        if r.get("source") == "sent_alert"
    }
    if not keys:
        append_history(records)
        return

    history = load_history()
    kept = [
        r for r in history
        if not (
            r.get("source") == "sent_alert"
            and (
                r.get("week_start"),
                (r.get("symbol") or "").upper(),
                (r.get("direction") or "").lower(),
            ) in keys
        )
    ]
    kept.extend(records)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        for r in kept:
            f.write(json.dumps(r, default=str) + "\n")


def load_journal() -> list[dict]:
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
                    pass
    return entries


# ─── Archive loading ────────────────────────────────────────────────────────────
def load_week_archives(week_start: date) -> list[dict]:
    """
    Load all scored archive files belonging to the given ISO week (Mon–Sun).
    Returns a flat list of scored candidate records.
    """
    if not ARCHIVE_DIR.exists():
        return []
    week_end = week_start + timedelta(days=6)
    candidates: list[dict] = []
    for f in sorted(ARCHIVE_DIR.glob("scored-*.json")):
        # filename: scored-YYYYMMDD-HHMM.json
        try:
            date_str = f.stem.split("-")[1]
            file_date = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        except (IndexError, ValueError):
            continue
        if week_start <= file_date <= week_end:
            try:
                data = json.loads(f.read_text())
                for rec in data.get("all_scored", []):
                    rec["_scan_date"] = data.get("scan_timestamp", "")
                    candidates.append(rec)
            except Exception as exc:
                logger.warning(f"Failed to read archive {f}: {exc}")
    return candidates


def cleanup_old_archives(keep_days: int = ARCHIVE_KEEP_DAYS) -> int:
    """Delete archive files older than keep_days. Returns count deleted."""
    if not ARCHIVE_DIR.exists():
        return 0
    cutoff = date.today() - timedelta(days=keep_days)
    deleted = 0
    for f in ARCHIVE_DIR.glob("scored-*.json"):
        try:
            date_str = f.stem.split("-")[1]
            file_date = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            if file_date < cutoff:
                f.unlink()
                deleted += 1
        except (IndexError, ValueError):
            pass
    if deleted:
        logger.info(f"Cleaned up {deleted} old archive files")
    return deleted


# ─── Outcome lookup ─────────────────────────────────────────────────────────────
def fetch_outcome(symbol: str, from_date: date, days: int = OUTCOME_WINDOW) -> Optional[float]:
    """
    Fetch the % price change over `days` trading days starting from `from_date`.
    Returns None if data unavailable (holiday, delisted, error).
    """
    try:
        import yfinance as yf
        end = from_date + timedelta(days=days + 5)  # buffer for weekends/holidays
        hist = yf.Ticker(symbol).history(
            start=from_date.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        if hist is None or hist.empty or len(hist) < 2:
            return None
        closes = hist["Close"].dropna().tolist()
        if len(closes) < 2:
            return None
        n = min(days, len(closes) - 1)
        return round((closes[n] - closes[0]) / closes[0] * 100, 2)
    except Exception as exc:
        logger.warning(f"Outcome fetch failed for {symbol}: {exc}")
        return None


# ─── Feature signature ──────────────────────────────────────────────────────────
def make_feature_signature(rec: dict) -> dict:
    """Build a 3-dimension feature signature for pattern matching."""
    pat = rec.get("patterns") or {}
    rsi  = pat.get("rsi")
    rel  = rec.get("relative_volume")
    mom  = rec.get("change_5d_pct")

    creators = rec.get("supporting_creators") or []
    primary_creator = creators[0].lstrip("@") if creators else "unknown"

    return {
        "rsi_bucket":     rsi_bucket(rsi),
        "momentum_5d":    momentum_bucket(mom),
        "rel_vol_bucket": rel_vol_bucket(rel),
        "creator":        primary_creator,
    }


def signatures_match(a: dict, b: dict) -> bool:
    """True if both signatures share the same RSI bucket, momentum, AND creator (3-of-3)."""
    return (
        a.get("rsi_bucket")  == b.get("rsi_bucket") and
        a.get("momentum_5d") == b.get("momentum_5d") and
        a.get("creator")     == b.get("creator")
    )


# ─── History record construction ────────────────────────────────────────────────
def build_history_record(
    rec: dict,
    week_start: date,
    outcome_pct: Optional[float],
    miss_type: str,
    source: str = "archive_skip",
) -> dict:
    """Convert a scored candidate into a reflect_history.jsonl record."""
    pat      = rec.get("patterns") or {}
    scan_ts  = rec.get("_scan_date", "")
    scan_date = scan_ts[:10] if scan_ts else date.today().isoformat()

    direction     = rec.get("direction", "skip")
    would_have    = rec.get("would_have_direction", "neutral")
    outcome_dir   = would_have if direction == "skip" else direction

    outcome_correct: Optional[bool] = None
    if outcome_pct is not None and outcome_dir in ("call", "put"):
        if outcome_dir == "put":
            outcome_correct = outcome_pct < -OUTCOME_MIN_MOVE
        else:
            outcome_correct = outcome_pct > OUTCOME_MIN_MOVE

    return {
        "week_start":           week_start.isoformat(),
        "scan_date":            scan_date,
        "symbol":               rec.get("symbol", ""),
        "direction":            direction,
        "score":                rec.get("score"),
        "skip_reason":          rec.get("skip_reason") or rec.get("rationale", ""),
        "would_have_direction": would_have,
        "supporting_creators":  rec.get("supporting_creators", []),
        "feature_signature":    make_feature_signature(rec),
        "outcome_5d_pct":       outcome_pct,
        "outcome_correct":      outcome_correct,
        "miss_type":            miss_type,
        "source":               source,
        "amendment_rejected":   False,
    }


# ─── Pattern detection ──────────────────────────────────────────────────────────
def detect_patterns(history: list[dict]) -> list[dict]:
    """
    Find feature signatures that have appeared 3+ times as false skips.
    Deduplicates the same symbol within 14 days.

    Returns list of pattern dicts with matched records.
    """
    # Only consider false skips (skipped + outcome was in the predicted direction)
    false_skips = [
        r for r in history
        if r.get("miss_type") == "false_skip"
        and r.get("outcome_correct") is True
        and not r.get("amendment_rejected", False)
    ]

    # Dedup: same symbol within 14 days counts as one occurrence
    deduped: list[dict] = []
    for r in sorted(false_skips, key=lambda x: x.get("scan_date", "")):
        symbol = r.get("symbol", "")
        scan_d = date.fromisoformat(r.get("scan_date", date.today().isoformat()))
        duplicate = False
        for seen in deduped:
            if seen.get("symbol") == symbol:
                seen_d = date.fromisoformat(seen.get("scan_date", date.today().isoformat()))
                if abs((scan_d - seen_d).days) <= 14:
                    duplicate = True
                    break
        if not duplicate:
            deduped.append(r)

    # Group by feature signature
    patterns: list[dict] = []
    consumed = set()
    for i, r in enumerate(deduped):
        if i in consumed:
            continue
        sig = r.get("feature_signature", {})
        matches = [r]
        for j, other in enumerate(deduped):
            if j <= i or j in consumed:
                continue
            if signatures_match(sig, other.get("feature_signature", {})):
                matches.append(other)
        if len(matches) >= MISS_THRESHOLD:
            for m in matches:
                consumed.add(deduped.index(m))
            patterns.append({
                "signature":   sig,
                "occurrences": len(matches),
                "records":     matches,
                "avg_outcome": round(
                    sum(m.get("outcome_5d_pct") or 0 for m in matches) / len(matches), 2
                ),
            })

    return patterns


# ─── Amendment generation ───────────────────────────────────────────────────────
def generate_amendment(creator_handle: str, pattern: dict) -> str:
    """
    Build a markdown amendment draft from a detected pattern.
    Template-based (no LLM). Human edits prose before applying.
    """
    sig       = pattern["signature"]
    records   = pattern["records"]
    avg_move  = pattern["avg_outcome"]
    n         = pattern["occurrences"]

    evidence_rows = "\n".join(
        f"| {r.get('scan_date','?')} | {r.get('symbol','?')} "
        f"| {r.get('skip_reason','?')[:60]} "
        f"| {r.get('outcome_5d_pct','?'):+.1f}% |"
        for r in records
    )

    return f"""# Framework Amendment Draft: @{creator_handle}
Date: {date.today().isoformat()}
Pattern detected: {n} occurrences of SKIP on RSI={sig.get('rsi_bucket')} + 5d-momentum={sig.get('momentum_5d')} → avg move {avg_move:+.1f}%
Evidence weeks: {', '.join(sorted(set(r.get('week_start','') for r in records)))}

## Proposed addition to "Setup Triggers" section

**[EDIT THIS SECTION — template generated, human prose required]**
When RSI is in the `{sig.get('rsi_bucket')}` range and the stock has shown
`{sig.get('momentum_5d')}` momentum over the prior 5 days, historical data from
this system ({n} occurrences, avg outcome {avg_move:+.1f}%) suggests a potential
reversal opportunity. Consider adding this as a lower-conviction entry condition
(score 0.55–0.65) with tighter position sizing.

## Evidence
| Date | Ticker | Skip reason | Actual move |
|------|--------|-------------|-------------|
{evidence_rows}

## Apply or reject
  python reflect.py apply {creator_handle} --date {date.today().isoformat()}
  python reflect.py reject {creator_handle} --date {date.today().isoformat()} --reason "reason here"

## Notes
- This amendment was auto-generated from pattern detection in reflect_history.jsonl
- Edit the "Proposed addition" prose above before applying
- Applying creates framework-v{{n+1}}.md and archives the prior version
"""


def write_amendment(creator_handle: str, content: str) -> Path:
    amend_dir = CREATORS_DIR / creator_handle / "amendments"
    amend_dir.mkdir(parents=True, exist_ok=True)
    path = amend_dir / f"amendment-{date.today().isoformat()}.md"
    path.write_text(content)
    return path


def load_amendments(creator_handle: Optional[str] = None) -> list[dict]:
    """Load all pending (non-rejected) amendment files."""
    amendments = []
    handles = [creator_handle] if creator_handle else [
        d.name for d in CREATORS_DIR.iterdir() if d.is_dir()
    ]
    for handle in handles:
        amend_dir = CREATORS_DIR / handle / "amendments"
        if not amend_dir.exists():
            continue
        for f in sorted(amend_dir.glob("amendment-*.md")):
            reject_marker = f.parent / f"{f.stem}.rejected"
            if not reject_marker.exists():
                amendments.append({
                    "handle": handle,
                    "date":   f.stem.replace("amendment-", ""),
                    "path":   f,
                })
    return amendments


# ─── Framework version management ───────────────────────────────────────────────
def _framework_version(creator_dir: Path) -> int:
    """Return the highest existing framework version number."""
    versions = []
    for f in creator_dir.glob("framework-v*.md"):
        m = re.search(r"framework-v(\d+)\.md$", f.name)
        if m and not f.stem.endswith(".archived"):
            versions.append(int(m.group(1)))
    return max(versions) if versions else 0


def apply_amendment(creator_handle: str, amendment_date: str) -> None:
    """
    Merge an amendment into the creator framework.
    Creates framework-v{n+1}.md, archives the previous version.
    """
    creator_dir = CREATORS_DIR / creator_handle
    amend_path  = creator_dir / "amendments" / f"amendment-{amendment_date}.md"

    if not amend_path.exists():
        print(f"  ✗ Amendment not found: {amend_path}")
        sys.exit(1)

    current_ver = _framework_version(creator_dir)
    if current_ver == 0:
        print(f"  ✗ No framework-v*.md found for @{creator_handle}")
        sys.exit(1)

    current_fw  = creator_dir / f"framework-v{current_ver}.md"
    new_ver     = current_ver + 1
    new_fw      = creator_dir / f"framework-v{new_ver}.md"
    archived_fw = creator_dir / f"framework-v{current_ver}.archived-{date.today().isoformat()}.md"

    # Read amendment (strip header boilerplate, get proposed section)
    amend_text = amend_path.read_text()
    fw_text    = current_fw.read_text()

    # Find insertion point: after "## Setup Triggers" section
    target_section = "## Setup Triggers"
    next_section   = re.compile(r"^## ", re.MULTILINE)

    if target_section in fw_text:
        idx = fw_text.index(target_section)
        # Find next ## after target section
        m = next_section.search(fw_text, idx + len(target_section))
        insert_at = m.start() if m else len(fw_text)
        proposed_block = _extract_proposed_block(amend_text)
        updated = fw_text[:insert_at] + "\n" + proposed_block + "\n\n" + fw_text[insert_at:]
    else:
        # Fallback: append at end
        proposed_block = _extract_proposed_block(amend_text)
        updated = fw_text + "\n\n" + proposed_block

    # Append changelog
    updated += f"\n\n## Changelog\n- {date.today().isoformat()}: Applied amendment from {amend_path.name}\n"

    # Archive current, write new version
    current_fw.rename(archived_fw)
    new_fw.write_text(updated)

    print(f"  ✓ Created framework-v{new_ver}.md for @{creator_handle}")
    print(f"  ✓ Archived framework-v{current_ver}.md → {archived_fw.name}")
    logger.info(f"Applied amendment {amend_path} → {new_fw}")


def _extract_proposed_block(amend_text: str) -> str:
    """Extract the proposed addition block from an amendment file."""
    m = re.search(
        r"## Proposed addition.*?\n\n(.*?)(?=\n## Evidence|\Z)",
        amend_text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return "[Amendment content could not be extracted — edit framework-v{n+1}.md manually]"


def reject_amendment(creator_handle: str, amendment_date: str, reason: str) -> None:
    """Mark an amendment as rejected (creates a .rejected marker file)."""
    amend_path  = CREATORS_DIR / creator_handle / "amendments" / f"amendment-{amendment_date}.md"
    reject_path = amend_path.parent / f"amendment-{amendment_date}.rejected"

    if not amend_path.exists():
        print(f"  ✗ Amendment not found: {amend_path}")
        sys.exit(1)

    reject_path.write_text(json.dumps({
        "rejected_at": datetime.now(timezone.utc).isoformat(),
        "reason":      reason,
    }, indent=2))

    # Mark records in history so this pattern won't regenerate
    history = load_history()
    updated = False
    with open(HISTORY_PATH, "w") as f:
        for r in history:
            week_dates = {rec.get("week_start") for rec in history}
            if (creator_handle in (r.get("supporting_creators") or []) or
                    creator_handle in "".join(r.get("supporting_creators") or [])):
                r["amendment_rejected"] = True
                updated = True
            f.write(json.dumps(r, default=str) + "\n")

    print(f"  ✓ Amendment rejected: {amend_path.name}")
    if reason:
        print(f"    Reason: {reason}")
    logger.info(f"Rejected amendment {amend_path} — {reason}")


# ─── Weekly processing ──────────────────────────────────────────────────────────
def _week_has_archive_records(history: list[dict], week_start: date) -> bool:
    """True if archive skip/journal records already exist for this week."""
    ws = week_start.isoformat()
    return any(
        r.get("week_start") == ws and r.get("source", "archive_skip") != "sent_alert"
        for r in history
    )


def _sent_alert_processed(
    history: list[dict], week_start: date, symbol: str, direction: str,
) -> bool:
    """True when a final (full-hold) score is already stored."""
    ws = week_start.isoformat()
    for r in history:
        if (
            r.get("week_start") == ws
            and r.get("source") == "sent_alert"
            and r.get("symbol", "").upper() == symbol.upper()
            and r.get("direction", "").lower() == direction.lower()
        ):
            if r.get("outcome_final"):
                return True
            if r.get("outcome_final") is None and r.get("outcome_option_pnl_pct") is not None:
                # Legacy rows before outcome_final existed — treat as final
                return True
    return False


def _direction_correct(direction: str, outcome_pct: float) -> bool:
    if direction == "call":
        return outcome_pct > 0
    return outcome_pct < 0


def process_sent_alerts(
    week_start: date, force: bool = False, config: dict | None = None,
) -> list[dict]:
    """
    Evaluate emailed swing alerts from sent_history.json.
    Scores option P&L mark-to-market through the latest session; refreshes
    interim scores until the full hold window completes.
    """
    if evaluate_swing_alert is None:
        logger.warning("option_outcome not available — skipping sent alerts")
        return []

    history = load_history()
    sent = load_week_sent_alerts(week_start)
    if not sent:
        logger.info(f"No sent alerts for week {week_start}")
        return []

    new_records: list[dict] = []
    for alert in sent:
        symbol    = alert["symbol"]
        direction = alert["direction"]
        if not force and _sent_alert_processed(history, week_start, symbol, direction):
            continue

        archive = find_archive_alert_on_date(symbol, direction, alert["sent_date"]) or {
            "symbol":               symbol,
            "direction":            direction,
            "score":                alert.get("score"),
            "rationale":            "Emailed alert (no matching archive record)",
            "supporting_creators":  [],
            "would_have_direction": direction,
            "_scan_date":           alert["sent_at"].isoformat(),
        }

        time.sleep(0.1)
        scored = evaluate_swing_alert(archive, alert["sent_date"], config)

        if scored.get("outcome_option_pnl_pct") is None:
            logger.info(f"No option price for {symbol} {direction} — skip")
            continue

        if scored.get("outcome_interim"):
            logger.info(
                f"Interim swing score for {symbol} {direction}: "
                f"{scored.get('outcome_option_pnl_pct'):+.1f}% "
                f"(through {scored.get('outcome_as_of')})"
            )

        rec = build_history_record(
            archive, week_start,
            scored.get("outcome_5d_pct"),
            scored["miss_type"],
            source="sent_alert",
        )
        rec = _apply_option_outcome(rec, scored)
        rec["scan_date"] = alert["sent_date"].isoformat()
        new_records.append(rec)

    return new_records


def process_archive_week(week_start: date, force: bool = False) -> list[dict]:
    """Load archived scan candidates (skips + journaled takes) for the week."""
    history = load_history()
    if _week_has_archive_records(history, week_start) and not force:
        logger.info(
            f"Archive week {week_start} already processed — skipping archive pass"
        )
        return []

    candidates = load_week_archives(week_start)
    if not candidates:
        logger.info(f"No archive files for week {week_start}")
        return []

    journal   = load_journal()
    journal_symbols = {
        e["symbol"].upper()
        for e in journal
        if e.get("entry_date", "")[:10] >= week_start.isoformat()
    }

    new_records: list[dict] = []
    for rec in candidates:
        symbol    = (rec.get("symbol") or "").upper()
        direction = rec.get("direction", "skip")
        scan_ts   = rec.get("_scan_date", "")
        scan_date = date.fromisoformat(scan_ts[:10]) if scan_ts else week_start

        if direction == "skip":
            outcome = None
            if symbol not in journal_symbols:
                time.sleep(0.1)
                outcome = fetch_outcome(symbol, scan_date)
            would_have = rec.get("would_have_direction", "neutral")
            is_miss = (
                outcome is not None and
                would_have in ("call", "put") and
                (
                    (would_have == "put"  and outcome < -OUTCOME_MIN_MOVE) or
                    (would_have == "call" and outcome >  OUTCOME_MIN_MOVE)
                )
            )
            miss_type = "false_skip" if is_miss else "correct_skip"
            new_records.append(
                build_history_record(rec, week_start, outcome, miss_type, "archive_skip")
            )

        elif symbol in journal_symbols:
            je = next(
                (e for e in journal
                 if e["symbol"].upper() == symbol
                 and e.get("entry_date", "")[:10] >= week_start.isoformat()),
                None,
            )
            if je:
                outcome = je.get("underlying_move") or je.get("pnl_pct")
                r_val   = je.get("pnl_r")
                miss_type = "false_take" if (r_val is not None and r_val < 0) else "correct_take"
                new_records.append(
                    build_history_record(rec, week_start, outcome, miss_type, "journal_take")
                )

    return new_records


def _intraday_alert_processed(
    history: list[dict], week_start: date, scan_timestamp: str,
) -> bool:
    ws = week_start.isoformat()
    return any(
        r.get("week_start") == ws
        and r.get("source") == "intraday_0dte"
        and r.get("scan_timestamp") == scan_timestamp
        for r in history
    )


def _apply_option_outcome(rec: dict, scored: dict) -> dict:
    rec["miss_type"] = scored["miss_type"]
    rec["outcome_5d_pct"] = scored.get("outcome_5d_pct")
    rec["outcome_option_pnl_pct"] = scored.get("outcome_option_pnl_pct")
    rec["outcome_underlying_pct"] = scored.get("outcome_underlying_pct")
    rec["option_entry_mid"] = scored.get("option_entry_mid")
    rec["option_exit_mid"] = scored.get("option_exit_mid")
    rec["option_exit_date"] = scored.get("option_exit_date")
    rec["option_target_exit_date"] = scored.get("option_target_exit_date")
    rec["outcome_as_of"] = scored.get("outcome_as_of")
    rec["contract_label"] = scored.get("contract_label")
    rec["outcome_final"] = scored.get("outcome_final", False)
    rec["outcome_interim"] = scored.get("outcome_interim", False)
    rec["outcome_correct"] = scored.get("miss_type") == "correct_take"
    return rec


def _apply_intraday_outcome(rec: dict, scored: dict) -> dict:
    return _apply_option_outcome(rec, scored)


def process_intraday_for_date(
    target: date, force: bool = False, config: dict | None = None,
) -> list[dict]:
    """
    Process intraday 0DTE entry alerts for a single day.

    P&L uses the matched exit alert when present; entries with no exit alert
    are scored as full loss (no_exit_loss_pct, default -100%).
    """
    if load_entry_alerts_for_date is None or evaluate_intraday_alert is None:
        logger.warning("intraday_0dte module not available — skipping")
        return []

    week_start = monday_of_week(target)
    history = load_history()
    day_alerts = load_alerts_for_date(target) if load_alerts_for_date else []
    entries = load_entry_alerts_for_date(target)
    if not entries:
        logger.info(f"No intraday entry alerts on {target}")
        return []

    new_records: list[dict] = []
    for entry in entries:
        scan_ts = entry.get("scan_timestamp", "")
        if not force and _intraday_alert_processed(history, week_start, scan_ts):
            continue

        exit_alert = (
            find_exit_for_entry(entry, day_alerts)
            if find_exit_for_entry else None
        )
        time.sleep(0.1)
        scored = evaluate_intraday_alert(entry, config, exit_alert=exit_alert)

        rec = build_history_record(
            entry, week_start, scored.get("outcome_5d_pct"), scored["miss_type"],
            source="intraday_0dte",
        )
        rec = _apply_intraday_outcome(rec, scored)
        rec["scan_date"] = target.isoformat()
        rec["scan_timestamp"] = scan_ts
        rec["pipeline"] = "intraday_0dte"
        rec["had_exit_alert"] = exit_alert is not None
        rec["exit_timestamp"] = scored.get("exit_timestamp")
        rec["outcome_no_exit"] = scored.get("outcome_no_exit", False)
        new_records.append(rec)

    return new_records


def process_intraday_0dte_week(
    week_start: date, force: bool = False, config: dict | None = None,
) -> list[dict]:
    """
    Evaluate logged 0–1 DTE intraday entry alerts for the week.

    Uses exit-alert pricing when matched; no exit → full loss.
    """
    if load_week_intraday_alerts is None or evaluate_intraday_alert is None:
        logger.warning("intraday_0dte module not available — skipping intraday pass")
        return []

    history = load_history()
    entries = load_week_intraday_alerts(week_start)
    if not entries:
        logger.info(f"No intraday 0DTE alerts for week {week_start}")
        return []

    new_records: list[dict] = []
    for entry in entries:
        scan_ts = entry.get("scan_timestamp", "")
        if not force and _intraday_alert_processed(history, week_start, scan_ts):
            continue

        try:
            alert_date = date.fromisoformat(scan_ts[:10])
        except (ValueError, TypeError):
            alert_date = week_start

        day_alerts = load_alerts_for_date(alert_date) if load_alerts_for_date else []
        exit_alert = (
            find_exit_for_entry(entry, day_alerts)
            if find_exit_for_entry else None
        )

        time.sleep(0.1)
        scored = evaluate_intraday_alert(entry, config, exit_alert=exit_alert)

        rec = build_history_record(
            entry, week_start, scored.get("outcome_5d_pct"), scored["miss_type"],
            source="intraday_0dte",
        )
        rec = _apply_intraday_outcome(rec, scored)
        rec["scan_date"] = alert_date.isoformat()
        rec["scan_timestamp"] = scan_ts
        rec["pipeline"] = "intraday_0dte"
        rec["had_exit_alert"] = exit_alert is not None
        rec["exit_timestamp"] = scored.get("exit_timestamp")
        rec["outcome_no_exit"] = scored.get("outcome_no_exit", False)
        new_records.append(rec)

    return new_records


def _load_config() -> dict:
    cfg_path = BASE_DIR / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _persist_week_records(records: list[dict]) -> None:
    sent = [r for r in records if r.get("source") == "sent_alert"]
    other = [r for r in records if r.get("source") != "sent_alert"]
    if sent:
        upsert_sent_alert_records(sent)
    if other:
        append_history(other)


def process_week(week_start: date, force: bool = False, config: dict | None = None) -> list[dict]:
    """
    Process archive skips/takes, emailed alerts, and intraday 0DTE alerts.
    Each sub-pass is idempotent unless force=True.
    """
    config = config or _load_config()
    archive_records  = process_archive_week(week_start, force=force)
    sent_records     = process_sent_alerts(week_start, force=force, config=config)
    intraday_records = process_intraday_0dte_week(week_start, force=force, config=config)
    return archive_records + sent_records + intraday_records


# ─── CLI Commands ───────────────────────────────────────────────────────────────
def cmd_report(args: argparse.Namespace) -> None:
    """Print this week's performance summary."""
    week_start = monday_of_week(date.today())
    history    = load_history()
    this_week  = [r for r in history if r.get("week_start") == week_start.isoformat()]

    if not this_week:
        print(f"\n  No reflection data for week of {week_start}.")
        print("  Run: reflect --auto   (or wait until Friday 5pm)\n")
        return

    taken  = [r for r in this_week if r["miss_type"] in ("correct_take", "false_take")]
    skips  = [r for r in this_week if r["miss_type"] in ("correct_skip", "false_skip")]
    wins   = [r for r in taken if r["miss_type"] == "correct_take"]
    f_skips = [r for r in skips if r["miss_type"] == "false_skip"]
    c_skips = [r for r in skips if r["miss_type"] == "correct_skip"]
    sent_taken = [r for r in taken if r.get("source") == "sent_alert"]
    journal_taken = [r for r in taken if r.get("source") == "journal_take"]
    intraday_taken = [r for r in taken if r.get("source") == "intraday_0dte"]

    win_rate = len(wins) / len(taken) * 100 if taken else 0
    skip_acc = len(c_skips) / len(skips) * 100 if skips else 0

    print(f"\n{'═'*60}")
    print(f"  WEEKLY REFLECTION  │  week of {week_start}")
    print(f"{'═'*60}")
    print(f"  Trades taken : {len(taken):>3}   Win rate : {win_rate:.0f}%")
    if sent_taken or journal_taken or intraday_taken:
        print(
            f"    emailed    : {len(sent_taken):>3}   journaled : {len(journal_taken):>3}"
            f"   0DTE intraday : {len(intraday_taken):>3}"
        )
    print(f"  Skips        : {len(skips):>3}   Skip accuracy : {skip_acc:.0f}%")
    print(f"  False skips  : {len(f_skips):>3}   (skips that would have been profitable)")
    print()

    if sent_taken:
        print("  Emailed swing alerts (option P&L; underlying in parens):")
        for r in sorted(sent_taken, key=lambda x: x.get("scan_date", "")):
            icon = "✓" if r["miss_type"] == "correct_take" else "✗"
            pnl = r.get("outcome_option_pnl_pct", r.get("outcome_5d_pct"))
            und = r.get("outcome_underlying_pct")
            pnl_s = f"{pnl:+.1f}%" if pnl is not None else "n/a"
            und_s = f"{und:+.1f}%" if und is not None else "n/a"
            interim = ""
            if r.get("outcome_interim"):
                interim = f" [interim→{r.get('option_target_exit_date', '?')}]"
            print(f"    {icon} {r['symbol']:<6} {r['direction']:<4}  option {pnl_s:>8}  "
                  f"(underlying {und_s}){interim}  score={r.get('score')}")
        print()

    if intraday_taken:
        print("  0–1 DTE intraday (exit-alert P&L; no exit = full loss):")
        for r in sorted(intraday_taken, key=lambda x: x.get("scan_date", "")):
            icon = "✓" if r["miss_type"] == "correct_take" else "✗"
            pnl = r.get("outcome_option_pnl_pct", r.get("outcome_5d_pct"))
            und = r.get("outcome_underlying_pct")
            pnl_s = f"{pnl:+.1f}%" if pnl is not None else "n/a"
            und_s = f"{und:+.2f}%" if und is not None else "n/a"
            exit_note = ""
            if r.get("outcome_no_exit"):
                exit_note = "  [no exit alert — full loss]"
            elif r.get("exit_timestamp"):
                exit_note = f"  [exit {r['exit_timestamp'][:16]}]"
            print(f"    {icon} {r['symbol']:<6} {r['direction']:<4}  option {pnl_s:>8}  "
                  f"(underlying {und_s}){exit_note}  score={r.get('score')}")
        print()

    # Per-creator breakdown
    creator_stats: dict[str, dict] = {}
    for r in this_week:
        for c in r.get("supporting_creators", []):
            handle = c.lstrip("@")
            creator_stats.setdefault(handle, {"taken": 0, "wins": 0, "false_skips": 0})
            if r["miss_type"] in ("correct_take", "false_take"):
                creator_stats[handle]["taken"] += 1
                if r["miss_type"] == "correct_take":
                    creator_stats[handle]["wins"] += 1
            elif r["miss_type"] == "false_skip":
                creator_stats[handle]["false_skips"] += 1

    if creator_stats:
        print("  Creator attribution:")
        for handle, st in sorted(creator_stats.items()):
            wr = st["wins"] / st["taken"] * 100 if st["taken"] else 0
            print(f"    @{handle:<16}  taken={st['taken']}  win%={wr:.0f}%  false_skips={st['false_skips']}")

    # Pattern accumulation
    all_history = load_history()
    patterns    = detect_patterns(all_history)
    print(f"\n  Accumulated patterns: {len(patterns)} above threshold ({MISS_THRESHOLD}+ occurrences)")
    pending = load_amendments()
    print(f"  Pending amendments:   {len(pending)}")
    print(f"{'─'*60}\n")


def cmd_patterns(args: argparse.Namespace) -> None:
    """Print accumulated miss patterns across all weeks."""
    history  = load_history()
    patterns = detect_patterns(history)

    # Also show candidates (2 occurrences)
    false_skips = [r for r in history if r.get("miss_type") == "false_skip"
                   and r.get("outcome_correct") is True
                   and not r.get("amendment_rejected", False)]

    print(f"\n{'═'*60}")
    print(f"  PATTERN LEDGER  │  {len(history)} total records")
    print(f"{'═'*60}")

    if not false_skips:
        print("  No false skips recorded yet.")
        print("  Run reflect --auto after Fridays to build history.\n")
        return

    if patterns:
        print(f"\n  ★ Amendment candidates ({len(patterns)} patterns ≥ {MISS_THRESHOLD} occurrences):")
        for i, p in enumerate(patterns, 1):
            sig = p["signature"]
            print(f"\n  [{i}] @{sig.get('creator')}  RSI={sig.get('rsi_bucket')}  "
                  f"momentum={sig.get('momentum_5d')}  occurrences={p['occurrences']}")
            print(f"      avg outcome: {p['avg_outcome']:+.1f}%")
            for r in p["records"]:
                print(f"      {r.get('scan_date','?')}  {r.get('symbol','?')}  "
                      f"outcome={r.get('outcome_5d_pct','?'):+.1f}%")
    else:
        print(f"\n  No patterns at threshold ({MISS_THRESHOLD}+) yet.")

    # Show building candidates (2 occurrences)
    print(f"\n  Building candidates (2 occurrences — need 1 more to trigger):")
    _shown = set()
    for r in false_skips:
        sig = r.get("feature_signature", {})
        key = (sig.get("rsi_bucket"), sig.get("momentum_5d"), sig.get("creator"))
        matches = [x for x in false_skips
                   if signatures_match(x.get("feature_signature", {}), sig)]
        if len(matches) == 2 and key not in _shown:
            _shown.add(key)
            print(f"    @{sig.get('creator')}  RSI={sig.get('rsi_bucket')}  "
                  f"momentum={sig.get('momentum_5d')}  (2 occurrences)")

    print(f"\n{'─'*60}\n")


def cmd_amendments(args: argparse.Namespace) -> None:
    """List pending draft amendments."""
    amendments = load_amendments()
    print(f"\n{'═'*60}")
    print(f"  PENDING AMENDMENTS  │  {len(amendments)} draft(s)")
    print(f"{'═'*60}")
    if not amendments:
        print("  No pending amendments.\n")
        return
    for a in amendments:
        print(f"\n  @{a['handle']}  {a['date']}")
        print(f"  File: {a['path']}")
        print(f"  → reflect apply {a['handle']} --date {a['date']}")
        print(f"  → reflect reject {a['handle']} --date {a['date']} --reason \"...\"")
    print()


def cmd_apply(args: argparse.Namespace) -> None:
    """Merge a draft amendment into the creator framework."""
    handle = args.handle
    adate  = args.date or date.today().isoformat()
    print(f"\n  Applying amendment for @{handle} dated {adate}...")
    apply_amendment(handle, adate)


def cmd_reject(args: argparse.Namespace) -> None:
    """Mark a draft amendment as rejected."""
    handle = args.handle
    adate  = args.date
    reason = args.reason or ""
    if not adate:
        print("  ✗ --date is required for reject")
        sys.exit(1)
    print(f"\n  Rejecting amendment for @{handle} dated {adate}...")
    reject_amendment(handle, adate, reason)


def cmd_intraday_daily(args: argparse.Namespace) -> None:
    """End-of-day 0DTE review: score today's alerts, append history, email summary."""
    from market_calendar import is_trading_day

    target = date.today()
    if not is_trading_day(target) and not getattr(args, "force", False):
        logger.info(f"NYSE closed on {target} — skipping intraday daily reflect")
        return

    logger.info(f"=== reflect --intraday-daily ({target}) ===")
    config = _load_config()
    new_records = process_intraday_for_date(
        target, force=getattr(args, "force", False), config=config,
    )
    if new_records:
        append_history(new_records)
        logger.info(f"Appended {len(new_records)} intraday record(s)")

    try:
        from notify import send_intraday_reflect_summary
        send_intraday_reflect_summary(new_records, target, config)
    except Exception as exc:
        logger.warning(f"Intraday reflect email failed: {exc}")

    if new_records:
        wins = sum(1 for r in new_records if r.get("miss_type") == "correct_take")
        print(f"\n  0DTE daily review: {wins}/{len(new_records)} correct "
              f"({target.isoformat()})\n")
    else:
        print(f"\n  0DTE daily review: no alerts to score ({target.isoformat()})\n")

    logger.info("=== reflect --intraday-daily complete ===")


def cmd_weekly(args: argparse.Namespace) -> None:
    """Friday weekly review: score past week's swing emailed alerts + email summary."""
    week_start = monday_of_week(date.today())
    config = _load_config()

    logger.info(f"=== reflect --weekly (week of {week_start}) ===")

    new_records = process_week(week_start, force=getattr(args, "force", False), config=config)
    if new_records:
        _persist_week_records(new_records)
        logger.info(f"Saved {len(new_records)} record(s) to reflect_history.jsonl")

    history = load_history()
    week_sent = [
        r for r in history
        if r.get("week_start") == week_start.isoformat()
        and r.get("source") == "sent_alert"
    ]
    sent_raw = load_week_sent_alerts(week_start)
    scored_keys = {(r["symbol"], r["direction"]) for r in week_sent}
    pending_sent = [
        a for a in sent_raw
        if (a["symbol"], a["direction"]) not in scored_keys
    ]

    try:
        from notify import send_weekly_swing_reflect_summary
        send_weekly_swing_reflect_summary(
            week_sent, week_start, config, pending_alerts=pending_sent,
        )
    except Exception as exc:
        logger.warning(f"Weekly swing reflect email failed: {exc}")

    if week_sent:
        wins = sum(1 for r in week_sent if r.get("miss_type") == "correct_take")
        interim_n = sum(1 for r in week_sent if r.get("outcome_interim"))
        interim_tag = f", {interim_n} interim" if interim_n else ""
        print(f"\n  Weekly swing review: {wins}/{len(week_sent)} wins "
              f"(week of {week_start}{interim_tag})\n")
        for r in sorted(week_sent, key=lambda x: x.get("scan_date", "")):
            icon = "✓" if r.get("miss_type") == "correct_take" else "✗"
            pnl = r.get("outcome_option_pnl_pct", r.get("outcome_5d_pct"))
            und = r.get("outcome_underlying_pct")
            pnl_s = f"{pnl:+.1f}%" if pnl is not None else "n/a"
            und_s = f"{und:+.1f}%" if und is not None else "n/a"
            interim = ""
            if r.get("outcome_interim"):
                interim = f" [interim through {r.get('outcome_as_of', '?')}]"
            print(f"    {icon} {r.get('scan_date','?')} {r.get('symbol','?'):<6} "
                  f"{r.get('direction','?'):<4}  option {pnl_s:>8}  (underlying {und_s})"
                  f"{interim}  score={r.get('score')}")
        print()
    elif pending_sent:
        print(f"\n  Weekly swing review: {len(pending_sent)} alert(s) could not be priced "
              f"(week of {week_start})\n")
        for a in pending_sent:
            print(f"    ⏳ {a['sent_date']} {a['symbol']:<6} {a['direction']:<4}  "
                  f"score={a.get('score')}")
        print()
    else:
        print(f"\n  Weekly swing review: no emailed alerts (week of {week_start})\n")

    # Pattern detection for swing pipeline (same as --auto)
    history = load_history()
    patterns = detect_patterns(history)
    for p in patterns:
        creator = p["signature"].get("creator", "")
        if not creator or creator == "unknown":
            continue
        existing = load_amendments(creator)
        already = any(a["date"] == date.today().isoformat() for a in existing)
        if already:
            continue
        content = generate_amendment(creator, p)
        path = write_amendment(creator, content)
        print(f"  📝 Amendment draft written: {path}")

    deleted = cleanup_old_archives()
    if deleted:
        print(f"  🗑  Cleaned up {deleted} old archive file(s)")

    logger.info("=== reflect --weekly complete ===")


def cmd_auto(args: argparse.Namespace) -> None:
    """
    Non-interactive full run for launchd.
    Processes the current week, updates history, generates amendments if threshold met.
    Idempotent — safe to re-run.
    """
    week_start = monday_of_week(date.today())
    config = _load_config()
    logger.info(f"=== reflect --auto starting (week {week_start}) ===")

    new_records = process_week(week_start, force=getattr(args, "force", False), config=config)
    if new_records:
        _persist_week_records(new_records)
        logger.info(f"Saved {len(new_records)} records to reflect_history.jsonl")
    else:
        logger.info("No new records to append (already processed or no archives)")

    # Detect patterns and generate amendments
    history  = load_history()
    patterns = detect_patterns(history)
    for p in patterns:
        creator = p["signature"].get("creator", "")
        if not creator or creator == "unknown":
            continue
        # Don't regenerate if an amendment for this creator already exists today
        existing = load_amendments(creator)
        already  = any(a["date"] == date.today().isoformat() for a in existing)
        if already:
            logger.info(f"Amendment for @{creator} already exists today — skipping")
            continue
        content  = generate_amendment(creator, p)
        path     = write_amendment(creator, content)
        print(f"  📝 Amendment draft written: {path}")
        logger.info(f"Amendment written for @{creator}: {path}")

    # Print brief summary
    cmd_report(args)

    # Clean up old archive files
    deleted = cleanup_old_archives()
    if deleted:
        print(f"  🗑  Cleaned up {deleted} old archive file(s)")

    logger.info("=== reflect --auto complete ===")


# ─── Argument parsing ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weekly self-reflection & framework evolution for the trading pipeline"
    )
    parser.add_argument("--auto",  action="store_true",
                        help="Non-interactive full run (for launchd)")
    parser.add_argument("--intraday-daily", action="store_true",
                        help="End-of-day 0DTE alert review + email summary (trading days)")
    parser.add_argument("--weekly", action="store_true",
                        help="Weekly swing alert review + email summary (Fridays on NAS)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess current week even if already done")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("report", help="This week's performance summary")
    sub.add_parser("patterns", help="Accumulated pattern ledger")
    sub.add_parser("amendments", help="List pending draft amendments")

    p_apply = sub.add_parser("apply", help="Merge a draft amendment into the framework")
    p_apply.add_argument("handle", help="Creator handle (e.g. kpak82)")
    p_apply.add_argument("--date", default=None, help="Amendment date (YYYY-MM-DD)")

    p_reject = sub.add_parser("reject", help="Mark a draft amendment as rejected")
    p_reject.add_argument("handle", help="Creator handle (e.g. kpak82)")
    p_reject.add_argument("--date", required=True, help="Amendment date (YYYY-MM-DD)")
    p_reject.add_argument("--reason", default="", help="Reason for rejection")

    args = parser.parse_args()

    if args.intraday_daily:
        cmd_intraday_daily(args)
    elif args.weekly:
        cmd_weekly(args)
    elif args.auto or args.command is None:
        cmd_auto(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "patterns":
        cmd_patterns(args)
    elif args.command == "amendments":
        cmd_amendments(args)
    elif args.command == "apply":
        cmd_apply(args)
    elif args.command == "reject":
        cmd_reject(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
