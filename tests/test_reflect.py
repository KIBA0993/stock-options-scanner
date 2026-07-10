"""
test_reflect.py — Unit tests for reflect.py (Weekly Self-Reflection Loop)

Coverage:
  - History I/O (load/append)
  - Feature signature construction and matching
  - Outcome correctness logic
  - History record construction
  - Pattern detection (threshold, dedup, signature match)
  - Amendment generation text structure
  - Amendment apply / reject wiring (file creation)
  - Framework version detection
  - Archive loading and week filtering
  - Idempotency of process_week()
  - cleanup_old_archives()
  - rsi_bucket / momentum_bucket / rel_vol_bucket (utils)
  - Regression guards: orchestrate + scanner smoke imports
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the trading directory is on the import path
TRADING_DIR = Path.home() / "trading"
if str(TRADING_DIR) not in sys.path:
    sys.path.insert(0, str(TRADING_DIR))

import reflect
import utils


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_trading(tmp_path, monkeypatch):
    """Redirect all reflect.py path constants to tmp_path."""
    monkeypatch.setattr(reflect, "BASE_DIR",     tmp_path)
    monkeypatch.setattr(reflect, "DATA_DIR",     tmp_path / "data")
    monkeypatch.setattr(reflect, "ARCHIVE_DIR",  tmp_path / "data" / "archive")
    monkeypatch.setattr(reflect, "LOG_DIR",      tmp_path / "logs")
    monkeypatch.setattr(reflect, "CREATORS_DIR", tmp_path / "creators")
    monkeypatch.setattr(reflect, "JOURNAL_PATH", tmp_path / "data" / "trade_journal.jsonl")
    monkeypatch.setattr(reflect, "HISTORY_PATH", tmp_path / "data" / "reflect_history.jsonl")
    monkeypatch.setattr(utils, "DATA_DIR",       tmp_path / "data")
    monkeypatch.setattr(utils, "SENT_PATH",      tmp_path / "data" / "sent_history.json")
    monkeypatch.setattr(utils, "ARCHIVE_DIR",    tmp_path / "data" / "archive")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "creators").mkdir(parents=True, exist_ok=True)
    return tmp_path


def make_skip_record(
    symbol="AAPL",
    rsi=72.0,
    mom=6.5,
    rel_vol=3.1,
    creator="kpak82",
    would_have="put",
    scan_date: str | None = None,
    week_start: str | None = None,
    outcome=None,
    miss_type="false_skip",
    outcome_correct=True,
) -> dict:
    today = date.today()
    ws    = week_start or (today - timedelta(days=today.weekday())).isoformat()
    sd    = scan_date or today.isoformat()
    return {
        "week_start":           ws,
        "scan_date":            sd,
        "symbol":               symbol,
        "direction":            "skip",
        "score":                0.55,
        "skip_reason":          "RSI elevated",
        "would_have_direction": would_have,
        "supporting_creators":  [creator],
        "feature_signature":    {
            "rsi_bucket":     utils.rsi_bucket(rsi),
            "momentum_5d":    utils.momentum_bucket(mom),
            "rel_vol_bucket": utils.rel_vol_bucket(rel_vol),
            "creator":        creator,
        },
        "outcome_5d_pct":       outcome,
        "outcome_correct":      outcome_correct,
        "miss_type":            miss_type,
        "amendment_rejected":   False,
    }


# ─── utils.py bucket functions ─────────────────────────────────────────────────

class TestRsiBucket:
    def test_extreme_oversold(self):
        assert utils.rsi_bucket(10) == "extreme_oversold"

    def test_low(self):
        assert utils.rsi_bucket(25) == "low"

    def test_neutral(self):
        assert utils.rsi_bucket(50) == "neutral"

    def test_elevated(self):
        assert utils.rsi_bucket(75) == "elevated_70_80"

    def test_extreme_overbought(self):
        assert utils.rsi_bucket(85) == "extreme_overbought"

    def test_none(self):
        assert utils.rsi_bucket(None) == "unknown"


class TestMomentumBucket:
    def test_flat(self):
        assert utils.momentum_bucket(1.0) == "flat"

    def test_mild(self):
        assert utils.momentum_bucket(3.5) == "mild"

    def test_extended(self):
        assert utils.momentum_bucket(7.0) == "extended"

    def test_parabolic(self):
        assert utils.momentum_bucket(12.0) == "parabolic"

    def test_negative_extended(self):
        assert utils.momentum_bucket(-7.0) == "extended"

    def test_none(self):
        assert utils.momentum_bucket(None) == "unknown"


class TestRelVolBucket:
    def test_normal(self):
        assert utils.rel_vol_bucket(1.5) == "normal"

    def test_elevated(self):
        assert utils.rel_vol_bucket(2.5) == "elevated"

    def test_extreme(self):
        assert utils.rel_vol_bucket(5.0) == "extreme"

    def test_none(self):
        assert utils.rel_vol_bucket(None) == "unknown"


# ─── History I/O ───────────────────────────────────────────────────────────────

class TestHistoryIO:
    def test_load_empty(self, tmp_trading):
        assert reflect.load_history() == []

    def test_append_and_reload(self, tmp_trading):
        records = [make_skip_record("AAPL"), make_skip_record("NVDA")]
        reflect.append_history(records)
        loaded = reflect.load_history()
        assert len(loaded) == 2
        assert loaded[0]["symbol"] == "AAPL"
        assert loaded[1]["symbol"] == "NVDA"

    def test_append_is_additive(self, tmp_trading):
        reflect.append_history([make_skip_record("AAPL")])
        reflect.append_history([make_skip_record("NVDA")])
        assert len(reflect.load_history()) == 2

    def test_load_skips_invalid_json(self, tmp_trading):
        path = tmp_trading / "data" / "reflect_history.jsonl"
        path.write_text('{"symbol": "AAPL"}\nNOT_JSON\n{"symbol": "NVDA"}\n')
        records = reflect.load_history()
        assert len(records) == 2


# ─── Feature signature ──────────────────────────────────────────────────────────

class TestFeatureSignature:
    def _make_candidate(self, rsi=75, mom=6.5, rel=3.1, creator="kpak82"):
        return {
            "patterns":  {"rsi": rsi},
            "relative_volume": rel,
            "change_5d_pct":   mom,
            "supporting_creators": [creator],
        }

    def test_builds_correctly(self):
        sig = reflect.make_feature_signature(self._make_candidate())
        assert sig["rsi_bucket"] == "elevated_70_80"
        assert sig["momentum_5d"] == "extended"
        assert sig["creator"] == "kpak82"

    def test_none_values_become_unknown(self):
        sig = reflect.make_feature_signature({
            "patterns":  {},
            "relative_volume": None,
            "change_5d_pct":   None,
            "supporting_creators": [],
        })
        assert sig["rsi_bucket"] == "unknown"
        assert sig["momentum_5d"] == "unknown"
        assert sig["creator"] == "unknown"


class TestSignaturesMatch:
    def _sig(self, rsi="elevated_70_80", mom="extended", creator="kpak82"):
        return {"rsi_bucket": rsi, "momentum_5d": mom, "creator": creator}

    def test_identical_match(self):
        assert reflect.signatures_match(self._sig(), self._sig())

    def test_different_rsi(self):
        assert not reflect.signatures_match(self._sig(), self._sig(rsi="neutral"))

    def test_different_momentum(self):
        assert not reflect.signatures_match(self._sig(), self._sig(mom="flat"))

    def test_different_creator(self):
        assert not reflect.signatures_match(self._sig(), self._sig(creator="puppy_trades"))


# ─── History record construction ────────────────────────────────────────────────

class TestBuildHistoryRecord:
    def _candidate(self):
        return {
            "_scan_date": "2026-06-06T20:05:00+00:00",
            "symbol": "PANW",
            "direction": "skip",
            "score": 0.55,
            "skip_reason": "RSI elevated",
            "would_have_direction": "put",
            "supporting_creators": ["kpak82"],
            "patterns": {"rsi": 72},
            "relative_volume": 3.1,
            "change_5d_pct": 7.5,
        }

    def test_basic_fields(self):
        r = reflect.build_history_record(
            self._candidate(),
            week_start=date(2026, 6, 2),
            outcome_pct=-8.5,
            miss_type="false_skip",
        )
        assert r["symbol"] == "PANW"
        assert r["week_start"] == "2026-06-02"
        assert r["miss_type"] == "false_skip"
        assert r["outcome_5d_pct"] == -8.5

    def test_outcome_correct_for_put_miss(self):
        r = reflect.build_history_record(
            self._candidate(),
            week_start=date(2026, 6, 2),
            outcome_pct=-8.5,   # dropped more than threshold → correct put
            miss_type="false_skip",
        )
        assert r["outcome_correct"] is True

    def test_outcome_incorrect_for_put_miss(self):
        r = reflect.build_history_record(
            self._candidate(),
            week_start=date(2026, 6, 2),
            outcome_pct=3.0,    # didn't drop — skip was wrong call but stock went up
            miss_type="false_skip",
        )
        assert r["outcome_correct"] is False

    def test_outcome_none_leaves_outcome_correct_none(self):
        r = reflect.build_history_record(
            self._candidate(),
            week_start=date(2026, 6, 2),
            outcome_pct=None,
            miss_type="false_skip",
        )
        assert r["outcome_correct"] is None

    def test_feature_signature_is_included(self):
        r = reflect.build_history_record(
            self._candidate(),
            week_start=date(2026, 6, 2),
            outcome_pct=-8.5,
            miss_type="false_skip",
        )
        assert "feature_signature" in r
        assert r["feature_signature"]["creator"] == "kpak82"


# ─── Pattern detection ──────────────────────────────────────────────────────────

class TestDetectPatterns:
    def test_no_patterns_below_threshold(self):
        records = [
            make_skip_record("AAPL", outcome=-8.5),
            make_skip_record("NVDA", outcome=-9.0),
        ]
        assert reflect.detect_patterns(records) == []

    def test_pattern_detected_at_threshold(self):
        # 3 same-signature false skips → 1 pattern
        records = [
            make_skip_record("AAPL", outcome=-8.5, scan_date="2026-05-02"),
            make_skip_record("NVDA", outcome=-9.0, scan_date="2026-05-09"),
            make_skip_record("TSLA", outcome=-7.5, scan_date="2026-05-16"),
        ]
        patterns = reflect.detect_patterns(records)
        assert len(patterns) == 1
        assert patterns[0]["occurrences"] == 3

    def test_dedup_same_symbol_within_14_days(self):
        # Same symbol 2x within 14 days → counts as one
        records = [
            make_skip_record("AAPL", outcome=-8.5, scan_date="2026-05-01"),
            make_skip_record("AAPL", outcome=-9.0, scan_date="2026-05-05"),  # dup
            make_skip_record("NVDA", outcome=-7.5, scan_date="2026-05-08"),
            make_skip_record("TSLA", outcome=-8.0, scan_date="2026-05-15"),
        ]
        patterns = reflect.detect_patterns(records)
        assert len(patterns) == 1
        assert patterns[0]["occurrences"] == 3  # AAPL, NVDA, TSLA (AAPL dup removed)

    def test_correct_skips_not_included(self):
        records = [
            make_skip_record("AAPL", miss_type="correct_skip", outcome_correct=False),
            make_skip_record("NVDA", miss_type="correct_skip", outcome_correct=False),
            make_skip_record("TSLA", miss_type="correct_skip", outcome_correct=False),
        ]
        assert reflect.detect_patterns(records) == []

    def test_rejected_amendments_excluded(self):
        records = [
            make_skip_record("AAPL", outcome=-8.5, scan_date="2026-05-01"),
            make_skip_record("NVDA", outcome=-9.0, scan_date="2026-05-08"),
            {**make_skip_record("TSLA", outcome=-7.5, scan_date="2026-05-15"),
             "amendment_rejected": True},
        ]
        assert reflect.detect_patterns(records) == []

    def test_avg_outcome_computed(self):
        records = [
            make_skip_record("AAPL", outcome=-8.0, scan_date="2026-05-02"),
            make_skip_record("NVDA", outcome=-10.0, scan_date="2026-05-09"),
            make_skip_record("TSLA", outcome=-9.0,  scan_date="2026-05-16"),
        ]
        patterns = reflect.detect_patterns(records)
        assert patterns[0]["avg_outcome"] == -9.0

    def test_different_creators_not_grouped(self):
        records = [
            make_skip_record("AAPL", creator="kpak82",       scan_date="2026-05-01"),
            make_skip_record("NVDA", creator="puppy_trades", scan_date="2026-05-08"),
            make_skip_record("TSLA", creator="kpak82",       scan_date="2026-05-15"),
        ]
        # kpak82 appears twice, puppy_trades once → no group reaches 3
        patterns = reflect.detect_patterns(records)
        assert len(patterns) == 0


# ─── Amendment generation ───────────────────────────────────────────────────────

class TestAmendmentGeneration:
    def _pattern(self):
        records = [
            make_skip_record("AAPL", outcome=-8.0, scan_date="2026-05-02"),
            make_skip_record("NVDA", outcome=-10.0, scan_date="2026-05-09"),
            make_skip_record("TSLA", outcome=-9.0,  scan_date="2026-05-16"),
        ]
        return {
            "signature": records[0]["feature_signature"],
            "occurrences": 3,
            "records": records,
            "avg_outcome": -9.0,
        }

    def test_contains_creator_handle(self):
        text = reflect.generate_amendment("kpak82", self._pattern())
        assert "@kpak82" in text

    def test_contains_apply_command(self):
        text = reflect.generate_amendment("kpak82", self._pattern())
        assert "reflect.py apply kpak82" in text

    def test_contains_reject_command(self):
        text = reflect.generate_amendment("kpak82", self._pattern())
        assert "reflect.py reject kpak82" in text

    def test_contains_evidence_section(self):
        text = reflect.generate_amendment("kpak82", self._pattern())
        assert "Evidence" in text

    def test_contains_avg_outcome(self):
        text = reflect.generate_amendment("kpak82", self._pattern())
        assert "-9.0%" in text


# ─── Framework version detection ────────────────────────────────────────────────

class TestFrameworkVersion:
    def test_detects_highest_version(self, tmp_trading):
        creator_dir = tmp_trading / "creators" / "kpak82"
        creator_dir.mkdir(parents=True, exist_ok=True)
        (creator_dir / "framework-v1.md").write_text("v1")
        (creator_dir / "framework-v2.md").write_text("v2")
        assert reflect._framework_version(creator_dir) == 2

    def test_returns_zero_when_no_frameworks(self, tmp_trading):
        creator_dir = tmp_trading / "creators" / "newcreator"
        creator_dir.mkdir(parents=True, exist_ok=True)
        assert reflect._framework_version(creator_dir) == 0


# ─── Amendment write / load ─────────────────────────────────────────────────────

class TestAmendmentFiles:
    def test_write_creates_file(self, tmp_trading):
        path = reflect.write_amendment("kpak82", "# Test amendment\n")
        assert path.exists()
        assert "kpak82" in str(path)

    def test_load_amendments_lists_pending(self, tmp_trading):
        reflect.write_amendment("kpak82", "# Test amendment\n")
        amendments = reflect.load_amendments()
        assert len(amendments) == 1
        assert amendments[0]["handle"] == "kpak82"

    def test_rejected_amendments_excluded(self, tmp_trading):
        path = reflect.write_amendment("kpak82", "# Test amendment\n")
        # Create rejection marker
        (path.parent / f"{path.stem}.rejected").write_text("{}")
        amendments = reflect.load_amendments()
        assert len(amendments) == 0

    def test_reject_creates_marker(self, tmp_trading):
        path = reflect.write_amendment("kpak82", "# Dummy\n  python reflect.py reject kpak82 --date 2026-06-07\n")
        # Patch HISTORY_PATH to empty
        (tmp_trading / "data" / "reflect_history.jsonl").write_text("")
        reflect.reject_amendment("kpak82", date.today().isoformat(), "too noisy")
        reject_marker = path.parent / f"amendment-{date.today().isoformat()}.rejected"
        assert reject_marker.exists()


# ─── Archive loading ────────────────────────────────────────────────────────────

class TestLoadWeekArchives:
    def test_loads_files_in_week(self, tmp_trading):
        archive_dir = tmp_trading / "data" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        week_start = date(2026, 6, 2)  # Monday
        # File on Wednesday of that week
        fname = archive_dir / "scored-20260604-1700.json"
        fname.write_text(json.dumps({
            "scan_timestamp": "2026-06-04T17:00:00Z",
            "all_scored": [{"symbol": "AAPL", "direction": "skip"}],
            "alerts": [],
        }))
        records = reflect.load_week_archives(week_start)
        assert len(records) == 1
        assert records[0]["symbol"] == "AAPL"

    def test_excludes_files_outside_week(self, tmp_trading):
        archive_dir = tmp_trading / "data" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        week_start = date(2026, 6, 2)
        # File from the prior week
        fname = archive_dir / "scored-20260527-1700.json"
        fname.write_text(json.dumps({
            "scan_timestamp": "2026-05-27T17:00:00Z",
            "all_scored": [{"symbol": "NVDA", "direction": "skip"}],
            "alerts": [],
        }))
        records = reflect.load_week_archives(week_start)
        assert len(records) == 0


# ─── Archive cleanup ────────────────────────────────────────────────────────────

class TestCleanupOldArchives:
    def test_deletes_old_files(self, tmp_trading):
        archive_dir = tmp_trading / "data" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        old_date = (date.today() - timedelta(days=20)).strftime("%Y%m%d")
        old_file = archive_dir / f"scored-{old_date}-1700.json"
        old_file.write_text("{}")
        deleted = reflect.cleanup_old_archives(keep_days=14)
        assert deleted == 1
        assert not old_file.exists()

    def test_keeps_recent_files(self, tmp_trading):
        archive_dir = tmp_trading / "data" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        new_date = date.today().strftime("%Y%m%d")
        new_file = archive_dir / f"scored-{new_date}-1700.json"
        new_file.write_text("{}")
        deleted = reflect.cleanup_old_archives(keep_days=14)
        assert deleted == 0
        assert new_file.exists()


# ─── Idempotency ────────────────────────────────────────────────────────────────

class TestProcessWeekIdempotency:
    def test_skips_already_processed_week(self, tmp_trading, monkeypatch):
        week_start = date(2026, 6, 2)
        existing = make_skip_record(week_start=week_start.isoformat())
        reflect.append_history([existing])

        result = reflect.process_week(week_start, force=False)
        assert result == []

    def test_force_reprocesses(self, tmp_trading, monkeypatch):
        week_start = date(2026, 6, 2)
        existing = make_skip_record(week_start=week_start.isoformat())
        reflect.append_history([existing])

        # No archive files → archive pass empty; no sent_history → sent pass empty
        result = reflect.process_week(week_start, force=True)
        assert result == []

    def test_sent_alerts_processed_independently(self, tmp_trading, monkeypatch):
        week_start = date(2026, 6, 9)
        reflect.append_history([make_skip_record(week_start=week_start.isoformat())])

        sent_path = tmp_trading / "data" / "sent_history.json"
        sent_path.write_text(json.dumps({
            "TNGX:call:ts": {
                "symbol": "TNGX", "direction": "call", "score": 0.8,
                "sent_at": "2026-06-09T13:41:42+00:00", "channels": ["email"],
            }
        }))

        monkeypatch.setattr(reflect, "evaluate_swing_alert", lambda alert, d, cfg=None, as_of=None: {
            "miss_type": "correct_take",
            "outcome_5d_pct": 25.0,
            "outcome_option_pnl_pct": 25.0,
            "outcome_underlying_pct": 5.0,
            "outcome_pending": True,
            "outcome_interim": True,
            "outcome_final": False,
            "outcome_as_of": "2026-06-19",
            "option_target_exit_date": "2026-06-23",
        })
        result = reflect.process_sent_alerts(week_start, force=False)
        assert len(result) == 1
        assert result[0]["source"] == "sent_alert"
        assert result[0]["miss_type"] == "correct_take"
        assert result[0]["outcome_option_pnl_pct"] == 25.0
        assert result[0]["outcome_interim"] is True

        # Interim scores refresh on re-run
        reflect.upsert_sent_alert_records(result)
        again = reflect.process_sent_alerts(week_start, force=False)
        assert len(again) == 1

        # Final scores are idempotent
        final = [{**result[0], "outcome_final": True, "outcome_interim": False}]
        reflect.upsert_sent_alert_records(final)
        assert reflect.process_sent_alerts(week_start, force=False) == []


# ─── Fetch outcome ──────────────────────────────────────────────────────────────

class TestFetchOutcome:
    def test_returns_none_on_error(self, monkeypatch):
        import yfinance as yf
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("network error")
        monkeypatch.setattr(yf, "Ticker", lambda s: mock_ticker)
        result = reflect.fetch_outcome("FAKE", date.today())
        assert result is None

    def test_returns_none_for_empty_history(self, monkeypatch):
        import pandas as pd
        import yfinance as yf
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        monkeypatch.setattr(yf, "Ticker", lambda s: mock_ticker)
        result = reflect.fetch_outcome("FAKE", date.today())
        assert result is None


# ─── Regression guards ──────────────────────────────────────────────────────────

class TestRegressionGuards:
    """Smoke tests that ensure modified files still import cleanly."""

    def test_utils_imports_ok(self):
        from utils import monday_of_week, load_budget, rsi_bucket, momentum_bucket, rel_vol_bucket
        assert callable(monday_of_week)

    def test_reflect_imports_ok(self):
        from reflect import (
            load_history, append_history, detect_patterns,
            generate_amendment, apply_amendment, reject_amendment,
            process_week, cleanup_old_archives,
        )
        assert callable(load_history)

    def test_orchestrate_imports_ok(self):
        import orchestrate
        assert hasattr(orchestrate, "score_candidates")
        assert hasattr(orchestrate, "filter_alerts")
        assert hasattr(orchestrate, "load_budget")

    def test_journal_imports_ok(self):
        import journal
        assert hasattr(journal, "load_journal")
        assert hasattr(journal, "load_budget")

    def test_monday_of_week_consistent(self):
        sunday = date(2026, 6, 7)  # Sunday → Monday of that week is Jun 1
        assert utils.monday_of_week(sunday) == date(2026, 6, 1)

    def test_monday_of_week_on_monday(self):
        monday = date(2026, 6, 1)  # June 1 2026 is a Monday
        assert utils.monday_of_week(monday) == date(2026, 6, 1)
