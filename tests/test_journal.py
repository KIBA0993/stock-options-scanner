"""
tests/test_journal.py — Unit tests for journal.py (Week 5)

Run: cd ~/trading && python -m pytest tests/test_journal.py -v
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from journal import (
    _monday_of_week,
    _new_trade_id,
    append_entry,
    calc_r_multiple,
    classify_outcome,
    find_alert,
    load_budget,
    load_journal,
    update_entry,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_journal(tmp_path: Path):
    """Return a tmp JOURNAL_PATH and patch the module constant."""
    path = tmp_path / "trade_journal.jsonl"
    with patch("journal.JOURNAL_PATH", path):
        yield path


@pytest.fixture
def tmp_budget(tmp_path: Path):
    """Return a tmp BUDGET_PATH and patch the module constant."""
    path = tmp_path / "budget.json"
    with patch("journal.BUDGET_PATH", path):
        yield path


@pytest.fixture
def sample_entry() -> dict:
    return {
        "id":            "2026-06-08-NVDA-call-001",
        "symbol":        "NVDA",
        "direction":     "call",
        "alert_score":   0.85,
        "alert_date":    "2026-06-08",
        "entry_date":    "2026-06-08",
        "entry_price":   2.35,
        "stop_price":    1.18,
        "target_price":  4.70,
        "strike":        900.0,
        "expiration":    "2026-06-20",
        "exit_date":     None,
        "exit_price":    None,
        "outcome":       None,
        "pnl_r":         None,
        "creator_match": ["kpak82"],
        "scoring_method": "heuristic",
        "notes":         "",
        "logged_at":     "2026-06-08T12:00:00+00:00",
    }


@pytest.fixture
def closed_entry(sample_entry) -> dict:
    e = dict(sample_entry)
    e.update({"exit_date": "2026-06-15", "exit_price": 4.10,
               "pnl_r": 1.48, "outcome": "partial_win"})
    return e


# ─── _monday_of_week ──────────────────────────────────────────────────────────
class TestMondayOfWeek:
    def test_monday_returns_itself(self) -> None:
        monday = date(2026, 6, 8)  # a Monday
        assert _monday_of_week(monday) == monday

    def test_friday_returns_monday(self) -> None:
        friday = date(2026, 6, 12)
        assert _monday_of_week(friday) == date(2026, 6, 8)

    def test_sunday_returns_monday(self) -> None:
        sunday = date(2026, 6, 14)
        assert _monday_of_week(sunday) == date(2026, 6, 8)


# ─── calc_r_multiple ──────────────────────────────────────────────────────────
class TestCalcRMultiple:
    def test_call_winner_returns_positive_r(self) -> None:
        r = calc_r_multiple(entry_price=2.00, exit_price=4.00, stop_price=1.00, direction="call")
        assert r == pytest.approx(2.0)

    def test_call_loser_returns_negative_r(self) -> None:
        r = calc_r_multiple(entry_price=2.00, exit_price=1.00, stop_price=1.00, direction="call")
        assert r == pytest.approx(-1.0)

    def test_put_winner_returns_positive_r(self) -> None:
        # For puts: R = (entry - exit) / risk
        r = calc_r_multiple(entry_price=3.00, exit_price=1.00, stop_price=1.50, direction="put")
        assert r == pytest.approx(2.0 / 1.5, rel=1e-4)

    def test_put_loser_returns_negative_r(self) -> None:
        r = calc_r_multiple(entry_price=2.00, exit_price=3.00, stop_price=1.00, direction="put")
        assert r == pytest.approx(-1.0)

    def test_no_stop_defaults_to_50pct_of_entry(self) -> None:
        r = calc_r_multiple(entry_price=2.00, exit_price=4.00, stop_price=None, direction="call")
        # risk = 2.00 * 0.5 = 1.00; gain = 2.00; R = 2.00 / 1.00 = 2.0
        assert r == pytest.approx(2.0)

    def test_zero_or_negative_risk_returns_zero(self) -> None:
        # stop >= entry → risk <= 0
        r = calc_r_multiple(entry_price=2.00, exit_price=3.00, stop_price=2.50, direction="call")
        assert r == 0.0

    def test_breakeven_returns_zero(self) -> None:
        r = calc_r_multiple(entry_price=2.00, exit_price=2.00, stop_price=1.00, direction="call")
        assert r == pytest.approx(0.0)


# ─── classify_outcome ─────────────────────────────────────────────────────────
class TestClassifyOutcome:
    def test_high_r_is_target_hit(self) -> None:
        assert classify_outcome(2.0) == "target_hit"
        assert classify_outcome(1.5) == "target_hit"

    def test_small_positive_is_partial_win(self) -> None:
        assert classify_outcome(0.5) == "partial_win"
        assert classify_outcome(0.0) == "partial_win"

    def test_moderate_loss_is_stop_hit(self) -> None:
        assert classify_outcome(-0.5) == "stop_hit"
        assert classify_outcome(-1.0) == "stop_hit"

    def test_large_loss_is_full_loss(self) -> None:
        assert classify_outcome(-1.5) == "full_loss"
        assert classify_outcome(-2.0) == "full_loss"


# ─── load_journal / append_entry ──────────────────────────────────────────────
class TestJournalIO:
    def test_empty_journal_returns_empty_list(self, tmp_journal) -> None:
        with patch("journal.JOURNAL_PATH", tmp_journal):
            result = load_journal()
        assert result == []

    def test_missing_journal_returns_empty_list(self, tmp_path) -> None:
        missing = tmp_path / "nonexistent.jsonl"
        with patch("journal.JOURNAL_PATH", missing):
            result = load_journal()
        assert result == []

    def test_append_creates_file(self, tmp_journal, sample_entry) -> None:
        with patch("journal.JOURNAL_PATH", tmp_journal):
            append_entry(sample_entry)
        assert tmp_journal.exists()

    def test_append_then_load_roundtrips(self, tmp_journal, sample_entry) -> None:
        with patch("journal.JOURNAL_PATH", tmp_journal):
            append_entry(sample_entry)
            entries = load_journal()
        assert len(entries) == 1
        assert entries[0]["symbol"] == "NVDA"

    def test_multiple_entries_appended(self, tmp_journal, sample_entry) -> None:
        e2 = dict(sample_entry)
        e2["id"] = "2026-06-08-SPY-put-001"
        e2["symbol"] = "SPY"
        with patch("journal.JOURNAL_PATH", tmp_journal):
            append_entry(sample_entry)
            append_entry(e2)
            entries = load_journal()
        assert len(entries) == 2

    def test_malformed_line_is_skipped(self, tmp_journal) -> None:
        tmp_journal.write_text('{"id":"valid"}\nBAD_JSON_LINE\n{"id":"valid2"}\n')
        with patch("journal.JOURNAL_PATH", tmp_journal):
            entries = load_journal()
        assert len(entries) == 2


# ─── update_entry ─────────────────────────────────────────────────────────────
class TestUpdateEntry:
    def test_updates_matching_entry(self, tmp_journal, sample_entry) -> None:
        with patch("journal.JOURNAL_PATH", tmp_journal):
            append_entry(sample_entry)
            result = update_entry(sample_entry["id"], {"exit_price": 4.10})
            entries = load_journal()
        assert result is True
        assert entries[0]["exit_price"] == 4.10

    def test_returns_false_when_not_found(self, tmp_journal, sample_entry) -> None:
        with patch("journal.JOURNAL_PATH", tmp_journal):
            append_entry(sample_entry)
            result = update_entry("nonexistent-id", {"exit_price": 4.10})
        assert result is False

    def test_other_entries_unchanged(self, tmp_journal, sample_entry) -> None:
        e2 = dict(sample_entry)
        e2["id"] = "other-id"
        e2["entry_price"] = 5.00
        with patch("journal.JOURNAL_PATH", tmp_journal):
            append_entry(sample_entry)
            append_entry(e2)
            update_entry(sample_entry["id"], {"exit_price": 4.10})
            entries = load_journal()
        other = next(e for e in entries if e["id"] == "other-id")
        assert other["entry_price"] == 5.00


# ─── load_budget ─────────────────────────────────────────────────────────────
class TestLoadBudget:
    def test_creates_fresh_budget_when_missing(self, tmp_budget) -> None:
        with patch("journal.BUDGET_PATH", tmp_budget), patch("utils.BUDGET_PATH", tmp_budget):
            budget = load_budget()
        assert "week_start" in budget
        assert budget["surfaced_this_week"] == 0

    def test_loads_existing_budget(self, tmp_budget) -> None:
        data = {"week_start": date.today().isoformat(), "surfaced_this_week": 3}
        tmp_budget.write_text(json.dumps(data))
        with patch("journal.BUDGET_PATH", tmp_budget), patch("utils.BUDGET_PATH", tmp_budget):
            budget = load_budget()
        assert budget["surfaced_this_week"] == 3

    def test_auto_resets_expired_budget(self, tmp_budget) -> None:
        old_monday = (_monday_of_week(date.today()) - timedelta(weeks=1)).isoformat()
        data = {"week_start": old_monday, "surfaced_this_week": 7}
        tmp_budget.write_text(json.dumps(data))
        with patch("journal.BUDGET_PATH", tmp_budget), patch("utils.BUDGET_PATH", tmp_budget):
            budget = load_budget()
        assert budget["surfaced_this_week"] == 0

    def test_does_not_reset_current_week(self, tmp_budget) -> None:
        this_monday = _monday_of_week(date.today()).isoformat()
        data = {"week_start": this_monday, "surfaced_this_week": 5}
        tmp_budget.write_text(json.dumps(data))
        with patch("journal.BUDGET_PATH", tmp_budget), patch("utils.BUDGET_PATH", tmp_budget):
            budget = load_budget()
        assert budget["surfaced_this_week"] == 5


# ─── _new_trade_id ────────────────────────────────────────────────────────────
class TestNewTradeId:
    def test_id_contains_symbol_and_direction(self, tmp_journal) -> None:
        with patch("journal.JOURNAL_PATH", tmp_journal):
            trade_id = _new_trade_id("NVDA", "call")
        assert "NVDA" in trade_id
        assert "call" in trade_id

    def test_id_contains_today(self, tmp_journal) -> None:
        with patch("journal.JOURNAL_PATH", tmp_journal):
            trade_id = _new_trade_id("NVDA", "call")
        assert date.today().isoformat() in trade_id

    def test_sequence_increments_for_same_symbol_today(self, tmp_journal) -> None:
        today = date.today().isoformat()
        existing = {
            "id":          f"{today}-NVDA-call-001",
            "symbol":      "NVDA",
            "direction":   "call",
            "entry_price": 2.35,
            "logged_at":   f"{today}T12:00:00+00:00",
        }
        with patch("journal.JOURNAL_PATH", tmp_journal):
            append_entry(existing)
            trade_id = _new_trade_id("NVDA", "call")
        assert trade_id.endswith("002")


# ─── find_alert ───────────────────────────────────────────────────────────────
class TestFindAlert:
    def test_finds_matching_alert(self, tmp_path) -> None:
        alerts_data = {
            "alerts": [
                {"symbol": "BBCP", "direction": "call", "score": 0.80},
                {"symbol": "NVDA", "direction": "put",  "score": 0.75},
            ]
        }
        alerts_path = tmp_path / "alerts.json"
        alerts_path.write_text(json.dumps(alerts_data))
        with patch("journal.ALERTS_PATH", alerts_path):
            result = find_alert("NVDA")
        assert result is not None
        assert result["direction"] == "put"

    def test_returns_none_when_not_found(self, tmp_path) -> None:
        alerts_data = {"alerts": [{"symbol": "AAPL", "direction": "call", "score": 0.8}]}
        alerts_path = tmp_path / "alerts.json"
        alerts_path.write_text(json.dumps(alerts_data))
        with patch("journal.ALERTS_PATH", alerts_path):
            result = find_alert("MISSING")
        assert result is None

    def test_case_insensitive_match(self, tmp_path) -> None:
        alerts_data = {"alerts": [{"symbol": "BBCP", "direction": "call", "score": 0.8}]}
        alerts_path = tmp_path / "alerts.json"
        alerts_path.write_text(json.dumps(alerts_data))
        with patch("journal.ALERTS_PATH", alerts_path):
            result = find_alert("bbcp")
        assert result is not None
