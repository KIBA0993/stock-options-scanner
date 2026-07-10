"""Tests for validate.py — forward performance validation harness.

Pure-logic + IO tests. Network calls (yfinance) are monkeypatched so the suite
runs offline and deterministically.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

TRADING_DIR = Path.home() / "trading"
if str(TRADING_DIR) not in sys.path:
    sys.path.insert(0, str(TRADING_DIR))

import validate as v


# ─── Pure helpers ───────────────────────────────────────────────────────────────
class TestPredictionId:
    def test_normalises_case(self):
        assert v.prediction_id("nvda", "CALL", "2026-07-01") == "NVDA:call:2026-07-01"

    def test_distinguishes_direction(self):
        a = v.prediction_id("SPY", "call", "2026-07-01")
        b = v.prediction_id("SPY", "put", "2026-07-01")
        assert a != b


class TestEntryContract:
    def test_prefers_atm_with_price(self):
        alert = {"recommended_contract": {"tiers": {
            "atm": {"strike": 100, "expiration": "2026-07-10", "mid_price": 2.0},
            "slight_otm": {"strike": 105, "expiration": "2026-07-10", "mid_price": 1.0},
        }}}
        assert v._entry_contract(alert)["strike"] == 100

    def test_falls_back_when_atm_unpriced(self):
        alert = {"recommended_contract": {"tiers": {
            "atm": {"strike": 100, "mid_price": 0},
            "slight_otm": {"strike": 105, "mid_price": 1.0},
        }}}
        assert v._entry_contract(alert)["strike"] == 105

    def test_none_when_no_priced_tier(self):
        alert = {"recommended_contract": {"tiers": {"atm": {"mid_price": 0}}}}
        assert v._entry_contract(alert) is None

    def test_none_when_no_contract(self):
        assert v._entry_contract({}) is None


class TestDirectionalCorrect:
    def test_call_up_is_correct(self):
        assert v._directional_correct("call", 1.5) is True

    def test_call_down_is_wrong(self):
        assert v._directional_correct("call", -1.5) is False

    def test_put_down_is_correct(self):
        assert v._directional_correct("put", -2.0) is True

    def test_put_up_is_wrong(self):
        assert v._directional_correct("put", 2.0) is False

    def test_none_move_is_none(self):
        assert v._directional_correct("call", None) is None


class TestScoreBucket:
    @pytest.mark.parametrize("score,bucket", [
        (0.95, "0.80+"), (0.80, "0.80+"), (0.75, "0.70-0.79"),
        (0.60, "0.60-0.69"), (0.4, "<0.60"), (None, "n/a"),
    ])
    def test_buckets(self, score, bucket):
        assert v._score_bucket(score) == bucket


class TestAgg:
    def test_directional_and_option_stats(self):
        rows = [
            {"direction": "call", "underlying_move_pct": 2.0,
             "directional_correct": True, "option_pnl_pct": 50.0},
            {"direction": "call", "underlying_move_pct": -1.0,
             "directional_correct": False, "option_pnl_pct": -100.0},
            {"direction": "put", "underlying_move_pct": -3.0,
             "directional_correct": True, "option_pnl_pct": None},
        ]
        a = v._agg(rows)
        assert a["n"] == 3
        assert a["n_directional"] == 3
        assert a["dir_hit_rate"] == pytest.approx(66.7, abs=0.1)
        # captured move: call +2, call -1, put -(-3)=+3 → mean = 4/3
        assert a["avg_captured_move"] == pytest.approx(1.33, abs=0.02)
        assert a["n_option"] == 2
        assert a["option_win_rate"] == 50.0
        assert a["avg_option_pnl"] == pytest.approx(-25.0)

    def test_empty(self):
        a = v._agg([])
        assert a["n"] == 0
        assert a["dir_hit_rate"] is None


# ─── Ledger + snapshot IO (redirected to tmp paths) ─────────────────────────────
@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    valid = tmp_path / "validation"
    arch = tmp_path / "archive"
    valid.mkdir(); arch.mkdir()
    monkeypatch.setattr(v, "VALID_DIR", valid)
    monkeypatch.setattr(v, "LEDGER_PATH", valid / "prediction_ledger.jsonl")
    monkeypatch.setattr(v, "OUTCOMES_PATH", valid / "prediction_outcomes.json")
    monkeypatch.setattr(v, "ARCHIVE_DIR", arch)
    monkeypatch.setattr(v, "ALERTS_PATH", tmp_path / "alerts.json")
    monkeypatch.setattr(v, "CONFIG_PATH", tmp_path / "config.json")
    return tmp_path, arch, valid


def _write_archive(arch: Path, name: str, scan_ts: str, alerts: list[dict]):
    (arch / name).write_text(json.dumps({
        "scan_timestamp": scan_ts, "all_scored": [], "alerts": alerts,
    }))


def _alert(symbol="NVDA", direction="call", score=0.8, mid=2.0):
    return {
        "symbol": symbol, "direction": direction, "score": score,
        "scoring_method": "llm", "supporting_creators": ["@kpak82"],
        "recommended_contract": {"tiers": {"atm": {
            "strike": 100, "expiration": "2026-07-31", "mid_price": mid}}},
    }


class TestSnapshot:
    def test_ingests_and_dedupes(self, tmp_env):
        _, arch, valid = tmp_env
        _write_archive(arch, "scored-20260701-0943.json", "2026-07-01T13:43:00+00:00",
                       [_alert("NVDA", "call"), _alert("SPY", "put")])
        # Second scan same day re-surfaces NVDA (same pid) + adds AMD
        _write_archive(arch, "scored-20260701-1203.json", "2026-07-01T16:03:00+00:00",
                       [_alert("NVDA", "call"), _alert("AMD", "call")])

        v.cmd_snapshot(_ns())
        led = v.load_ledger()
        assert set(led) == {
            "NVDA:call:2026-07-01", "SPY:put:2026-07-01", "AMD:call:2026-07-01"}

    def test_snapshot_is_idempotent(self, tmp_env):
        _, arch, valid = tmp_env
        _write_archive(arch, "scored-20260701-0943.json", "2026-07-01T13:43:00+00:00",
                       [_alert("NVDA", "call")])
        v.cmd_snapshot(_ns())
        v.cmd_snapshot(_ns())  # second run adds nothing
        assert len(v.load_ledger()) == 1
        # Ledger file has exactly one line (no duplicate append)
        lines = [l for l in v.LEDGER_PATH.read_text().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_skips_skip_direction(self, tmp_env):
        _, arch, valid = tmp_env
        _write_archive(arch, "scored-20260701-0943.json", "2026-07-01T13:43:00+00:00",
                       [{"symbol": "XYZ", "direction": "skip", "score": 0.1}])
        v.cmd_snapshot(_ns())
        assert len(v.load_ledger()) == 0

    def test_records_missing_contract_as_none(self, tmp_env):
        _, arch, valid = tmp_env
        a = _alert("NVDA", "call")
        a["recommended_contract"] = {"tiers": {}}  # unpriced (pre-market scan)
        _write_archive(arch, "scored-20260701-0943.json", "2026-07-01T13:43:00+00:00", [a])
        v.cmd_snapshot(_ns())
        snap = v.load_ledger()["NVDA:call:2026-07-01"]
        assert snap["entry_contract"] is None


# ─── mark (network monkeypatched) ───────────────────────────────────────────────
class TestMark:
    def test_marks_directional_outcome(self, tmp_env, monkeypatch):
        _, arch, valid = tmp_env
        # Entry far enough in the past that all horizons are final.
        _write_archive(arch, "scored-20260601-0943.json", "2026-06-01T13:43:00+00:00",
                       [_alert("NVDA", "call", mid=2.0)])
        v.cmd_snapshot(_ns())

        # Underlying: entry 100 → always 105 (call correct, +5%)
        monkeypatch.setattr(v, "underlying_close_on", lambda sym, d: 100.0 if d == date(2026, 6, 1) else 105.0)
        # Option P&L stub
        monkeypatch.setattr(v, "fetch_swing_option_outcome",
                            lambda *a, **k: {"outcome_option_pnl_pct": 40.0,
                                             "entry_mid": 2.0, "exit_mid": 2.8})
        v.cmd_mark(_ns(force=False))

        oc = v.load_outcomes()["NVDA:call:2026-06-01"]
        assert oc["all_final"] is True
        h1 = oc["horizons"]["h1"]
        assert h1["underlying_move_pct"] == 5.0
        assert h1["directional_correct"] is True
        assert h1["option_pnl_pct"] == 40.0

    def test_mark_skips_already_final(self, tmp_env, monkeypatch):
        _, arch, valid = tmp_env
        _write_archive(arch, "scored-20260601-0943.json", "2026-06-01T13:43:00+00:00",
                       [_alert("NVDA", "call")])
        v.cmd_snapshot(_ns())
        monkeypatch.setattr(v, "underlying_close_on", lambda sym, d: 100.0 if d == date(2026, 6, 1) else 105.0)
        monkeypatch.setattr(v, "fetch_swing_option_outcome", lambda *a, **k: None)
        v.cmd_mark(_ns(force=False))
        first = v.load_outcomes()["NVDA:call:2026-06-01"]["marked_at"]
        # Re-mark without force should not touch a fully-final prediction.
        v.cmd_mark(_ns(force=False))
        assert v.load_outcomes()["NVDA:call:2026-06-01"]["marked_at"] == first


# ─── scorecard ──────────────────────────────────────────────────────────────────
class TestScorecard:
    def test_build_scorecard_aggregates(self, tmp_env, monkeypatch):
        _, arch, valid = tmp_env
        _write_archive(arch, "scored-20260601-0943.json", "2026-06-01T13:43:00+00:00",
                       [_alert("NVDA", "call", score=0.85),
                        _alert("SPY", "put", score=0.72)])
        v.cmd_snapshot(_ns())
        monkeypatch.setattr(v, "underlying_close_on",
                            lambda sym, d: 100.0 if d == date(2026, 6, 1) else 108.0)
        monkeypatch.setattr(v, "fetch_swing_option_outcome", lambda *a, **k: None)
        v.cmd_mark(_ns(force=False))

        sc = v.build_scorecard(final_only=True, config={})
        h1 = sc["by_horizon"]["h1"]["overall"]
        # NVDA call +8% correct; SPY put +8% wrong → 50% dir hit
        assert h1["n"] == 2
        assert h1["dir_hit_rate"] == 50.0
        # score buckets present
        assert "0.80+" in sc["by_horizon"]["h1"]["by_score"]
        assert "0.70-0.79" in sc["by_horizon"]["h1"]["by_score"]


# ─── HTML scorecard ─────────────────────────────────────────────────────────────
class TestHtmlScorecard:
    def _sc(self):
        return {
            "generated_at": "2026-07-10T12:00:00+00:00",
            "horizons_days": [1, 3],
            "ledger_total": 5, "marked_total": 5,
            "by_horizon": {
                "h1": {
                    "overall": {"n": 5, "n_directional": 5, "dir_hit_rate": 60.0,
                                "avg_abs_move": 2.0, "avg_captured_move": 1.1,
                                "n_option": 3, "option_win_rate": 33.0, "avg_option_pnl": -5.0},
                    "by_method": {"llm": {"n": 5, "dir_hit_rate": 60.0, "avg_captured_move": 1.1,
                                          "n_option": 3, "option_win_rate": 33.0, "avg_option_pnl": -5.0}},
                    "by_score": {}, "by_direction": {}, "by_creator": {},
                },
                "h3": {"overall": {"n": 0, "n_directional": 0, "dir_hit_rate": None,
                                   "avg_captured_move": None, "n_option": 0,
                                   "option_win_rate": None, "avg_option_pnl": None},
                       "by_method": {}, "by_score": {}, "by_direction": {}, "by_creator": {}},
            },
        }

    def test_renders_key_content(self):
        html = v._scorecard_html(self._sc(), weeks=None, include_interim=False)
        assert "Performance scorecard" in html
        assert "H1" in html
        assert "60%" in html            # directional hit rate
        assert "+10 pts" in html         # edge vs coin-flip
        # empty h3 horizon should be skipped, not rendered as a section
        assert "H3" not in html

    def test_edge_color_thresholds(self):
        assert v._edge_color(70) == "#1f9d55"   # green
        assert v._edge_color(40) == "#d1495b"   # red
        assert v._edge_color(50) == "#bd7a10"   # amber (neutral band)
        assert v._edge_color(None) == "#8695a4"  # grey

    def test_cmd_report_html_writes_file(self, tmp_env, monkeypatch):
        _, arch, valid = tmp_env
        _write_archive(arch, "scored-20260601-0943.json", "2026-06-01T13:43:00+00:00",
                       [_alert("NVDA", "call")])
        v.cmd_snapshot(_ns())
        monkeypatch.setattr(v, "underlying_close_on",
                            lambda sym, d: 100.0 if d == date(2026, 6, 1) else 105.0)
        monkeypatch.setattr(v, "fetch_swing_option_outcome", lambda *a, **k: None)
        v.cmd_mark(_ns())
        out = valid / "scorecard.html"
        monkeypatch.setattr(v, "SCORECARD_PATH", out)
        v.cmd_report(_ns(html=True))
        assert out.exists()
        text = out.read_text()
        assert "<!doctype html>" in text
        assert "Performance scorecard" in text


# ─── helpers ────────────────────────────────────────────────────────────────────
def _ns(**kw):
    """Minimal argparse.Namespace stand-in for cmd_* functions."""
    import argparse
    defaults = dict(force=False, weeks=None, include_interim=False, json=False,
                    html=False, out=None)
    defaults.update(kw)
    return argparse.Namespace(**defaults)
