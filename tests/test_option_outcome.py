"""Tests for option_outcome.py — swing + intraday option P&L scoring."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

TRADING_DIR = Path.home() / "trading"
if str(TRADING_DIR) not in sys.path:
    sys.path.insert(0, str(TRADING_DIR))

import option_outcome as oo


def _alert_with_contract(entry_mid: float = 2.0, strike: float = 100.0) -> dict:
    return {
        "symbol": "SPY",
        "direction": "call",
        "scan_timestamp": "2026-06-17T14:00:00-04:00",
        "underlying_price": 100.0,
        "recommended_contract": {
            "tiers": {
                "atm": {
                    "strike": strike,
                    "expiration": "2026-06-17",
                    "mid_price": entry_mid,
                    "label": "$100C 6/17",
                }
            }
        },
    }


class TestPickContractTier:
    def test_prefers_atm(self):
        alert = _alert_with_contract()
        tier = oo.pick_contract_tier(alert)
        assert tier is not None
        assert tier["strike"] == 100.0


class TestEvaluateIntraday:
    def test_win_on_exit_alert_pnl(self, monkeypatch):
        entry = _alert_with_contract(entry_mid=2.0)
        exit_alert = {
            "alert_action": "exit",
            "scan_timestamp": "2026-06-17T15:00:00-04:00",
            "exit_option_mid": 3.0,
            "underlying_price": 101.0,
        }
        scored = oo.evaluate_intraday_alert(
            entry, {"intraday_0dte": {}}, exit_alert=exit_alert,
        )
        assert scored["miss_type"] == "correct_take"
        assert scored["outcome_option_pnl_pct"] == 50.0
        assert scored["outcome_exit_alert"] is True
        assert scored["outcome_no_exit"] is False

    def test_loss_on_exit_alert_pnl(self, monkeypatch):
        entry = _alert_with_contract(entry_mid=2.0)
        exit_alert = {
            "scan_timestamp": "2026-06-17T15:00:00-04:00",
            "exit_option_mid": 1.0,
            "underlying_price": 99.0,
        }
        scored = oo.evaluate_intraday_alert(
            entry, {"intraday_0dte": {}}, exit_alert=exit_alert,
        )
        assert scored["miss_type"] == "false_take"
        assert scored["outcome_option_pnl_pct"] == -50.0

    def test_no_exit_is_full_loss(self):
        scored = oo.evaluate_intraday_alert(
            _alert_with_contract(), {"intraday_0dte": {"no_exit_loss_pct": -100}},
            exit_alert=None,
        )
        assert scored["miss_type"] == "false_take"
        assert scored["outcome_option_pnl_pct"] == -100.0
        assert scored["outcome_no_exit"] is True
        assert scored["outcome_exit_alert"] is False

    def test_win_on_positive_pnl_legacy_mock(self, monkeypatch):
        monkeypatch.setattr(
            oo, "fetch_intraday_exit_outcome",
            lambda e, x: {"outcome_option_pnl_pct": 55.0, "outcome_underlying_pct": 0.3,
                          "entry_mid": 2.0, "exit_mid": 3.1, "exit_date": "2026-06-17",
                          "outcome_exit_alert": True},
        )
        scored = oo.evaluate_intraday_alert(
            _alert_with_contract(), {"intraday_0dte": {}}, exit_alert={"x": 1},
        )
        assert scored["miss_type"] == "correct_take"
        assert scored["outcome_option_pnl_pct"] == 55.0


class TestEvaluateSwing:
    def test_interim_score_before_full_hold(self, monkeypatch):
        monkeypatch.setattr(
            oo, "fetch_swing_option_outcome",
            lambda a, d, hold_days=5, as_of=None: {
                "outcome_option_pnl_pct": 11.0,
                "outcome_underlying_pct": 0.6,
                "entry_mid": 28.4,
                "exit_mid": 31.5,
                "exit_date": "2026-06-19",
                "outcome_interim": True,
                "target_exit_date": "2026-06-23",
                "outcome_as_of": "2026-06-19",
                "contract_label": "$300C",
            },
        )
        scored = oo.evaluate_swing_alert(
            _alert_with_contract(), date(2026, 6, 15), {"swing_reflect": {}},
        )
        assert scored["miss_type"] == "correct_take"
        assert scored["outcome_option_pnl_pct"] == 11.0
        assert scored["outcome_interim"] is True
        assert scored["outcome_final"] is False

    def test_win_on_final_option_pnl(self, monkeypatch):
        monkeypatch.setattr(
            oo, "fetch_swing_option_outcome",
            lambda a, d, hold_days=5, as_of=None: {
                "outcome_option_pnl_pct": 40.0,
                "outcome_underlying_pct": 4.0,
                "entry_mid": 2.0,
                "exit_mid": 2.8,
                "exit_date": "2026-06-24",
                "outcome_interim": False,
                "target_exit_date": "2026-06-24",
                "outcome_as_of": "2026-06-24",
                "contract_label": "$100C",
            },
        )
        scored = oo.evaluate_swing_alert(
            _alert_with_contract(), date(2026, 6, 17), {"swing_reflect": {}},
        )
        assert scored["miss_type"] == "correct_take"
        assert scored["outcome_option_pnl_pct"] == 40.0
        assert scored["outcome_final"] is True


class TestFindArchiveOnDate:
    def test_finds_alert_from_archive(self, tmp_path, monkeypatch):
        import json
        import utils

        archive = tmp_path / "data" / "archive"
        archive.mkdir(parents=True)
        monkeypatch.setattr(utils, "ARCHIVE_DIR", archive)
        monkeypatch.setattr(utils, "DATA_DIR", tmp_path / "data")

        archive.joinpath("scored-20260617-1200.json").write_text(json.dumps({
            "scan_timestamp": "2026-06-17T12:00:00Z",
            "alerts": [
                {"symbol": "STX", "direction": "call", "score": 0.72,
                 "recommended_contract": {"tiers": {"atm": {"mid_price": 3.0}}}},
            ],
            "all_scored": [],
        }))

        found = utils.find_archive_alert_on_date("STX", "call", date(2026, 6, 17))
        assert found is not None
        assert found["symbol"] == "STX"
        assert found["score"] == 0.72
