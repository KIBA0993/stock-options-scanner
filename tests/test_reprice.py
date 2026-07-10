"""Tests for reprice.py — backfilling option contracts for empty-tier alerts.

The live yfinance/contract fetch (orchestrate.pick_option_contract) is
monkeypatched so the suite runs offline and deterministically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TRADING_DIR = Path.home() / "trading"
if str(TRADING_DIR) not in sys.path:
    sys.path.insert(0, str(TRADING_DIR))

import reprice as rp


# ─── has_priced_tier ────────────────────────────────────────────────────────────
class TestHasPricedTier:
    def test_true_when_atm_has_mid(self):
        a = {"recommended_contract": {"tiers": {"atm": {"mid_price": 2.5}}}}
        assert rp.has_priced_tier(a) is True

    def test_false_when_tiers_empty(self):
        assert rp.has_priced_tier({"recommended_contract": {"tiers": {}}}) is False

    def test_false_when_mid_zero(self):
        a = {"recommended_contract": {"tiers": {"atm": {"mid_price": 0}}}}
        assert rp.has_priced_tier(a) is False

    def test_false_when_no_contract(self):
        assert rp.has_priced_tier({}) is False

    def test_true_via_fallback_tier(self):
        a = {"recommended_contract": {"tiers": {
            "atm": {"mid_price": 0}, "affordable": {"mid_price": 1.1}}}}
        assert rp.has_priced_tier(a) is True


# ─── fixtures / env ─────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    arch = data / "archive"
    arch.mkdir(parents=True)
    monkeypatch.setattr(rp, "DATA_DIR", data)
    monkeypatch.setattr(rp, "ARCHIVE_DIR", arch)
    monkeypatch.setattr(rp, "ALERTS_PATH", data / "alerts.json")
    monkeypatch.setattr(rp, "ALL_DATA_PATH", data / "all_data.json")
    monkeypatch.setattr(rp, "CONFIG_PATH", data / "config.json")
    return tmp_path, data, arch


SCAN_TS = "2026-07-07T13:40:34+00:00"


def _priced_contract(strike=100):
    return {"tiers": {"atm": {"strike": strike, "expiration": "2026-07-31",
                              "mid_price": 2.0, "label": f"${strike}C"}},
            "notes": "ok"}


def _empty_contract():
    return {"tiers": {"atm": None, "slight_otm": None, "affordable": None},
            "notes": "No liquid options found."}


def _write_env(data, arch, alerts, all_data=None):
    (data / "alerts.json").write_text(json.dumps({
        "scan_timestamp": SCAN_TS, "alert_count": len(alerts), "alerts": alerts}))
    (data / "config.json").write_text(json.dumps({"budget": {"total_usd": 500}}))
    if all_data is not None:
        (data / "all_data.json").write_text(json.dumps(all_data))
    # matching archive
    (arch / "scored-20260707-0943.json").write_text(json.dumps({
        "scan_timestamp": SCAN_TS, "all_scored": [],
        "alerts": [dict(a) for a in alerts]}))


def _alert(symbol="NVDA", direction="call", contract=None):
    return {"symbol": symbol, "direction": direction, "score": 0.8,
            "suggested_dte": "7-14 days",
            "recommended_contract": contract if contract is not None else _empty_contract()}


# ─── reprice() behaviour ────────────────────────────────────────────────────────
class TestReprice:
    def test_fills_empty_alert_and_archive(self, tmp_env, monkeypatch):
        _, data, arch = tmp_env
        _write_env(data, arch, [_alert("NVDA", "call")],
                   all_data={"tickers": [{"symbol": "NVDA", "price": 100,
                                          "options_chain": {}}]})
        monkeypatch.setattr(rp, "pick_option_contract", lambda **k: _priced_contract())

        res = rp.reprice(dry_run=False)
        assert res == {"candidates": 1, "filled": 1, "archive_updated": 1, "dry_run": False}

        # alerts.json now carries the contract
        alerts = json.loads((data / "alerts.json").read_text())
        assert rp.has_priced_tier(alerts["alerts"][0])
        assert "repriced_at" in alerts
        # archive too
        archp = json.loads((arch / "scored-20260707-0943.json").read_text())
        assert rp.has_priced_tier(archp["alerts"][0])

    def test_skips_alert_that_already_has_contract(self, tmp_env, monkeypatch):
        _, data, arch = tmp_env
        _write_env(data, arch, [_alert("NVDA", "call", contract=_priced_contract())],
                   all_data={"tickers": []})
        called = {"n": 0}
        def spy(**k): called["n"] += 1; return _priced_contract()
        monkeypatch.setattr(rp, "pick_option_contract", spy)

        res = rp.reprice(dry_run=False)
        assert res["candidates"] == 0
        assert called["n"] == 0  # never tried to reprice an already-priced alert

    def test_no_live_contract_leaves_alert_empty(self, tmp_env, monkeypatch):
        _, data, arch = tmp_env
        _write_env(data, arch, [_alert("NVDA", "call")],
                   all_data={"tickers": [{"symbol": "NVDA", "price": 100, "options_chain": {}}]})
        # market closed → still empty tiers returned
        monkeypatch.setattr(rp, "pick_option_contract", lambda **k: _empty_contract())

        res = rp.reprice(dry_run=False)
        assert res["filled"] == 0
        assert res["archive_updated"] == 0
        alerts = json.loads((data / "alerts.json").read_text())
        assert not rp.has_priced_tier(alerts["alerts"][0])

    def test_dry_run_writes_nothing(self, tmp_env, monkeypatch):
        _, data, arch = tmp_env
        _write_env(data, arch, [_alert("NVDA", "call")],
                   all_data={"tickers": [{"symbol": "NVDA", "price": 100, "options_chain": {}}]})
        monkeypatch.setattr(rp, "pick_option_contract", lambda **k: _priced_contract())

        before_alerts = (data / "alerts.json").read_text()
        before_arch = (arch / "scored-20260707-0943.json").read_text()
        res = rp.reprice(dry_run=True)

        # dry-run reports what it WOULD do (consistent with `filled`) but writes nothing
        assert res["filled"] == 1
        assert res["archive_updated"] == 1
        assert res["dry_run"] is True
        assert (data / "alerts.json").read_text() == before_alerts
        assert (arch / "scored-20260707-0943.json").read_text() == before_arch

    def test_is_idempotent(self, tmp_env, monkeypatch):
        _, data, arch = tmp_env
        _write_env(data, arch, [_alert("NVDA", "call")],
                   all_data={"tickers": [{"symbol": "NVDA", "price": 100, "options_chain": {}}]})
        monkeypatch.setattr(rp, "pick_option_contract", lambda **k: _priced_contract())
        rp.reprice(dry_run=False)
        second = rp.reprice(dry_run=False)   # nothing left to do
        assert second["candidates"] == 0
        assert second["filled"] == 0

    def test_only_updates_matching_scan_timestamp_archive(self, tmp_env, monkeypatch):
        _, data, arch = tmp_env
        _write_env(data, arch, [_alert("NVDA", "call")],
                   all_data={"tickers": [{"symbol": "NVDA", "price": 100, "options_chain": {}}]})
        # an unrelated archive from a different scan must stay untouched
        other = {"scan_timestamp": "2026-07-06T18:30:00+00:00", "all_scored": [],
                 "alerts": [_alert("NVDA", "call")]}
        (arch / "scored-20260706-1433.json").write_text(json.dumps(other))
        monkeypatch.setattr(rp, "pick_option_contract", lambda **k: _priced_contract())

        rp.reprice(dry_run=False)
        untouched = json.loads((arch / "scored-20260706-1433.json").read_text())
        assert not rp.has_priced_tier(untouched["alerts"][0])

    def test_handles_missing_alerts_file(self, tmp_env):
        _, data, arch = tmp_env
        res = rp.reprice(dry_run=False)
        assert res == {"candidates": 0, "filled": 0, "archive_updated": 0}
