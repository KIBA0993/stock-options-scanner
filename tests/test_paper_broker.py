"""Tests for paper_broker.py — Alpaca PAPER execution scaffolding (P2-2).

Covers the offline/pure surface: OCC symbol construction, alert→order mapping,
the paper-only safety guard, enable-gating, dedup, and fill reconciliation.
No network and no alpaca-py import are exercised.
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

import paper_broker as pb


# ─── OCC symbol ─────────────────────────────────────────────────────────────────
class TestBuildOccSymbol:
    def test_call(self):
        assert pb.build_occ_symbol("SPY", "2026-07-31", "call", 750) == "SPY260731C00750000"

    def test_put(self):
        assert pb.build_occ_symbol("ASTH", "2026-07-17", "put", 50.0) == "ASTH260717P00050000"

    def test_fractional_strike(self):
        # 249.5 → 249500 → 00249500
        assert pb.build_occ_symbol("MRVL", "2026-07-17", "call", 249.5) == "MRVL260717C00249500"

    def test_high_strike(self):
        # MU at 975 → 00975000
        assert pb.build_occ_symbol("MU", "2026-07-17", "put", 975) == "MU260717P00975000"


# ─── alert_to_order ─────────────────────────────────────────────────────────────
def _alert(symbol="ASTH", direction="put", strike=50.0, mid=1.73, ask=1.95):
    return {
        "symbol": symbol, "direction": direction, "score": 0.8,
        "recommended_contract": {"tiers": {"atm": {
            "strike": strike, "expiration": "2026-07-17", "direction": direction,
            "bid": 1.5, "ask": ask, "mid_price": mid, "cost_per_contract": mid * 100}}},
    }


class TestAlertToOrder:
    def test_maps_priced_alert(self):
        o = pb.alert_to_order(_alert(), scan_date="2026-07-02", qty=1)
        assert o["occ_symbol"] == "ASTH260717P00050000"
        assert o["side"] == "buy"
        assert o["type"] == "limit"
        assert o["limit_price"] == 1.95        # buys at the ask (realistic fill)
        assert o["entry_mid"] == 1.73
        assert o["prediction_id"] == "ASTH:put:2026-07-02"

    def test_market_order_variant(self):
        o = pb.alert_to_order(_alert(), scan_date="2026-07-02", limit=False)
        assert o["type"] == "market"
        assert o["limit_price"] is None

    def test_none_when_no_priced_contract(self):
        a = _alert()
        a["recommended_contract"] = {"tiers": {}}
        assert pb.alert_to_order(a, scan_date="2026-07-02") is None

    def test_qty_passthrough(self):
        o = pb.alert_to_order(_alert(), scan_date="2026-07-02", qty=3)
        assert o["qty"] == 3


# ─── paper-only safety guard ────────────────────────────────────────────────────
class TestPaperGuard:
    def test_allows_paper_endpoint(self):
        pb._assert_paper({"paper": True, "base_url": "https://paper-api.alpaca.markets"})

    def test_rejects_live_endpoint(self):
        with pytest.raises(RuntimeError):
            pb._assert_paper({"paper": True, "base_url": "https://api.alpaca.markets"})

    def test_rejects_paper_false(self):
        with pytest.raises(RuntimeError):
            pb._assert_paper({"paper": False, "base_url": "https://paper-api.alpaca.markets"})


# ─── enable gating ──────────────────────────────────────────────────────────────
class TestEnableGating:
    def test_disabled_by_default(self):
        assert pb.is_enabled({"alpaca": {}}) is False

    def test_enabled_flag(self):
        assert pb.is_enabled({"alpaca": {"enabled": True}}) is True

    def test_submit_noops_when_disabled(self, monkeypatch):
        # Even if a client were reachable, disabled config must not submit.
        monkeypatch.setattr(pb, "_client", lambda cfg: pytest.fail("must not build client"))
        r = pb.submit_alerts(config={"alpaca": {"enabled": False}})
        assert r == {"enabled": False, "submitted": 0, "skipped": 0}


# ─── submit (dry-run, no SDK) ───────────────────────────────────────────────────
class TestSubmitDryRun:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        data = tmp_path / "data"; paper = data / "paper"; paper.mkdir(parents=True)
        monkeypatch.setattr(pb, "DATA_DIR", data)
        monkeypatch.setattr(pb, "PAPER_DIR", paper)
        monkeypatch.setattr(pb, "ORDERS_PATH", paper / "paper_orders.jsonl")
        monkeypatch.setattr(pb, "FILLS_PATH", paper / "paper_fills.json")
        monkeypatch.setattr(pb, "ALERTS_PATH", data / "alerts.json")
        (data / "alerts.json").write_text(json.dumps({
            "scan_timestamp": "2026-07-02T13:43:00+00:00",
            "alerts": [_alert("ASTH", "put"), _alert("NVDA", "call", strike=200, mid=0.92, ask=0.93)]}))
        return tmp_path

    def test_dry_run_builds_orders_without_client_or_write(self, env, monkeypatch):
        monkeypatch.setattr(pb, "_client", lambda cfg: pytest.fail("dry-run must not build client"))
        cfg = {"alpaca": {"enabled": True, "paper": True,
                          "base_url": "https://paper-api.alpaca.markets"}}
        r = pb.submit_alerts(config=cfg, dry_run=True)
        assert r["submitted"] == 2
        assert r["dry_run"] is True
        # nothing persisted on dry-run
        assert not pb.ORDERS_PATH.exists()
        occs = {o["occ_symbol"] for o in r["orders"]}
        assert occs == {"ASTH260717P00050000", "NVDA260717C00200000"}


# ─── reconcile (pure) ───────────────────────────────────────────────────────────
class TestReconcile:
    def test_maps_fills_to_prediction_ids_with_slippage(self):
        orders = [
            {"order_id": "o1", "prediction_id": "ASTH:put:2026-07-02",
             "occ_symbol": "ASTH260717P00050000", "entry_mid": 1.73},
            {"order_id": "o2", "prediction_id": "NVDA:call:2026-07-02",
             "occ_symbol": "NVDA260717C00200000", "entry_mid": 0.92},
        ]
        status = {
            "o1": {"filled_avg_price": 1.80, "status": "filled", "filled_at": "2026-07-02T14:00Z"},
            "o2": {"filled_avg_price": None, "status": "new"},   # not yet filled
        }
        fills = pb.reconcile(orders, status)
        assert set(fills) == {"ASTH:put:2026-07-02"}            # only the filled one
        f = fills["ASTH:put:2026-07-02"]
        assert f["entry_fill"] == 1.80
        assert f["slippage_vs_mid"] == 0.07                      # 1.80 - 1.73

    def test_ignores_orders_without_id_or_prediction(self):
        orders = [{"occ_symbol": "X", "entry_mid": 1.0}]         # no order_id/prediction_id
        assert pb.reconcile(orders, {}) == {}


# ─── dedup helper ───────────────────────────────────────────────────────────────
class TestDedup:
    def test_submitted_today_reads_orders(self, tmp_path, monkeypatch):
        paper = tmp_path / "paper"; paper.mkdir()
        monkeypatch.setattr(pb, "ORDERS_PATH", paper / "paper_orders.jsonl")
        today = date.today().isoformat()
        (paper / "paper_orders.jsonl").write_text(
            json.dumps({"occ_symbol": "ASTH260717P00050000", "submitted_at": today + "T14:00:00Z"}) + "\n"
            + json.dumps({"occ_symbol": "OLD260101C00100000", "submitted_at": "2020-01-01T14:00:00Z"}) + "\n")
        assert pb._submitted_occ_today() == {"ASTH260717P00050000"}
