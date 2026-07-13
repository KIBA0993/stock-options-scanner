"""Tests for paper_broker.py — Alpaca PAPER execution scaffolding (P2-2).

Covers the offline/pure surface: OCC symbol construction, alert→order mapping,
the paper-only safety guard, enable-gating, dedup, and fill reconciliation.
No network and no alpaca-py import are exercised.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
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


# ─── exit order builder (pure) ───────────────────────────────────────────────────
class TestBuildExitOrder:
    def test_sell_to_close_market(self):
        entry = {"occ_symbol": "NVDA260717C00200000", "underlying": "NVDA",
                 "direction": "call", "strike": 200.0, "expiration": "2026-07-17",
                 "qty": 2, "entry_mid": 0.92, "order_id": "o1",
                 "prediction_id": "NVDA:call:2026-07-02"}
        x = pb.build_exit_order(entry)
        assert x["side"] == "sell"
        assert x["type"] == "market"
        assert x["limit_price"] is None
        assert x["qty"] == 2
        assert x["closes_prediction_id"] == "NVDA:call:2026-07-02"
        assert x["entry_order_id"] == "o1"

    def test_qty_override(self):
        entry = {"occ_symbol": "X", "qty": 5, "prediction_id": "X:call:2026-07-02"}
        assert pb.build_exit_order(entry, qty=1)["qty"] == 1


# ─── hold-window helpers ─────────────────────────────────────────────────────────
class TestHoldWindow:
    def test_hold_days_defaults_to_max_horizon(self):
        assert pb._hold_trading_days({"validation": {"horizons_days": [1, 3, 5]}}) == 5

    def test_hold_days_explicit_override(self):
        assert pb._hold_trading_days({"alpaca": {"hold_trading_days": 2}}) == 2

    def test_entry_date_from_prediction_id(self):
        assert pb._entry_date_of({"prediction_id": "NVDA:call:2026-07-02"}) == date(2026, 7, 2)

    def test_entry_date_fallback_to_submitted_at(self):
        assert pb._entry_date_of({"submitted_at": "2026-07-02T14:00:00Z"}) == date(2026, 7, 2)

    def test_add_trading_days_skips_weekend(self):
        # Fri 2026-07-10 + 1 trading day = Mon 2026-07-13
        assert pb._add_trading_days(date(2026, 7, 10), 1) == date(2026, 7, 13)


# ─── exit selection (dry-run, no SDK) ────────────────────────────────────────────
class TestExitPositions:
    @pytest.fixture
    def orders_env(self, tmp_path, monkeypatch):
        paper = tmp_path / "paper"; paper.mkdir(parents=True)
        monkeypatch.setattr(pb, "PAPER_DIR", paper)
        monkeypatch.setattr(pb, "ORDERS_PATH", paper / "paper_orders.jsonl")
        today = date.today().isoformat()
        rows = [
            # held long ago → due for exit
            {"side": "buy", "occ_symbol": "AAA260717C00100000", "qty": 1,
             "order_id": "oA", "prediction_id": "AAA:call:2000-01-01"},
            # entered today → still inside the 5-session hold window
            {"side": "buy", "occ_symbol": "BBB260717C00100000", "qty": 1,
             "order_id": "oB", "prediction_id": f"BBB:call:{today}"},
            # held long ago but already has a sell order → skip
            {"side": "buy", "occ_symbol": "CCC260717C00100000", "qty": 1,
             "order_id": "oC", "prediction_id": "CCC:call:2000-01-01"},
            {"side": "sell", "occ_symbol": "CCC260717C00100000",
             "order_id": "oCx", "closes_prediction_id": "CCC:call:2000-01-01"},
        ]
        (paper / "paper_orders.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return paper

    def test_dry_run_selects_only_due_unclosed(self, orders_env, monkeypatch):
        monkeypatch.setattr(pb, "_client", lambda cfg: pytest.fail("dry-run must not build client"))
        cfg = {"alpaca": {"enabled": True, "paper": True,
                          "base_url": "https://paper-api.alpaca.markets"}}
        r = pb.exit_positions(config=cfg, dry_run=True)
        assert r["exited"] == 1
        occs = {o["occ_symbol"] for o in r["orders"]}
        assert occs == {"AAA260717C00100000"}
        assert r["orders"][0]["side"] == "sell"
        # nothing persisted on dry-run
        assert not (orders_env / "paper_orders.jsonl").read_text().count('"dry_run"')

    def test_noops_when_disabled(self, monkeypatch):
        monkeypatch.setattr(pb, "_client", lambda cfg: pytest.fail("must not build client"))
        r = pb.exit_positions(config={"alpaca": {"enabled": False}})
        assert r == {"enabled": False, "exited": 0, "skipped": 0}


# ─── reconcile with exit legs (pure) ─────────────────────────────────────────────
class TestReconcileExits:
    def test_folds_exit_fill_and_round_trip_pnl(self):
        orders = [
            {"side": "buy", "order_id": "o1", "prediction_id": "NVDA:call:2026-07-02",
             "occ_symbol": "NVDA260717C00200000", "entry_mid": 0.92},
            {"side": "sell", "order_id": "o2", "closes_prediction_id": "NVDA:call:2026-07-02",
             "occ_symbol": "NVDA260717C00200000"},
        ]
        status = {
            "o1": {"filled_avg_price": 1.00, "status": "filled", "filled_at": "t1"},
            "o2": {"filled_avg_price": 1.50, "status": "filled", "filled_at": "t2"},
        }
        f = pb.reconcile(orders, status)["NVDA:call:2026-07-02"]
        assert f["entry_fill"] == 1.00
        assert f["exit_fill"] == 1.50
        assert f["round_trip_pnl_pct"] == 50.0            # (1.5-1.0)/1.0

    def test_exit_without_entry_fill_has_no_round_trip(self):
        orders = [
            {"side": "sell", "order_id": "o2", "closes_prediction_id": "NVDA:call:2026-07-02",
             "occ_symbol": "NVDA260717C00200000"},
        ]
        status = {"o2": {"filled_avg_price": 1.50, "status": "filled"}}
        f = pb.reconcile(orders, status)["NVDA:call:2026-07-02"]
        assert f["exit_fill"] == 1.50
        assert "round_trip_pnl_pct" not in f
        assert "entry_fill" not in f

    def test_tags_strategy_on_fill(self):
        orders = [{"side": "buy", "order_id": "o1", "prediction_id": "SPY:call:T1",
                   "occ_symbol": "SPY260713C00550000", "strategy": "intraday", "entry_mid": 1.0}]
        status = {"o1": {"filled_avg_price": 1.1, "status": "filled"}}
        assert pb.reconcile(orders, status)["SPY:call:T1"]["strategy"] == "intraday"


# ─── intraday 0DTE paper ─────────────────────────────────────────────────────────
def _today_ts(hhmm="10:20:00"):
    return f"{date.today().isoformat()}T{hhmm}-04:00"


def _intraday_entry(symbol="SPY", direction="call", ts=None, mid=1.2, ask=1.3, strike=550):
    return {
        "symbol": symbol, "direction": direction, "score": 0.75,
        "alert_action": "entry", "scan_timestamp": ts or _today_ts(),
        "recommended_contract": {"tiers": {"atm": {
            "strike": strike, "expiration": date.today().isoformat(), "direction": direction,
            "bid": mid - 0.1, "ask": ask, "mid_price": mid, "cost_per_contract": mid * 100}}},
    }


def _intraday_exit(entry_ts, symbol="SPY", direction="call", ts=None):
    return {"symbol": symbol, "direction": direction, "alert_action": "exit",
            "scan_timestamp": ts or _today_ts("11:00:00"), "exit_for_entry_ts": entry_ts}


class TestIntradayHelpers:
    def test_prediction_id_includes_timestamp(self):
        a = _intraday_entry(ts="2026-07-13T10:20:00-04:00")
        assert pb.intraday_prediction_id(a) == "SPY:call:2026-07-13T10:20:00-04:00"

    def test_entries_on_filters_action_direction_and_day(self):
        pool = [
            _intraday_entry(ts=_today_ts("10:00:00")),
            _intraday_exit(_today_ts("10:00:00")),                       # exit — excluded
            {"symbol": "SPY", "direction": "skip", "alert_action": "entry",
             "scan_timestamp": _today_ts("10:05:00")},                    # skip dir — excluded
            _intraday_entry(ts="2020-01-01T10:00:00-04:00"),              # other day — excluded
        ]
        got = pb.intraday_entries_on(pool, date.today().isoformat())
        assert len(got) == 1 and got[0]["alert_action"] == "entry"

    def test_order_from_alert_tags_intraday(self):
        a = _intraday_entry(ts="2026-07-13T10:20:00-04:00", mid=1.2, ask=1.3)
        o = pb.intraday_order_from_alert(a, qty=1)
        assert o["side"] == "buy"
        assert o["strategy"] == "intraday"
        assert o["entry_ref_ts"] == "2026-07-13T10:20:00-04:00"
        assert o["prediction_id"] == "SPY:call:2026-07-13T10:20:00-04:00"
        assert o["limit_price"] == 1.3                                    # buys at the ask

    def test_order_from_alert_none_when_unpriced(self):
        a = _intraday_entry()
        a["recommended_contract"] = {"tiers": {}}
        assert pb.intraday_order_from_alert(a) is None

    def test_eod_exit_time_boundary(self):
        cfg = {"eod_exit_enabled": True, "eod_exit_time": "15:45"}
        assert pb._is_past_eod_exit(cfg, datetime(2026, 7, 13, 15, 50, tzinfo=pb.ET)) is True
        assert pb._is_past_eod_exit(cfg, datetime(2026, 7, 13, 10, 0, tzinfo=pb.ET)) is False
        assert pb._is_past_eod_exit({"eod_exit_enabled": False},
                                    datetime(2026, 7, 13, 15, 50, tzinfo=pb.ET)) is False


class TestSelectIntradayExits:
    def _entry_order(self, ref="T1", pid="SPY:call:T1", occ="SPY260713C00550000"):
        return {"strategy": "intraday", "side": "buy", "occ_symbol": occ,
                "entry_ref_ts": ref, "prediction_id": pid}

    def test_closes_on_matching_exit_alert(self):
        orders = [self._entry_order()]
        pool = [_intraday_exit("T1")]
        due = pb.select_intraday_exits(orders, pool, closed_pids=set(), eod=False)
        assert [o["prediction_id"] for o in due] == ["SPY:call:T1"]

    def test_closes_all_at_eod(self):
        orders = [self._entry_order()]
        due = pb.select_intraday_exits(orders, [], closed_pids=set(), eod=True)
        assert len(due) == 1

    def test_skips_when_no_exit_and_not_eod(self):
        orders = [self._entry_order()]
        assert pb.select_intraday_exits(orders, [], closed_pids=set(), eod=False) == []

    def test_skips_already_closed(self):
        orders = [self._entry_order()]
        due = pb.select_intraday_exits(orders, [_intraday_exit("T1")],
                                       closed_pids={"SPY:call:T1"}, eod=True)
        assert due == []


class TestIntradayGating:
    def test_disabled_when_alpaca_off(self):
        assert pb.intraday_enabled({"alpaca": {"enabled": False}}) is False

    def test_on_by_default_when_alpaca_on(self):
        assert pb.intraday_enabled({"alpaca": {"enabled": True}}) is True

    def test_opt_out_flag(self):
        assert pb.intraday_enabled({"alpaca": {"enabled": True, "intraday_enabled": False}}) is False

    def test_submit_noops_when_disabled(self, monkeypatch):
        monkeypatch.setattr(pb, "_client", lambda cfg: pytest.fail("must not build client"))
        r = pb.submit_intraday(config={"alpaca": {"enabled": True, "intraday_enabled": False}})
        assert r == {"enabled": False, "submitted": 0, "skipped": 0}


class TestIntradaySubmitDryRun:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        paper = tmp_path / "paper"; paper.mkdir(parents=True)
        monkeypatch.setattr(pb, "PAPER_DIR", paper)
        monkeypatch.setattr(pb, "ORDERS_PATH", paper / "paper_orders.jsonl")
        monkeypatch.setattr(pb, "INTRADAY_ALERTS_PATH", tmp_path / "intraday.jsonl")
        rows = [
            _intraday_entry("SPY", "call", ts=_today_ts("10:20:00")),
            _intraday_entry("QQQ", "put", ts=_today_ts("10:20:05"), strike=480),
        ]
        (tmp_path / "intraday.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return tmp_path

    def test_dry_run_submits_priced_entries(self, env, monkeypatch):
        monkeypatch.setattr(pb, "_client", lambda cfg: pytest.fail("dry-run must not build client"))
        cfg = {"alpaca": {"enabled": True, "paper": True,
                          "base_url": "https://paper-api.alpaca.markets"}}
        r = pb.submit_intraday(config=cfg, dry_run=True)
        assert r["submitted"] == 2
        assert all(o["strategy"] == "intraday" for o in r["orders"])
        assert not pb.ORDERS_PATH.exists()                               # nothing persisted on dry-run

    def test_dedups_against_already_submitted(self, env, monkeypatch):
        monkeypatch.setattr(pb, "_client", lambda cfg: pytest.fail("dry-run must not build client"))
        # pre-record one of the two entries as already submitted
        pb.ORDERS_PATH.write_text(json.dumps(
            {"strategy": "intraday", "entry_ref_ts": _today_ts("10:20:00"),
             "occ_symbol": "SPY260713C00550000"}) + "\n")
        cfg = {"alpaca": {"enabled": True, "paper": True,
                          "base_url": "https://paper-api.alpaca.markets"}}
        r = pb.submit_intraday(config=cfg, dry_run=True)
        assert r["submitted"] == 1                                       # only the QQQ entry
        assert r["skipped"] == 1


# ─── Alpaca-side contract pricing (Option A) ─────────────────────────────────────
class TestDteAndOcc:
    def test_parse_dte_window(self):
        assert pb._parse_dte_window({"options": {"preferred_dte": "7-14 days"}}) == (7, 14)
        assert pb._parse_dte_window({"options": {"preferred_dte": "10 days"}}) == (10, 10)
        assert pb._parse_dte_window({}) == (7, 14)

    def test_parse_occ_roundtrip(self):
        occ = pb.build_occ_symbol("IWM", "2026-07-23", "put", 298.0)
        assert occ == "IWM260723P00298000"
        exp, cp, strike = pb._parse_occ(occ)
        assert (exp, cp, strike) == ("2026-07-23", "P", 298.0)


class TestSelectAlpacaContract:
    def _snaps(self):
        return [
            {"strike": 298.0, "expiration": "2026-07-23", "bid": 4.55, "ask": 4.78},  # ATM-ish
            {"strike": 290.0, "expiration": "2026-07-23", "bid": 2.00, "ask": 2.10},  # further OTM
            {"strike": 350.0, "expiration": "2026-07-23", "bid": 9.00, "ask": 9.20},  # >15% away
            {"strike": 296.0, "expiration": "2026-07-23", "bid": 1.00, "ask": 3.00},  # spread too wide
        ]

    def test_picks_nearest_atm_within_budget(self):
        c = pb.select_alpaca_contract(295.42, self._snaps(), "put", budget=500, max_spread_pct=50)
        t = c["tiers"]["atm"]
        assert t["strike"] == 298.0                       # closest to 295.42 among valid
        assert t["mid_price"] == 4.67                      # round((4.55+4.78)/2, 2)
        assert t["source"] == "alpaca_indicative"

    def test_excludes_wide_spread_and_far_strikes(self):
        # tighten spread cap so 296 (100% spread) is gone; 350 excluded by 15% window
        c = pb.select_alpaca_contract(295.42, self._snaps(), "put", budget=500, max_spread_pct=10)
        strikes_considered = {298.0, 290.0}
        assert c["tiers"]["atm"]["strike"] in strikes_considered

    def test_falls_back_to_cheapest_when_none_in_budget(self):
        # budget below every contract's cost → still returns nearest-ATM (no skip)
        c = pb.select_alpaca_contract(295.42, self._snaps(), "put", budget=10, max_spread_pct=50)
        assert c is not None
        assert c["tiers"]["atm"]["strike"] == 298.0

    def test_none_when_no_valid_contracts(self):
        assert pb.select_alpaca_contract(100.0, [], "put") is None
        one_sided = [{"strike": 100, "expiration": "2026-07-23", "bid": 0, "ask": 1.0}]
        assert pb.select_alpaca_contract(100.0, one_sided, "put") is None


class TestEnsurePricedContract:
    def _priced(self):
        return {"symbol": "NVDA", "direction": "call", "recommended_contract": {"tiers": {"atm": {
            "strike": 100, "expiration": "2026-07-24", "mid_price": 2.0, "ask": 2.1}}}}

    def _unpriced(self):
        return {"symbol": "IWM", "direction": "put",
                "recommended_contract": {"tiers": None, "notes": "no liquid options"}}

    def test_keeps_scan_priced_contract(self, monkeypatch):
        monkeypatch.setattr(pb, "price_contract_via_alpaca",
                            lambda *a, **k: pytest.fail("must not call Alpaca when already priced"))
        alert, src = pb.ensure_priced_contract(self._priced(), {"alpaca": {}})
        assert src == "scan"

    def test_prices_via_alpaca_when_unpriced(self, monkeypatch):
        built = {"tiers": {"atm": {"strike": 298, "expiration": "2026-07-23",
                                   "mid_price": 4.66, "ask": 4.78, "source": "alpaca_indicative"}}}
        monkeypatch.setattr(pb, "price_contract_via_alpaca", lambda s, d, c: built)
        alert, src = pb.ensure_priced_contract(self._unpriced(), {"alpaca": {}})
        assert src == "alpaca"
        assert alert["recommended_contract"] is built

    def test_gate_off_skips_alpaca(self, monkeypatch):
        monkeypatch.setattr(pb, "price_contract_via_alpaca",
                            lambda *a, **k: pytest.fail("gate off — must not call Alpaca"))
        _, src = pb.ensure_priced_contract(self._unpriced(), {"alpaca": {"price_via_alpaca": False}})
        assert src == "unpriced"

    def test_unpriced_when_alpaca_returns_none(self, monkeypatch):
        monkeypatch.setattr(pb, "price_contract_via_alpaca", lambda s, d, c: None)
        _, src = pb.ensure_priced_contract(self._unpriced(), {"alpaca": {}})
        assert src == "unpriced"


class TestSubmitUsesAlpacaPricing:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        data = tmp_path / "data"; paper = data / "paper"; paper.mkdir(parents=True)
        monkeypatch.setattr(pb, "DATA_DIR", data)
        monkeypatch.setattr(pb, "PAPER_DIR", paper)
        monkeypatch.setattr(pb, "ORDERS_PATH", paper / "paper_orders.jsonl")
        monkeypatch.setattr(pb, "ALERTS_PATH", data / "alerts.json")
        (data / "alerts.json").write_text(json.dumps({
            "scan_timestamp": "2026-07-13T13:46:00+00:00",
            "alerts": [{"symbol": "IWM", "direction": "put", "score": 0.8,
                        "recommended_contract": {"tiers": None}}]}))
        return tmp_path

    def test_unpriced_alert_gets_alpaca_priced_and_submitted(self, env, monkeypatch):
        monkeypatch.setattr(pb, "_client", lambda cfg: pytest.fail("dry-run must not build client"))
        built = {"tiers": {"atm": {"strike": 298, "expiration": "2026-07-23",
                                   "mid_price": 4.66, "ask": 4.78, "source": "alpaca_indicative"}}}
        monkeypatch.setattr(pb, "price_contract_via_alpaca", lambda s, d, c: built)
        cfg = {"alpaca": {"enabled": True, "paper": True,
                          "base_url": "https://paper-api.alpaca.markets"}}
        r = pb.submit_alerts(config=cfg, dry_run=True)
        assert r["submitted"] == 1
        assert r["alpaca_priced"] == 1
        o = r["orders"][0]
        assert o["occ_symbol"] == "IWM260723P00298000"
        assert o["contract_source"] == "alpaca"
        assert o["limit_price"] == 4.78                    # buys at the Alpaca ask
