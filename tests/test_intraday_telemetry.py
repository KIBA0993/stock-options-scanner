"""Tests for intraday_telemetry snapshot + framework flag helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from intraday_telemetry import (
    append_telemetry,
    build_scan_snapshot,
    compute_anti_chase_flags,
    compute_freshness_flags,
    momentum_freshness_passes,
    telemetry_cfg,
)

ET = ZoneInfo("America/New_York")


def _sample_bars(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range(
        datetime(2026, 6, 24, 9, 30, tzinfo=ET),
        periods=n,
        freq="5min",
    )
    closes = [100.0 + i * 0.05 for i in range(n)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 0.1 for c in closes],
            "Low": [c - 0.1 for c in closes],
            "Close": closes,
            "Volume": [100_000] * n,
        },
        index=idx,
    )


class TestFreshnessFlags:
    def test_fresh_or_break_call(self) -> None:
        flags = compute_freshness_flags(
            "call", 101.0, 100.5, 100.2, 100.8, 99.5, 55.0, 50.0,
        )
        assert flags["fresh_or_break"] is True
        assert momentum_freshness_passes(flags)

    def test_no_freshness_when_extended(self) -> None:
        flags = compute_freshness_flags(
            "call", 102.0, 101.8, 100.0, 100.5, 99.5, 60.0, 58.0,
        )
        assert flags["fresh_or_break"] is False
        assert flags["fresh_vwap_cross"] is False


class TestAntiChase:
    def test_blocks_high_rsi_call(self) -> None:
        cfg = {"anti_chase_rsi_call": 62, "anti_chase_vwap_pct": 0.0015}
        flags = compute_anti_chase_flags("call", 100.0, 99.5, 65.0, cfg)
        assert flags["block_rsi"] is True
        assert flags["would_block"] is True

    def test_blocks_vwap_extension(self) -> None:
        cfg = {"anti_chase_rsi_call": 62, "anti_chase_vwap_pct": 0.0015}
        flags = compute_anti_chase_flags("call", 100.2, 100.0, 55.0, cfg)
        assert flags["block_vwap"] is True


class TestBuildSnapshot:
    def test_snapshot_has_required_backtest_fields(self, tmp_path: Path) -> None:
        config = {
            "intraday_0dte": {"min_score": 0.70, "or_minutes": 15},
            "intraday_telemetry": {"path": str(tmp_path / "tel.jsonl")},
            "framework_backtest": {},
            "budget": {"total_usd": 500},
        }
        tcfg = telemetry_cfg(config)
        cfg = config["intraday_0dte"]
        bars = _sample_bars()
        options = {"calls": [], "puts": [], "call_put_ratio": None}
        snap = build_scan_snapshot(
            "SPY",
            bars,
            options,
            cfg,
            tcfg,
            scan_timestamp=datetime.now(ET).isoformat(),
            scan_source="scan_5m",
            dedup_minutes=30,
            config=config,
        )
        for key in (
            "record_type", "scan_timestamp", "scan_source", "symbol",
            "prev_close", "prev_rsi", "vwap_dist_pct", "call_score", "put_score",
            "qualifies_entry", "would_fire_entry", "freshness_call", "anti_chase_call",
            "schema_version", "call_signals", "put_signals", "options_flow", "bar_latest",
        ):
            assert key in snap, f"missing {key}"
        assert snap["record_type"] == "scan_snapshot"
        assert snap["schema_version"] == 2


class TestAppendTelemetry:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        config = {
            "intraday_telemetry": {
                "enabled": True,
                "path": str(tmp_path / "tel.jsonl"),
            },
        }
        append_telemetry({"record_type": "scan_snapshot", "symbol": "SPY"}, config)
        lines = (tmp_path / "tel.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["symbol"] == "SPY"
