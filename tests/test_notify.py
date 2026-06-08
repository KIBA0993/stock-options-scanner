"""
tests/test_notify.py — Unit tests for notify.py (Week 4)

Run: cd ~/trading && python -m pytest tests/test_notify.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from notify import (
    _fingerprint,
    filter_new_alerts,
    format_email_html,
    format_email_text,
    format_telegram,
    format_telegram_summary,
    record_sent,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────
SCAN_TS = "2026-06-08T01:29:27.581677+00:00"

@pytest.fixture
def call_alert() -> dict:
    return {
        "symbol":              "NVDA",
        "score":               0.85,
        "direction":           "call",
        "rationale":           "Bullish EMA stack + RSI momentum + strong call flow.",
        "supporting_creators": ["kpak82"],
        "key_signals": [
            "Bullish EMA stack (20>50>200)",
            "RSI momentum zone (62)",
            "Heavy call flow (C/P: 3.1x)",
        ],
        "suggested_dte":  "7-14 days",
        "risk_level":     "medium",
        "skip_reason":    None,
        "scoring_method": "heuristic",
    }


@pytest.fixture
def put_alert() -> dict:
    return {
        "symbol":              "SPY",
        "score":               0.78,
        "direction":           "put",
        "rationale":           "RSI extreme overbought + bearish EMA + heavy put flow.",
        "supporting_creators": ["kpak82", "MasterPandaWu"],
        "key_signals": [
            "RSI extreme overbought (82)",
            "Bearish EMA stack",
            "Heavy put flow (C/P: 0.4x)",
        ],
        "suggested_dte":  "7-14 days",
        "risk_level":     "medium",
        "skip_reason":    None,
        "scoring_method": "llm",
    }


@pytest.fixture
def earnings_alert() -> dict:
    return {
        "symbol":              "MSFT",
        "score":               0.61,
        "direction":           "call",
        "rationale":           "Bullish signals but earnings within 48h.",
        "supporting_creators": [],
        "key_signals": ["⚠️ EARNINGS WITHIN 48H (score reduced)", "TV strong buy"],
        "suggested_dte":  "7-21 days",
        "risk_level":     "high",
        "skip_reason":    None,
        "scoring_method": "heuristic",
    }


# ─── _fingerprint ──────────────────────────────────────────────────────────────
class TestFingerprint:
    def test_includes_symbol_direction_and_timestamp(self, call_alert: dict) -> None:
        fp = _fingerprint(call_alert, SCAN_TS)
        assert "NVDA" in fp
        assert "call" in fp
        assert SCAN_TS in fp

    def test_different_directions_produce_different_fingerprints(
        self, call_alert: dict, put_alert: dict
    ) -> None:
        fp_call = _fingerprint(call_alert, SCAN_TS)
        fp_put  = _fingerprint(put_alert, SCAN_TS)
        assert fp_call != fp_put

    def test_different_timestamps_produce_different_fingerprints(
        self, call_alert: dict
    ) -> None:
        fp1 = _fingerprint(call_alert, "2026-06-07T10:00:00")
        fp2 = _fingerprint(call_alert, "2026-06-08T10:00:00")
        assert fp1 != fp2


# ─── filter_new_alerts ─────────────────────────────────────────────────────────
class TestFilterNewAlerts:
    def test_returns_all_when_history_empty(
        self, call_alert: dict, put_alert: dict
    ) -> None:
        result = filter_new_alerts(
            [call_alert, put_alert], SCAN_TS, history={}, force=False
        )
        assert len(result) == 2

    def test_filters_out_already_sent(self, call_alert: dict, put_alert: dict) -> None:
        fp = _fingerprint(call_alert, SCAN_TS)
        history = {fp: {"symbol": "NVDA", "sent_at": "2026-06-08"}}
        result = filter_new_alerts(
            [call_alert, put_alert], SCAN_TS, history=history, force=False
        )
        assert len(result) == 1
        assert result[0]["symbol"] == "SPY"

    def test_force_returns_all_regardless_of_history(
        self, call_alert: dict, put_alert: dict
    ) -> None:
        fp = _fingerprint(call_alert, SCAN_TS)
        history = {fp: {"symbol": "NVDA"}}
        result = filter_new_alerts(
            [call_alert, put_alert], SCAN_TS, history=history, force=True
        )
        assert len(result) == 2

    def test_empty_alert_list(self) -> None:
        result = filter_new_alerts([], SCAN_TS, history={}, force=False)
        assert result == []


# ─── record_sent ───────────────────────────────────────────────────────────────
class TestRecordSent:
    def test_writes_fingerprint_to_file(
        self, call_alert: dict, tmp_path: Path
    ) -> None:
        with patch("notify.SENT_PATH", tmp_path / "sent_history.json"):
            record_sent([call_alert], SCAN_TS, channels=["telegram"])
            history = json.loads((tmp_path / "sent_history.json").read_text())

        fp = _fingerprint(call_alert, SCAN_TS)
        assert fp in history
        assert history[fp]["symbol"] == "NVDA"
        assert "telegram" in history[fp]["channels"]

    def test_appends_to_existing_history(
        self, call_alert: dict, put_alert: dict, tmp_path: Path
    ) -> None:
        path = tmp_path / "sent_history.json"
        existing_fp = _fingerprint(put_alert, SCAN_TS)
        path.write_text(json.dumps({existing_fp: {"symbol": "SPY"}}))

        with patch("notify.SENT_PATH", path):
            record_sent([call_alert], SCAN_TS, channels=["telegram"])
            history = json.loads(path.read_text())

        assert len(history) == 2
        assert existing_fp in history


# ─── format_telegram ───────────────────────────────────────────────────────────
class TestFormatTelegram:
    def test_call_alert_contains_symbol_and_direction(
        self, call_alert: dict
    ) -> None:
        msg = format_telegram(call_alert, SCAN_TS, context=None)
        assert "NVDA" in msg
        assert "CALL" in msg

    def test_put_alert_contains_put_emoji(self, put_alert: dict) -> None:
        msg = format_telegram(put_alert, SCAN_TS, context=None)
        assert "📉" in msg
        assert "PUT" in msg

    def test_call_alert_contains_call_emoji(self, call_alert: dict) -> None:
        msg = format_telegram(call_alert, SCAN_TS, context=None)
        assert "📈" in msg

    def test_includes_score_as_percentage(self, call_alert: dict) -> None:
        msg = format_telegram(call_alert, SCAN_TS, context=None)
        assert "85%" in msg

    def test_includes_dte(self, call_alert: dict) -> None:
        msg = format_telegram(call_alert, SCAN_TS, context=None)
        assert "7-14 days" in msg

    def test_includes_creator_handles(self, call_alert: dict) -> None:
        msg = format_telegram(call_alert, SCAN_TS, context=None)
        assert "@kpak82" in msg

    def test_includes_context_when_provided(self, call_alert: dict) -> None:
        msg = format_telegram(call_alert, SCAN_TS, context="Fed day, cautious")
        assert "Fed day" in msg

    def test_no_context_line_when_none(self, call_alert: dict) -> None:
        msg = format_telegram(call_alert, SCAN_TS, context=None)
        assert "Context" not in msg

    def test_includes_signals(self, call_alert: dict) -> None:
        msg = format_telegram(call_alert, SCAN_TS, context=None)
        assert "Bullish EMA stack" in msg

    def test_llm_scoring_shows_llm_label(self, put_alert: dict) -> None:
        msg = format_telegram(put_alert, SCAN_TS, context=None)
        assert "LLM" in msg

    def test_heuristic_scoring_shows_heuristic_label(self, call_alert: dict) -> None:
        msg = format_telegram(call_alert, SCAN_TS, context=None)
        assert "Heuristic" in msg

    def test_risk_emojis_present(self, call_alert: dict) -> None:
        msg = format_telegram(call_alert, SCAN_TS, context=None)
        # medium risk → 🟡
        assert "🟡" in msg

    def test_high_risk_emoji(self, earnings_alert: dict) -> None:
        msg = format_telegram(earnings_alert, SCAN_TS, context=None)
        assert "🔴" in msg


# ─── format_telegram_summary ───────────────────────────────────────────────────
class TestFormatTelegramSummary:
    def test_shows_count(self, call_alert: dict, put_alert: dict) -> None:
        msg = format_telegram_summary([call_alert, put_alert], SCAN_TS, context=None)
        assert "2" in msg

    def test_singular_for_one_alert(self, call_alert: dict) -> None:
        msg = format_telegram_summary([call_alert], SCAN_TS, context=None)
        assert "Alert" in msg
        # Should not be "Alerts" for 1
        assert "1 New Trade Alert" in msg

    def test_includes_context(self, call_alert: dict) -> None:
        msg = format_telegram_summary([call_alert], SCAN_TS, context="bearish macro")
        assert "bearish macro" in msg


# ─── format_email_text ─────────────────────────────────────────────────────────
class TestFormatEmailText:
    def test_includes_symbol(self, call_alert: dict) -> None:
        text = format_email_text([call_alert], SCAN_TS, context=None)
        assert "NVDA" in text

    def test_includes_direction(self, call_alert: dict) -> None:
        text = format_email_text([call_alert], SCAN_TS, context=None)
        assert "CALL" in text

    def test_includes_all_alerts(self, call_alert: dict, put_alert: dict) -> None:
        text = format_email_text([call_alert, put_alert], SCAN_TS, context=None)
        assert "NVDA" in text
        assert "SPY" in text

    def test_includes_scan_timestamp(self) -> None:
        text = format_email_text([], SCAN_TS, context=None)
        assert "2026-06-08" in text

    def test_includes_context(self, call_alert: dict) -> None:
        text = format_email_text([call_alert], SCAN_TS, context="bearish")
        assert "bearish" in text


# ─── format_email_html ─────────────────────────────────────────────────────────
class TestFormatEmailHtml:
    def test_is_valid_html(self, call_alert: dict) -> None:
        html = format_email_html([call_alert], SCAN_TS, context=None)
        assert "<html>" in html
        assert "</html>" in html

    def test_includes_symbol(self, call_alert: dict) -> None:
        html = format_email_html([call_alert], SCAN_TS, context=None)
        assert "NVDA" in html

    def test_includes_direction(self, call_alert: dict) -> None:
        html = format_email_html([call_alert], SCAN_TS, context=None)
        assert "CALL" in html

    def test_call_uses_green_color(self, call_alert: dict) -> None:
        html = format_email_html([call_alert], SCAN_TS, context=None)
        assert "#1a7f37" in html  # green for calls

    def test_put_uses_red_color(self, put_alert: dict) -> None:
        html = format_email_html([put_alert], SCAN_TS, context=None)
        assert "#cf222e" in html  # red for puts

    def test_includes_score_percentage(self, call_alert: dict) -> None:
        html = format_email_html([call_alert], SCAN_TS, context=None)
        assert "85%" in html

    def test_includes_key_signals(self, call_alert: dict) -> None:
        html = format_email_html([call_alert], SCAN_TS, context=None)
        assert "Bullish EMA stack" in html

    def test_shows_alert_count_in_heading(
        self, call_alert: dict, put_alert: dict
    ) -> None:
        html = format_email_html([call_alert, put_alert], SCAN_TS, context=None)
        assert "signal" in html  # heading contains "2 new signals"
