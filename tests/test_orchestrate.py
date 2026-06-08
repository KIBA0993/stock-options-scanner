"""
tests/test_orchestrate.py — Unit tests for orchestrate.py (Week 3)

Run: cd ~/trading && python -m pytest tests/test_orchestrate.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the trading root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrate import (
    _extract_sections,
    _framework_summary,
    _parse_llm_json,
    _skip,
    _ticker_summary,
    filter_alerts,
    heuristic_score,
    load_creator_frameworks,
    score_candidates,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def strong_call_ticker() -> dict:
    """Ticker with clearly bullish signals."""
    return {
        "symbol": "NVDA",
        "name": "NVIDIA Corp",
        "price": 872.40,
        "change_pct": 6.5,
        "relative_volume": 4.2,
        "volume": 45_000_000,
        "avg_volume_10d": 10_700_000,
        "earnings_within_48h": False,
        "patterns": {
            "rsi": 58.0,
            "macd": 4.21,
            "macd_signal": 3.87,
            "ema20": 840.0,
            "ema50": 810.0,
            "ema200": 750.0,
            "ema_alignment": "bullish",
            "tv_recommendation": 0.45,
        },
        "options_total_volume": 63_000,
        "options_call_volume": 45_000,
        "options_put_volume": 18_000,
        "call_put_ratio": 2.5,
        "options_chain": {"calls": [], "puts": []},
        "news": [{"title": "NVDA announces AI partnership", "publisher": "Reuters",
                  "published_at": "2026-06-07", "summary": "..."}],
    }


@pytest.fixture
def strong_put_ticker() -> dict:
    """Ticker with clearly bearish / reversal signals (kpak82 style)."""
    return {
        "symbol": "SPY",
        "name": "S&P 500 ETF",
        "price": 530.00,
        "change_pct": -1.2,
        "relative_volume": 2.8,
        "volume": 90_000_000,
        "avg_volume_10d": 32_000_000,
        "earnings_within_48h": False,
        "patterns": {
            "rsi": 82.0,
            "macd": -0.50,
            "macd_signal": 0.10,
            "ema20": 528.0,
            "ema50": 532.0,
            "ema200": 490.0,
            "ema_alignment": "bearish",
            "tv_recommendation": -0.35,
        },
        "options_total_volume": 200_000,
        "options_call_volume": 60_000,
        "options_put_volume": 140_000,
        "call_put_ratio": 0.43,
        "options_chain": {"calls": [], "puts": []},
        "news": [],
    }


@pytest.fixture
def no_signal_ticker() -> dict:
    """Ticker with neutral / mixed signals."""
    return {
        "symbol": "AAPL",
        "name": "Apple Inc",
        "price": 195.0,
        "change_pct": 0.3,
        "relative_volume": 1.8,
        "volume": 55_000_000,
        "avg_volume_10d": 30_000_000,
        "earnings_within_48h": False,
        "patterns": {
            "rsi": 52.0,
            "macd": 0.01,
            "macd_signal": 0.01,
            "ema20": 194.0,
            "ema50": 193.0,
            "ema200": 185.0,
            "ema_alignment": "bullish",
            "tv_recommendation": 0.05,
        },
        "options_total_volume": 30_000,
        "options_call_volume": 16_000,
        "options_put_volume": 14_000,
        "call_put_ratio": 1.14,
        "options_chain": {"calls": [], "puts": []},
        "news": [],
    }


@pytest.fixture
def earnings_ticker(strong_call_ticker: dict) -> dict:
    """Bullish ticker but with earnings in 48h."""
    t = dict(strong_call_ticker)
    t["symbol"] = "MSFT"
    t["earnings_within_48h"] = True
    return t


@pytest.fixture
def sample_framework_md() -> str:
    return """# Trading Framework: @kpak82
Distilled: 2026-06-07
Version: v1
Asset focus: US equities

## Trading Personality
Pure technician.

## Setup Triggers — What Gets Their Attention
1. Multi-decade channel resistance
2. RSI negative divergence at tops
3. VIX divergence

## Entry Rules
1. Wait for reversal candle at resistance
2. 4H candle close as confirmation

## Red Flags — What They Explicitly Avoid
- Don't chase extreme overbought
- Avoid boredom trades

## Market Conditions They Avoid
- News-driven moves
- Choppy sideways markets
"""


# ─── _extract_sections ─────────────────────────────────────────────────────────
class TestExtractSections:
    def test_extracts_matching_section(self, sample_framework_md: str) -> None:
        result = _extract_sections(sample_framework_md, ["Setup Triggers"])
        assert "## Setup Triggers" in result
        assert "Multi-decade channel resistance" in result

    def test_extracts_multiple_sections(self, sample_framework_md: str) -> None:
        result = _extract_sections(sample_framework_md, ["Entry Rules", "Red Flags"])
        assert "## Entry Rules" in result
        assert "## Red Flags" in result
        assert "reversal candle" in result
        assert "boredom trades" in result

    def test_excludes_non_matching_sections(self, sample_framework_md: str) -> None:
        result = _extract_sections(sample_framework_md, ["Entry Rules"])
        assert "Trading Personality" not in result
        assert "Pure technician" not in result

    def test_empty_result_for_missing_section(self, sample_framework_md: str) -> None:
        result = _extract_sections(sample_framework_md, ["Exit Rules"])
        assert result == ""

    def test_case_insensitive_matching(self, sample_framework_md: str) -> None:
        result = _extract_sections(sample_framework_md, ["setup triggers"])
        assert "## Setup Triggers" in result


# ─── _framework_summary ────────────────────────────────────────────────────────
class TestFrameworkSummary:
    def test_includes_handle_and_weight(self, sample_framework_md: str) -> None:
        result = _framework_summary("kpak82", 1.0, sample_framework_md)
        assert "@kpak82" in result
        assert "weight: 1.0x" in result

    def test_truncates_long_content(self) -> None:
        long_md = "## Setup Triggers\n" + ("x " * 5000)
        result  = _framework_summary("test", 0.5, long_md)
        assert len(result) <= 3_300  # 3000 content + header overhead
        assert "[...truncated...]" in result


# ─── _ticker_summary ───────────────────────────────────────────────────────────
class TestTickerSummary:
    def test_includes_symbol(self, strong_call_ticker: dict) -> None:
        result = _ticker_summary(strong_call_ticker)
        assert "NVDA" in result

    def test_includes_rsi(self, strong_call_ticker: dict) -> None:
        result = _ticker_summary(strong_call_ticker)
        assert "58" in result  # RSI 58.0

    def test_includes_earnings_warning(self, earnings_ticker: dict) -> None:
        result = _ticker_summary(earnings_ticker)
        assert "YES" in result

    def test_no_earnings_flag(self, strong_call_ticker: dict) -> None:
        result = _ticker_summary(strong_call_ticker)
        assert "YES" not in result

    def test_includes_call_put_ratio(self, strong_call_ticker: dict) -> None:
        result = _ticker_summary(strong_call_ticker)
        assert "2.50" in result  # call_put_ratio

    def test_handles_missing_rsi(self) -> None:
        ticker = {
            "symbol": "X", "name": "X", "price": 10.0, "change_pct": 0,
            "relative_volume": 1.0, "earnings_within_48h": False,
            "patterns": {}, "news": [],
            "options_call_volume": 0, "options_put_volume": 0,
            "call_put_ratio": None,
        }
        result = _ticker_summary(ticker)
        assert "N/A" in result


# ─── _parse_llm_json ───────────────────────────────────────────────────────────
class TestParseLlmJson:
    def test_parses_clean_json(self) -> None:
        raw = json.dumps({
            "evaluations": [
                {"symbol": "NVDA", "score": 0.8, "direction": "call",
                 "rationale": "Test", "supporting_creators": ["kpak82"],
                 "key_signals": [], "suggested_dte": "7-14 days",
                 "risk_level": "medium", "skip_reason": None}
            ]
        })
        result = _parse_llm_json(raw)
        assert len(result) == 1
        assert result[0]["symbol"] == "NVDA"
        assert result[0]["score"] == 0.8

    def test_strips_markdown_fences(self) -> None:
        raw = "```json\n" + json.dumps({"evaluations": [{"symbol": "SPY"}]}) + "\n```"
        result = _parse_llm_json(raw)
        assert len(result) == 1
        assert result[0]["symbol"] == "SPY"

    def test_returns_empty_on_invalid_json(self) -> None:
        result = _parse_llm_json("this is not json")
        assert result == []

    def test_returns_empty_on_missing_evaluations_key(self) -> None:
        result = _parse_llm_json(json.dumps({"data": []}))
        assert result == []


# ─── heuristic_score ───────────────────────────────────────────────────────────
class TestHeuristicScore:
    def test_strong_call_ticker_scores_above_threshold(
        self, strong_call_ticker: dict
    ) -> None:
        result = heuristic_score(strong_call_ticker)
        assert result["direction"] == "call"
        assert result["score"] >= 0.60

    def test_strong_put_ticker_scores_put(self, strong_put_ticker: dict) -> None:
        result = heuristic_score(strong_put_ticker)
        assert result["direction"] == "put"
        assert result["score"] >= 0.60

    def test_kpak82_extreme_rsi_bearish_ema_bonus_applied(
        self, strong_put_ticker: dict
    ) -> None:
        # RSI 82 + bearish EMA → kpak82 reversal bonus
        result = heuristic_score(strong_put_ticker)
        assert result["score"] >= 0.75
        assert "kpak82" in result["supporting_creators"]

    def test_earnings_penalty_reduces_score(self, earnings_ticker: dict) -> None:
        without = heuristic_score(dict(earnings_ticker) | {"earnings_within_48h": False})
        with_   = heuristic_score(earnings_ticker)
        assert with_["score"] < without["score"]
        assert any("EARNINGS" in s for s in with_["key_signals"])

    def test_no_signal_ticker_returns_skip_or_low_score(
        self, no_signal_ticker: dict
    ) -> None:
        result = heuristic_score(no_signal_ticker)
        # AAPL has bullish EMA + bullish MACD (barely) → low call, not necessarily skip
        # Just confirm score is below 0.7 (not a strong signal)
        assert result["score"] < 0.7

    def test_puppy_trades_sector_match_adds_bonus(self) -> None:
        ticker = {
            "symbol": "FSLR",  # in _PUPPY_TICKERS
            "name": "First Solar",
            "price": 200.0,
            "change_pct": 3.0,
            "relative_volume": 2.0,
            "earnings_within_48h": False,
            "patterns": {
                "rsi": 55.0,
                "macd": 1.0, "macd_signal": 0.5,
                "ema_alignment": "bullish",
                "tv_recommendation": 0.2,
            },
            "news": [],
            "options_call_volume": 5000,
            "options_put_volume": 2000,
            "call_put_ratio": 2.5,
        }
        result = heuristic_score(ticker)
        assert "puppy_trades" in result["supporting_creators"]
        assert result["direction"] == "call"

    def test_scoring_method_is_heuristic(self, strong_call_ticker: dict) -> None:
        result = heuristic_score(strong_call_ticker)
        assert result["scoring_method"] == "heuristic"

    def test_output_has_required_keys(self, strong_call_ticker: dict) -> None:
        result = heuristic_score(strong_call_ticker)
        required = [
            "symbol", "score", "direction", "rationale",
            "supporting_creators", "key_signals", "suggested_dte",
            "risk_level", "skip_reason", "scoring_method",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_score_is_bounded_0_to_1(self, strong_call_ticker: dict) -> None:
        result = heuristic_score(strong_call_ticker)
        assert 0.0 <= result["score"] <= 1.0


# ─── _skip helper ──────────────────────────────────────────────────────────────
class TestSkipHelper:
    def test_skip_sets_direction(self) -> None:
        result = _skip("X", "reason")
        assert result["direction"] == "skip"

    def test_skip_sets_reason(self) -> None:
        result = _skip("X", "no signals")
        assert result["skip_reason"] == "no signals"

    def test_skip_with_base_score(self) -> None:
        result = _skip("X", "reason", 0.12)
        assert result["score"] == pytest.approx(0.12)


# ─── filter_alerts ─────────────────────────────────────────────────────────────
class TestFilterAlerts:
    def _make_scored(self, sym: str, score: float, direction: str = "call") -> dict:
        return {
            "symbol": sym, "score": score, "direction": direction,
            "rationale": "", "supporting_creators": [], "key_signals": [],
            "suggested_dte": "7-14 days", "risk_level": "medium",
            "skip_reason": None, "scoring_method": "heuristic",
        }

    def test_filters_below_min_score(self) -> None:
        scored = [
            self._make_scored("NVDA", 0.80),
            self._make_scored("AAPL", 0.45),
            self._make_scored("SPY",  0.72),
        ]
        alerts = filter_alerts(scored, min_score=0.60, max_alerts=10)
        assert len(alerts) == 2
        assert all(a["score"] >= 0.60 for a in alerts)

    def test_removes_skips(self) -> None:
        scored = [
            self._make_scored("NVDA", 0.80),
            self._make_scored("AAPL", 0.20, direction="skip"),
        ]
        alerts = filter_alerts(scored, min_score=0.10, max_alerts=10)
        assert all(a["direction"] != "skip" for a in alerts)

    def test_respects_max_alerts(self) -> None:
        scored = [self._make_scored(f"T{i}", 0.9 - i * 0.05) for i in range(6)]
        alerts = filter_alerts(scored, min_score=0.60, max_alerts=2)
        assert len(alerts) == 2

    def test_sorted_by_score_descending(self) -> None:
        scored = [
            self._make_scored("A", 0.65),
            self._make_scored("B", 0.85),
            self._make_scored("C", 0.75),
        ]
        alerts = filter_alerts(scored, min_score=0.60, max_alerts=10)
        scores = [a["score"] for a in alerts]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input(self) -> None:
        assert filter_alerts([], min_score=0.60, max_alerts=5) == []


# ─── score_candidates (heuristic fallback) ─────────────────────────────────────
class TestScoreCandidates:
    def test_falls_back_to_heuristic_when_no_api_key(
        self, strong_call_ticker: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Patch _ollama_available to False so no API key → heuristic (not Ollama)
        import orchestrate as orch
        monkeypatch.setattr(orch, "_ollama_available", lambda: False)
        config = {"llm": {"api_key": "PLACEHOLDER"}, "scan": {}}
        results = score_candidates(
            [strong_call_ticker], frameworks=[], config=config, context=None
        )
        assert len(results) == 1
        assert results[0]["scoring_method"] == "heuristic"

    def test_force_heuristic_flag(self, strong_call_ticker: dict) -> None:
        config = {"llm": {"api_key": "some_key"}, "scan": {}}
        results = score_candidates(
            [strong_call_ticker], frameworks=[], config=config,
            context=None, force_heuristic=True,
        )
        assert results[0]["scoring_method"] == "heuristic"

    def test_returns_one_result_per_ticker(
        self, strong_call_ticker: dict, strong_put_ticker: dict
    ) -> None:
        config = {"llm": {"api_key": "PLACEHOLDER"}, "scan": {}}
        results = score_candidates(
            [strong_call_ticker, strong_put_ticker],
            frameworks=[], config=config, context=None,
        )
        assert len(results) == 2

    @patch("orchestrate.llm_score")
    def test_uses_llm_results_when_available(
        self, mock_llm: MagicMock, strong_call_ticker: dict
    ) -> None:
        mock_llm.return_value = [{
            "symbol": "NVDA", "score": 0.92, "direction": "call",
            "rationale": "LLM says buy", "supporting_creators": ["kpak82"],
            "key_signals": ["RSI momentum"], "suggested_dte": "7 days",
            "risk_level": "medium", "skip_reason": None,
        }]
        config = {"llm": {"api_key": "real_key"}, "scan": {}}
        results = score_candidates(
            [strong_call_ticker], frameworks=[], config=config, context=None
        )
        assert results[0]["scoring_method"] == "llm"
        assert results[0]["score"] == 0.92

    @patch("orchestrate.llm_score")
    def test_heuristic_fallback_for_missing_llm_symbol(
        self, mock_llm: MagicMock, strong_call_ticker: dict, strong_put_ticker: dict
    ) -> None:
        # LLM only returns eval for NVDA, not SPY → SPY should use heuristic
        mock_llm.return_value = [{
            "symbol": "NVDA", "score": 0.85, "direction": "call",
            "rationale": "ok", "supporting_creators": [],
            "key_signals": [], "suggested_dte": "7 days",
            "risk_level": "medium", "skip_reason": None,
        }]
        config = {"llm": {"api_key": "real_key"}, "scan": {}}
        results = score_candidates(
            [strong_call_ticker, strong_put_ticker],
            frameworks=[], config=config, context=None
        )
        assert len(results) == 2
        nvda = next(r for r in results if r["symbol"] == "NVDA")
        spy  = next(r for r in results if r["symbol"] == "SPY")
        assert nvda["scoring_method"] == "llm"
        assert spy["scoring_method"]  == "heuristic"


# ─── load_creator_frameworks ───────────────────────────────────────────────────
class TestLoadCreatorFrameworks:
    def test_loads_active_frameworks(self, tmp_path: Path) -> None:
        creator_dir = tmp_path / "creators" / "kpak82"
        creator_dir.mkdir(parents=True)
        (creator_dir / "creator_meta.json").write_text(json.dumps({
            "handle": "kpak82",
            "scanner_relevance": "high",
            "display_name": "kpak",
            "asset_focus": "US equities",
        }))
        (creator_dir / "framework-v1.md").write_text("# Framework\n## Setup Triggers\nTest")

        with patch("orchestrate.CREATORS_DIR", tmp_path / "creators"):
            result = load_creator_frameworks()

        assert len(result) == 1
        assert result[0]["handle"] == "kpak82"
        assert result[0]["weight"] == 1.0

    def test_skips_disqualified_creators(self, tmp_path: Path) -> None:
        creator_dir = tmp_path / "creators" / "puppy_trades"
        creator_dir.mkdir(parents=True)
        (creator_dir / "creator_meta.json").write_text(json.dumps({
            "handle": "puppy_trades",
            "scanner_relevance": "disqualified",
        }))
        (creator_dir / "framework-v1.md").write_text("# Framework")

        with patch("orchestrate.CREATORS_DIR", tmp_path / "creators"):
            result = load_creator_frameworks()

        assert result == []

    def test_loads_latest_framework_version(self, tmp_path: Path) -> None:
        creator_dir = tmp_path / "creators" / "kpak82"
        creator_dir.mkdir(parents=True)
        (creator_dir / "creator_meta.json").write_text(json.dumps({
            "handle": "kpak82", "scanner_relevance": "high",
        }))
        (creator_dir / "framework-v1.md").write_text("Version 1")
        (creator_dir / "framework-v2.md").write_text("Version 2")
        (creator_dir / "framework-v3.md").write_text("Version 3 — latest")

        with patch("orchestrate.CREATORS_DIR", tmp_path / "creators"):
            result = load_creator_frameworks()

        assert "Version 3" in result[0]["framework_text"]

    def test_correct_weights_by_relevance(self, tmp_path: Path) -> None:
        for handle, relevance in [("a", "high"), ("b", "low"), ("c", "medium")]:
            d = tmp_path / "creators" / handle
            d.mkdir(parents=True)
            (d / "creator_meta.json").write_text(json.dumps({
                "handle": handle, "scanner_relevance": relevance,
            }))
            (d / "framework-v1.md").write_text("# F")

        with patch("orchestrate.CREATORS_DIR", tmp_path / "creators"):
            result = load_creator_frameworks()

        weights = {r["handle"]: r["weight"] for r in result}
        assert weights["a"] == 1.0
        assert weights["b"] == 0.3
        assert weights["c"] == 0.6
