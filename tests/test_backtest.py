"""
tests/test_backtest.py — Unit tests for backtest.py (Week 5)

Tests focus on pure math functions (indicators, strategy logic) — no network
calls. yfinance-dependent integration tests are skipped unless --run-live.

Run: cd ~/trading && python -m pytest tests/test_backtest.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest import (
    _ema,
    _rsi,
    _atr,
    _ema_cross_signals,
    _rsi_oversold_signals,
    _supertrend_signals,
    _verdict,
    _overall_verdict,
    run_strategy,
    symbols_from_alerts,
    symbols_from_journal,
)


# ─── EMA ──────────────────────────────────────────────────────────────────────
class TestEma:
    def test_returns_none_for_insufficient_data(self) -> None:
        prices = [1.0, 2.0, 3.0]
        result = _ema(prices, period=5)
        assert all(v is None for v in result)

    def test_first_value_is_simple_average(self) -> None:
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _ema(prices, period=5)
        assert result[4] == pytest.approx(3.0)

    def test_ema_reacts_to_price_increase(self) -> None:
        prices = [10.0] * 10 + [20.0] * 10
        result = _ema(prices, period=5)
        # After 10 periods of 20, EMA should be significantly above 10
        assert result[-1] > 15.0

    def test_length_preserved(self) -> None:
        prices = list(range(1, 21))
        result = _ema(prices, period=5)
        assert len(result) == 20

    def test_none_prefix_before_period(self) -> None:
        prices = list(range(1, 11))
        result = _ema(prices, period=5)
        assert all(v is None for v in result[:4])
        assert result[4] is not None


# ─── RSI ──────────────────────────────────────────────────────────────────────
class TestRsi:
    def test_returns_none_for_insufficient_data(self) -> None:
        prices = [1.0, 2.0, 3.0]
        result = _rsi(prices, period=14)
        assert all(v is None for v in result)

    def test_rsi_between_0_and_100(self) -> None:
        import random
        random.seed(42)
        prices = [100.0 + random.gauss(0, 2) for _ in range(50)]
        result = _rsi(prices, period=14)
        for v in result:
            if v is not None:
                assert 0.0 <= v <= 100.0

    def test_all_gains_gives_100(self) -> None:
        prices = list(range(1, 30))   # strictly increasing
        result = _rsi(prices, period=14)
        # After sufficient gains, RSI → 100
        assert result[-1] == pytest.approx(100.0)

    def test_all_losses_gives_0(self) -> None:
        prices = list(range(30, 0, -1))   # strictly decreasing
        result = _rsi(prices, period=14)
        assert result[-1] == pytest.approx(0.0, abs=1e-6)


# ─── ATR ──────────────────────────────────────────────────────────────────────
class TestAtr:
    def test_returns_none_before_period(self) -> None:
        n = 20
        closes = [100.0] * n
        highs  = [101.0] * n
        lows   = [99.0]  * n
        result = _atr(highs, lows, closes, period=10)
        assert all(v is None for v in result[:10])

    def test_atr_is_positive(self) -> None:
        n = 30
        closes = [100.0 + i * 0.1 for i in range(n)]
        highs  = [c + 1.0 for c in closes]
        lows   = [c - 1.0 for c in closes]
        result = _atr(highs, lows, closes, period=10)
        for v in result:
            if v is not None:
                assert v > 0


# ─── EMA Cross Signals ────────────────────────────────────────────────────────
class TestEmaCrossSignals:
    def test_golden_cross_detected(self) -> None:
        # Create a clear golden cross: prices rise sharply after 50 bars
        prices = [50.0] * 60 + [100.0] * 10
        signals = _ema_cross_signals(prices)
        # At least one crossover signal should fire after the price jump
        assert any(signals[50:])

    def test_no_cross_in_flat_series(self) -> None:
        prices = [100.0] * 100
        signals = _ema_cross_signals(prices)
        assert not any(signals)

    def test_output_length_matches_input(self) -> None:
        prices = [float(i) for i in range(1, 101)]
        signals = _ema_cross_signals(prices)
        assert len(signals) == 100


# ─── RSI Oversold Signals ─────────────────────────────────────────────────────
class TestRsiOversoldSignals:
    def test_no_signals_in_flat_series(self) -> None:
        prices = [100.0] * 50
        signals = _rsi_oversold_signals(prices)
        assert not any(signals)

    def test_oversold_bounce_detected(self) -> None:
        # Sharp drop then recovery
        prices = [100.0] * 15 + [60.0] * 5 + [80.0] * 10 + [95.0] * 10
        signals = _rsi_oversold_signals(prices)
        assert any(signals)

    def test_output_length_matches_input(self) -> None:
        prices = [float(i % 10 + 90) for i in range(80)]
        signals = _rsi_oversold_signals(prices)
        assert len(signals) == 80


# ─── run_strategy ─────────────────────────────────────────────────────────────
class TestRunStrategy:
    def _flat_data(self, n: int = 200):
        dates  = [f"2025-{(i//30+1):02d}-{(i%30+1):02d}" for i in range(n)]
        closes = [100.0] * n
        highs  = [101.0] * n
        lows   = [99.0]  * n
        return dates, closes, highs, lows

    def _trending_data(self, n: int = 200):
        dates  = [f"2025-{(i//30+1):02d}-{(i%30+1):02d}" for i in range(n)]
        closes = [50.0 + i * 0.5 for i in range(n)]
        highs  = [c + 1.0 for c in closes]
        lows   = [c - 1.0 for c in closes]
        return dates, closes, highs, lows

    def test_flat_series_produces_no_signals(self) -> None:
        dates, closes, highs, lows = self._flat_data()
        result = run_strategy(dates, closes, highs, lows, "ema_cross")
        assert result.get("signal_count", 0) == 0 or result["verdict"] == "NO_SIGNALS"

    def test_result_has_required_keys(self) -> None:
        dates, closes, highs, lows = self._trending_data()
        result = run_strategy(dates, closes, highs, lows, "ema_cross")
        assert "strategy" in result
        assert "signal_count" in result

    def test_invalid_strategy_raises(self) -> None:
        dates, closes, highs, lows = self._flat_data()
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_strategy(dates, closes, highs, lows, "invalid_strategy")

    def test_win_rate_between_0_and_1(self) -> None:
        dates, closes, highs, lows = self._trending_data()
        result = run_strategy(dates, closes, highs, lows, "rsi_oversold")
        if result.get("signal_count", 0) > 0:
            assert 0.0 <= result["win_rate"] <= 1.0

    def test_no_overlapping_trades(self) -> None:
        dates, closes, highs, lows = self._trending_data()
        result = run_strategy(dates, closes, highs, lows, "ema_cross", hold_days=10)
        trades = result.get("trades", [])
        for i in range(1, len(trades)):
            prev_exit = trades[i - 1]["exit_date"]
            curr_entry = trades[i]["entry_date"]
            assert curr_entry >= prev_exit


# ─── _verdict ─────────────────────────────────────────────────────────────────
class TestVerdict:
    def test_robust_when_all_strong(self) -> None:
        assert _verdict(0.65, 3.0, 1.2, n=10) == "ROBUST"

    def test_moderate_when_marginal(self) -> None:
        assert _verdict(0.55, 1.0, 0.5, n=10) == "MODERATE"

    def test_weak_when_below_thresholds(self) -> None:
        assert _verdict(0.40, -0.5, 0.2, n=10) == "WEAK"

    def test_insufficient_data_when_few_signals(self) -> None:
        from backtest import MIN_SIGNALS
        assert _verdict(0.70, 5.0, 2.0, n=MIN_SIGNALS - 1) == "INSUFFICIENT_DATA"


# ─── _overall_verdict ─────────────────────────────────────────────────────────
class TestOverallVerdict:
    def test_robust_if_any_robust(self) -> None:
        strategies = {
            "ema_cross":   {"verdict": "ROBUST"},
            "rsi_oversold": {"verdict": "WEAK"},
        }
        assert _overall_verdict(strategies) == "ROBUST"

    def test_moderate_if_no_robust_but_moderate(self) -> None:
        strategies = {
            "ema_cross":   {"verdict": "MODERATE"},
            "rsi_oversold": {"verdict": "WEAK"},
        }
        assert _overall_verdict(strategies) == "MODERATE"

    def test_weak_when_all_weak(self) -> None:
        strategies = {
            "ema_cross":   {"verdict": "WEAK"},
            "supertrend":  {"verdict": "WEAK"},
        }
        assert _overall_verdict(strategies) == "WEAK"

    def test_empty_dict_returns_unknown(self) -> None:
        assert _overall_verdict({}) == "UNKNOWN"


# ─── symbols_from_alerts / symbols_from_journal ───────────────────────────────
class TestSymbolLoaders:
    def test_symbols_from_alerts_returns_list(self, tmp_path) -> None:
        alerts_data = {
            "alerts": [
                {"symbol": "NVDA", "direction": "call"},
                {"symbol": "AAPL", "direction": "put"},
            ]
        }
        path = tmp_path / "alerts.json"
        path.write_text(json.dumps(alerts_data))
        with patch("backtest.ALERTS_PATH", path):
            result = symbols_from_alerts()
        assert result == ["NVDA", "AAPL"]

    def test_symbols_from_alerts_empty_when_missing(self, tmp_path) -> None:
        with patch("backtest.ALERTS_PATH", tmp_path / "nonexistent.json"):
            result = symbols_from_alerts()
        assert result == []

    def test_symbols_from_journal_deduplicated(self, tmp_path) -> None:
        path = tmp_path / "trade_journal.jsonl"
        entries = [
            {"symbol": "NVDA", "direction": "call"},
            {"symbol": "AAPL", "direction": "put"},
            {"symbol": "NVDA", "direction": "call"},   # duplicate
        ]
        path.write_text("\n".join(json.dumps(e) for e in entries))
        with patch("backtest.JOURNAL_PATH", path):
            result = symbols_from_journal()
        assert result.count("NVDA") == 1
        assert "AAPL" in result

    def test_symbols_from_journal_empty_when_missing(self, tmp_path) -> None:
        with patch("backtest.JOURNAL_PATH", tmp_path / "nonexistent.jsonl"):
            result = symbols_from_journal()
        assert result == []
