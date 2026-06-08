"""
test_scanner.py — Week 1 pytest suite for scanner.py (tradingview-screener version)

Covers:
  - get_volume_leaders: success, screener error, empty result
  - _ema_alignment: all branches
  - check_earnings: various scenarios, yfinance exceptions
  - _to_utc: datetime coercions
  - get_options_data: success, failure, call/put ratio
  - get_news: yfinance v1.4 nested format, failure, limit
  - _write_output: context flag, data_quality, required keys
  - load_config: missing file exits, valid config loads
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

# Make scanner importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
import scanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "scan": {
        "min_relative_volume": 2.0,
        "min_price": 10.0,
        "min_total_volume": 2_000_000,
        "earnings_buffer_hours": 48,
        "pre_filter_top_n": 10,
    }
}

def _screener_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame as tradingview-screener would return."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# get_volume_leaders
# ---------------------------------------------------------------------------
class TestGetVolumeLeaders:

    def _mock_screener(self, rows: list[dict]):
        """Patch Query to return a fake screener result."""
        df = _screener_df(rows)
        mock_chain = MagicMock()
        mock_chain.get_scanner_data.return_value = (len(rows), df)
        mock_chain.set_markets.return_value = mock_chain
        mock_chain.select.return_value = mock_chain
        mock_chain.where.return_value = mock_chain
        mock_chain.order_by.return_value = mock_chain
        mock_chain.limit.return_value = mock_chain
        return mock_chain

    def test_success_returns_candidates(self):
        rows = [
            {"ticker": "NASDAQ:AAPL", "name": "Apple", "close": 185.0,
             "volume": 5_000_000, "average_volume_10d_calc": 1_000_000,
             "relative_volume_10d_calc": 5.0, "change": 2.5,
             "RSI": 62.0, "MACD.macd": 0.8, "MACD.signal": 0.5,
             "EMA20": 180.0, "EMA50": 175.0, "EMA200": 160.0, "Recommend.All": 0.4},
        ]
        with patch("scanner.Query") as MockQuery:
            MockQuery.return_value = self._mock_screener(rows)
            result, had_errors = scanner.get_volume_leaders(DEFAULT_CONFIG)

        assert not had_errors
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["relative_volume"] == 5.0
        assert result[0]["rsi"] == 62.0

    def test_screener_exception_returns_empty_with_error(self):
        with patch("scanner.Query", side_effect=Exception("network timeout")):
            result, had_errors = scanner.get_volume_leaders(DEFAULT_CONFIG)
        assert result == []
        assert had_errors is True

    def test_empty_dataframe_returns_empty_no_error(self):
        with patch("scanner.Query") as MockQuery:
            mock_chain = MagicMock()
            mock_chain.get_scanner_data.return_value = (0, pd.DataFrame())
            mock_chain.set_markets.return_value = mock_chain
            mock_chain.select.return_value = mock_chain
            mock_chain.where.return_value = mock_chain
            mock_chain.order_by.return_value = mock_chain
            mock_chain.limit.return_value = mock_chain
            MockQuery.return_value = mock_chain
            result, had_errors = scanner.get_volume_leaders(DEFAULT_CONFIG)
        assert result == []
        assert had_errors is False

    def test_none_ta_fields_handled_gracefully(self):
        rows = [
            {"ticker": "NYSE:GS", "name": "Goldman", "close": 450.0,
             "volume": 3_000_000, "average_volume_10d_calc": 500_000,
             "relative_volume_10d_calc": 3.0, "change": 1.0,
             "RSI": None, "MACD.macd": None, "MACD.signal": None,
             "EMA20": None, "EMA50": None, "EMA200": None, "Recommend.All": None},
        ]
        with patch("scanner.Query") as MockQuery:
            MockQuery.return_value = self._mock_screener(rows)
            result, _ = scanner.get_volume_leaders(DEFAULT_CONFIG)
        assert result[0]["rsi"] is None
        assert result[0]["tv_recommendation"] is None

    def test_ticker_parsed_correctly(self):
        rows = [
            {"ticker": "NASDAQ:NVDA", "name": "NVIDIA", "close": 120.0,
             "volume": 10_000_000, "average_volume_10d_calc": 2_000_000,
             "relative_volume_10d_calc": 4.5, "change": 3.0,
             "RSI": 55.0, "MACD.macd": 0.5, "MACD.signal": 0.3,
             "EMA20": 115.0, "EMA50": 110.0, "EMA200": 100.0, "Recommend.All": 0.2},
        ]
        with patch("scanner.Query") as MockQuery:
            MockQuery.return_value = self._mock_screener(rows)
            result, _ = scanner.get_volume_leaders(DEFAULT_CONFIG)
        assert result[0]["symbol"] == "NVDA"
        assert result[0]["tv_ticker"] == "NASDAQ:NVDA"


# ---------------------------------------------------------------------------
# _ema_alignment
# ---------------------------------------------------------------------------
class TestEmaAlignment:
    def test_bullish_all_three(self):
        assert scanner._ema_alignment(200, 180, 160) == "bullish"

    def test_bearish_all_three(self):
        assert scanner._ema_alignment(150, 170, 190) == "bearish"

    def test_mixed(self):
        assert scanner._ema_alignment(175, 160, 180) == "mixed"

    def test_bullish_two_emas_only(self):
        assert scanner._ema_alignment(200, 180, None) == "bullish"

    def test_bearish_two_emas_only(self):
        assert scanner._ema_alignment(150, 170, None) == "bearish"

    def test_neutral_equal_emas(self):
        assert scanner._ema_alignment(180, 180, None) == "neutral"

    def test_missing_ema20_returns_none(self):
        assert scanner._ema_alignment(None, 180, 160) is None

    def test_missing_both_emas_returns_none(self):
        assert scanner._ema_alignment(None, None, 160) is None


# ---------------------------------------------------------------------------
# check_earnings
# ---------------------------------------------------------------------------
class TestCheckEarnings:
    def _mock_ticker(self, cal_value):
        t = MagicMock()
        t.calendar = cal_value
        return t

    def test_no_calendar_returns_false(self):
        with patch("scanner.yf.Ticker") as cls:
            cls.return_value = self._mock_ticker(None)
            assert scanner.check_earnings(["AAPL"]) == {"AAPL": False}

    def test_empty_calendar_returns_false(self):
        with patch("scanner.yf.Ticker") as cls:
            cls.return_value = self._mock_ticker(pd.DataFrame())
            assert scanner.check_earnings(["AAPL"]) == {"AAPL": False}

    def test_earnings_within_48h_returns_true(self):
        tomorrow = datetime.now(timezone.utc) + timedelta(hours=20)
        cal = pd.DataFrame({"Earnings Date": [tomorrow]})
        with patch("scanner.yf.Ticker") as cls:
            cls.return_value = self._mock_ticker(cal)
            assert scanner.check_earnings(["AAPL"]) == {"AAPL": True}

    def test_earnings_after_48h_returns_false(self):
        far_future = datetime.now(timezone.utc) + timedelta(days=30)
        cal = pd.DataFrame({"Earnings Date": [far_future]})
        with patch("scanner.yf.Ticker") as cls:
            cls.return_value = self._mock_ticker(cal)
            assert scanner.check_earnings(["AAPL"]) == {"AAPL": False}

    def test_yfinance_exception_defaults_false(self):
        with patch("scanner.yf.Ticker", side_effect=Exception("network error")):
            assert scanner.check_earnings(["AAPL"]) == {"AAPL": False}

    def test_multiple_tickers_independent(self):
        earnings_soon  = datetime.now(timezone.utc) + timedelta(hours=12)
        earnings_later = datetime.now(timezone.utc) + timedelta(days=20)

        def side_effect(symbol):
            t = MagicMock()
            t.calendar = pd.DataFrame({
                "Earnings Date": [earnings_soon if symbol == "AAPL" else earnings_later]
            })
            return t

        with patch("scanner.yf.Ticker", side_effect=side_effect):
            result = scanner.check_earnings(["AAPL", "MSFT"])
            assert result["AAPL"] is True
            assert result["MSFT"] is False


# ---------------------------------------------------------------------------
# _to_utc
# ---------------------------------------------------------------------------
class TestToUtc:
    def test_naive_datetime(self):
        dt = datetime(2026, 6, 10, 12, 0, 0)
        result = scanner._to_utc(dt)
        assert result.tzinfo is not None

    def test_aware_datetime_unchanged(self):
        dt = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
        result = scanner._to_utc(dt)
        assert result == dt

    def test_date_object(self):
        d = date(2026, 6, 10)
        result = scanner._to_utc(d)
        assert isinstance(result, datetime)
        assert result.tzinfo is not None


# ---------------------------------------------------------------------------
# get_options_data
# ---------------------------------------------------------------------------
class TestGetOptionsData:
    def test_yfinance_failure_returns_empty_structure(self):
        with patch("scanner.yf.Ticker", side_effect=Exception("network")):
            result = scanner.get_options_data(["AAPL"])
        assert result["AAPL"]["calls"] == []
        assert result["AAPL"]["puts"] == []
        assert result["AAPL"]["total_volume"] == 0

    def test_no_expirations_returns_empty(self):
        mock_ticker = MagicMock()
        mock_ticker.options = []
        with patch("scanner.yf.Ticker", return_value=mock_ticker):
            result = scanner.get_options_data(["AAPL"])
        assert result["AAPL"]["calls"] == []

    def test_call_put_ratio_calculated(self):
        mock_ticker = MagicMock()
        mock_ticker.options = ["2026-06-20"]
        calls_df = pd.DataFrame({
            "strike": [185.0], "volume": [300.0],
            "openInterest": [500.0], "bid": [2.0],
            "ask": [2.5], "impliedVolatility": [0.35],
        })
        puts_df = pd.DataFrame({
            "strike": [185.0], "volume": [100.0],
            "openInterest": [200.0], "bid": [1.5],
            "ask": [2.0], "impliedVolatility": [0.35],
        })
        chain = MagicMock()
        chain.calls = calls_df
        chain.puts  = puts_df
        mock_ticker.option_chain.return_value = chain
        with patch("scanner.yf.Ticker", return_value=mock_ticker):
            result = scanner.get_options_data(["AAPL"])
        assert result["AAPL"]["call_put_ratio"] == 3.0
        assert result["AAPL"]["total_call_volume"] == 300
        assert result["AAPL"]["total_put_volume"] == 100


# ---------------------------------------------------------------------------
# get_news (yfinance v1.4 nested format)
# ---------------------------------------------------------------------------
class TestGetNews:
    def test_yfinance_failure_returns_empty_list(self):
        with patch("scanner.yf.Ticker", side_effect=Exception("timeout")):
            result = scanner.get_news(["AAPL"])
        assert result["AAPL"] == []

    def test_nested_v14_format_parsed(self):
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {
                "id": "abc",
                "content": {
                    "title": "AAPL hits record high",
                    "provider": {"displayName": "Bloomberg"},
                    "pubDate": "2026-06-07T10:00:00Z",
                    "summary": "Apple shares surged.",
                }
            }
        ]
        with patch("scanner.yf.Ticker", return_value=mock_ticker):
            result = scanner.get_news(["AAPL"])
        assert len(result["AAPL"]) == 1
        item = result["AAPL"][0]
        assert item["title"] == "AAPL hits record high"
        assert item["publisher"] == "Bloomberg"
        assert item["published_at"] == "2026-06-07T10:00:00Z"

    def test_flat_legacy_format_parsed(self):
        """Older yfinance format (flat dict, not nested)."""
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {
                "title": "Old format headline",
                "publisher": "Reuters",
                "providerPublishTime": 1748000000,
            }
        ]
        with patch("scanner.yf.Ticker", return_value=mock_ticker):
            result = scanner.get_news(["MSFT"])
        item = result["MSFT"][0]
        assert item["title"] == "Old format headline"
        assert item["publisher"] == "Reuters"

    def test_news_limited_to_5_items(self):
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {"content": {"title": f"H{i}", "provider": {"displayName": "X"},
                         "pubDate": "2026-06-07", "summary": ""}}
            for i in range(10)
        ]
        with patch("scanner.yf.Ticker", return_value=mock_ticker):
            result = scanner.get_news(["AAPL"])
        assert len(result["AAPL"]) == 5

    def test_news_has_required_fields(self):
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {"content": {
                "title": "Big move", "provider": {"displayName": "CNBC"},
                "pubDate": "2026-06-07T12:00:00Z", "summary": "Stocks surge.",
            }}
        ]
        with patch("scanner.yf.Ticker", return_value=mock_ticker):
            result = scanner.get_news(["AAPL"])
        item = result["AAPL"][0]
        assert "title" in item
        assert "publisher" in item
        assert "published_at" in item
        assert "summary" in item


# ---------------------------------------------------------------------------
# _write_output: context flag, data_quality, required keys
# ---------------------------------------------------------------------------
class TestOutputStructure:
    def test_context_present_in_output(self, tmp_path):
        with patch("scanner.DATA_DIR", tmp_path):
            scanner._write_output([], context="bearish bias", had_errors=False)
            output = json.loads((tmp_path / "all_data.json").read_text())
        assert output["context"] == "bearish bias"

    def test_context_none_when_not_provided(self, tmp_path):
        with patch("scanner.DATA_DIR", tmp_path):
            scanner._write_output([], context=None, had_errors=False)
            output = json.loads((tmp_path / "all_data.json").read_text())
        assert output["context"] is None

    def test_data_quality_complete_when_no_errors(self, tmp_path):
        with patch("scanner.DATA_DIR", tmp_path):
            scanner._write_output([], context=None, had_errors=False)
            output = json.loads((tmp_path / "all_data.json").read_text())
        assert output["data_quality"] == "complete"

    def test_data_quality_partial_when_errors(self, tmp_path):
        with patch("scanner.DATA_DIR", tmp_path):
            scanner._write_output([], context=None, had_errors=True)
            output = json.loads((tmp_path / "all_data.json").read_text())
        assert output["data_quality"] == "partial"

    def test_output_has_required_top_level_keys(self, tmp_path):
        with patch("scanner.DATA_DIR", tmp_path):
            scanner._write_output([], context=None, had_errors=False)
            output = json.loads((tmp_path / "all_data.json").read_text())
        for key in ("scan_timestamp", "data_quality", "context", "tickers"):
            assert key in output, f"Missing key: {key}"

    def test_ticker_record_has_required_fields(self, tmp_path):
        record = {
            "symbol": "AAPL", "name": "Apple", "price": 185.40,
            "change_pct": 1.2, "relative_volume": 3.2, "volume": 5_000_000,
            "avg_volume_10d": 1_000_000, "earnings_within_48h": False,
            "patterns": {
                "rsi": 62.0, "macd": 0.8, "macd_signal": 0.5,
                "ema20": 180.0, "ema50": 175.0, "ema200": 160.0,
                "ema_alignment": "bullish", "tv_recommendation": 0.4,
            },
            "options_total_volume": 50_000, "options_call_volume": 30_000,
            "options_put_volume": 20_000, "call_put_ratio": 1.5,
            "options_chain": {"calls": [], "puts": []},
            "ohlcv": [], "news": [],
        }
        with patch("scanner.DATA_DIR", tmp_path):
            scanner._write_output([record], context=None, had_errors=False)
            output = json.loads((tmp_path / "all_data.json").read_text())
        t = output["tickers"][0]
        for field in ("symbol", "price", "relative_volume", "earnings_within_48h",
                      "patterns", "options_chain", "news", "ohlcv"):
            assert field in t, f"Missing field in ticker record: {field}"
        assert "rsi" in t["patterns"]
        assert "ema_alignment" in t["patterns"]
        assert "tv_recommendation" in t["patterns"]


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
class TestConfigLoading:
    def test_missing_config_exits_with_code_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scanner.CONFIG_PATH", tmp_path / "nonexistent.json")
        with pytest.raises(SystemExit) as exc_info:
            scanner.load_config()
        assert exc_info.value.code == 1

    def test_valid_config_loaded(self, tmp_path, monkeypatch):
        cfg = {"scan": {"min_relative_volume": 2.0}}
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        monkeypatch.setattr("scanner.CONFIG_PATH", cfg_path)
        result = scanner.load_config()
        assert result == cfg
