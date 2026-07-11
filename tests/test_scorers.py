"""Tests for the pluggable scorer surface in orchestrate.py (P2-1).

Covers the Scorer registry, LLM/Heuristic scorers, primary selection, per-ticker
heuristic fallback, and the shadow (A/B) scorer. llm_score is monkeypatched so
no network is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TRADING_DIR = Path.home() / "trading"
if str(TRADING_DIR) not in sys.path:
    sys.path.insert(0, str(TRADING_DIR))

import orchestrate as o


def _ticker(symbol="NVDA", rsi=82.0):
    return {
        "symbol": symbol, "name": symbol, "price": 100.0, "change_pct": 1.0,
        "relative_volume": 2.0, "call_put_ratio": 1.5, "earnings_within_48h": False,
        "news": [], "patterns": {"rsi": rsi, "ema_alignment": "bearish",
                                 "macd": -0.2, "macd_signal": -0.1, "tv_recommendation": -0.4},
    }


def _fake_llm_eval(symbol, direction="call", score=0.75):
    return {"symbol": symbol, "score": score, "direction": direction,
            "rationale": "x", "supporting_creators": ["kpak82"], "key_signals": [],
            "risk_level": "medium", "skip_reason": None}


# ─── registry ───────────────────────────────────────────────────────────────────
class TestRegistry:
    def test_get_known_scorers(self):
        assert isinstance(o.get_scorer("llm"), o.LLMScorer)
        assert isinstance(o.get_scorer("heuristic"), o.HeuristicScorer)

    def test_unknown_falls_back_to_llm(self):
        assert isinstance(o.get_scorer("does-not-exist"), o.LLMScorer)


# ─── HeuristicScorer ────────────────────────────────────────────────────────────
class TestHeuristicScorer:
    def test_scores_every_ticker(self):
        evals = o.HeuristicScorer().score([_ticker("NVDA"), _ticker("AMD")], [], {}, None)
        assert {e["symbol"] for e in evals} == {"NVDA", "AMD"}
        assert all(e["scoring_method"] == "heuristic" for e in evals)


# ─── LLMScorer ──────────────────────────────────────────────────────────────────
class TestLLMScorer:
    def test_tags_method_and_conviction(self, monkeypatch):
        monkeypatch.setattr(o, "llm_score",
                            lambda t, f, c, ctx, session_note="": [_fake_llm_eval("NVDA")])
        evals = o.LLMScorer().score([_ticker("NVDA")], [], {"llm": {}}, None)
        assert evals[0]["scoring_method"] == "llm"
        assert evals[0]["conviction"] == "medium"   # normalised from risk_level

    def test_batches_large_sets(self, monkeypatch):
        calls = {"n": 0}
        def fake(tks, f, c, ctx, session_note=""):
            calls["n"] += 1
            return [_fake_llm_eval(t["symbol"]) for t in tks]
        monkeypatch.setattr(o, "llm_score", fake)
        tickers = [_ticker(f"S{i}") for i in range(9)]        # >8 default batch
        evals = o.LLMScorer().score(tickers, [], {"llm": {"batch_size": 4}}, None)
        assert calls["n"] == 3                                 # 4+4+1
        assert len(evals) == 9


# ─── score_candidates ───────────────────────────────────────────────────────────
class TestScoreCandidates:
    def test_force_heuristic_uses_heuristic(self, monkeypatch):
        monkeypatch.setattr(o, "llm_score",
                            lambda *a, **k: pytest.fail("LLM must not be called"))
        scored = o.score_candidates([_ticker("NVDA")], [], {}, None, force_heuristic=True)
        assert scored[0]["scoring_method"] == "heuristic"

    def test_primary_from_config(self, monkeypatch):
        monkeypatch.setattr(o, "llm_score",
                            lambda *a, **k: pytest.fail("LLM must not be called"))
        scored = o.score_candidates([_ticker("NVDA")], [], {"scoring": {"primary": "heuristic"}}, None)
        assert scored[0]["scoring_method"] == "heuristic"

    def test_llm_primary_with_heuristic_fallback(self, monkeypatch):
        # LLM only returns NVDA; AMD must fall back to heuristic
        monkeypatch.setattr(o, "llm_score",
                            lambda t, f, c, ctx, session_note="": [_fake_llm_eval("NVDA")])
        scored = o.score_candidates([_ticker("NVDA"), _ticker("AMD")], [], {"llm": {}}, None)
        by_sym = {s["symbol"]: s for s in scored}
        assert by_sym["NVDA"]["scoring_method"] == "llm"
        assert by_sym["AMD"]["scoring_method"] == "heuristic"


# ─── shadow_score ───────────────────────────────────────────────────────────────
class TestShadowScore:
    def test_disabled_returns_empty(self):
        assert o.shadow_score([_ticker("NVDA")], [], {}, None) == []
        assert o.shadow_score([_ticker("NVDA")], [], {"scoring": {"shadow": None}}, None) == []

    def test_runs_configured_shadow_scorer(self, monkeypatch):
        monkeypatch.setattr(o, "llm_score",
                            lambda *a, **k: pytest.fail("heuristic shadow must not call LLM"))
        out = o.shadow_score([_ticker("NVDA"), _ticker("AMD")], [],
                             {"scoring": {"shadow": "heuristic"}}, None)
        assert {e["symbol"] for e in out} == {"NVDA", "AMD"}
        assert all(e["scoring_method"] == "heuristic" for e in out)

    def test_unknown_shadow_is_noop(self):
        assert o.shadow_score([_ticker("NVDA")], [], {"scoring": {"shadow": "nope"}}, None) == []
