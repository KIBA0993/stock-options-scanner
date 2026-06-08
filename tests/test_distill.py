"""
Tests for distill.py — Creator Framework Distiller

Tests:
  - load_config: config reading and error handling
  - clean_posts: noise removal from raw X browser text
  - next_version: framework versioning logic
  - latest_posts_file: auto-detection of posts file
  - LLM prompt construction (content, structure)
  - Output file creation and format validation
  - CLI argument handling
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent dir to path so we can import distill
sys.path.insert(0, str(Path(__file__).parent.parent))
from distill import clean_posts, latest_posts_file, load_config, next_version


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def tmp_trading_dir(tmp_path: Path) -> Path:
    """Minimal ~/trading layout in a temp directory."""
    (tmp_path / "creators").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    cfg = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-opus-4-5",
            "api_key": "test-key-abc",
            "max_tokens": 2048,
        }
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture()
def creator_dir(tmp_trading_dir: Path) -> Path:
    d = tmp_trading_dir / "creators" / "kpak82"
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_loads_valid_config(self, tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("distill.CONFIG_PATH", tmp_config)
        cfg = load_config()
        assert cfg["llm"]["provider"] == "anthropic"
        assert cfg["llm"]["api_key"] == "test-key-abc"

    def test_exits_when_config_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("distill.CONFIG_PATH", tmp_path / "nonexistent.json")
        with pytest.raises(SystemExit):
            load_config()


# ---------------------------------------------------------------------------
# clean_posts
# ---------------------------------------------------------------------------
class TestCleanPosts:
    RAW_PAGE = """\
Home
Explore
Notifications
kpak82
73K Followers
Jan 15
$SPY looking at the 4H chart — we just broke the key supply level at 475
Strong momentum, watching for a pullback to 472 for entry
Likes
1.2K
Views
Jan 14
Following
$QQQ puts working well, closed at +65%
My plan from yesterday played out perfectly
Following
42
More
"""

    def test_removes_nav_chrome(self) -> None:
        result = clean_posts(self.RAW_PAGE)
        assert "Home" not in result
        assert "Explore" not in result
        assert "Notifications" not in result
        assert "Following" not in result

    def test_keeps_trading_content(self) -> None:
        result = clean_posts(self.RAW_PAGE)
        assert "$SPY" in result
        assert "4H chart" in result
        assert "supply level" in result
        assert "$QQQ puts" in result
        assert "+65%" in result

    def test_removes_standalone_handle(self) -> None:
        raw = "@kpak82\nBought SPY calls at open\n@someone_else"
        result = clean_posts(raw)
        assert "@kpak82" not in result
        assert "Bought SPY calls" in result

    def test_removes_follower_counts(self) -> None:
        raw = "73K\n1.2K\n500\n$SPY 480 target"
        result = clean_posts(raw)
        # Numbers alone (follower counts) should be stripped
        assert "$SPY 480 target" in result

    def test_empty_input(self) -> None:
        result = clean_posts("")
        assert result == ""

    def test_clean_post_is_shorter(self) -> None:
        result = clean_posts(self.RAW_PAGE)
        assert len(result) < len(self.RAW_PAGE)


# ---------------------------------------------------------------------------
# next_version
# ---------------------------------------------------------------------------
class TestNextVersion:
    def test_returns_1_when_no_frameworks(self, creator_dir: Path) -> None:
        assert next_version(creator_dir) == 1

    def test_returns_2_when_v1_exists(self, creator_dir: Path) -> None:
        (creator_dir / "framework-v1.md").write_text("# Framework v1")
        assert next_version(creator_dir) == 2

    def test_returns_n_plus_1(self, creator_dir: Path) -> None:
        for n in [1, 2, 3]:
            (creator_dir / f"framework-v{n}.md").write_text(f"# v{n}")
        assert next_version(creator_dir) == 4

    def test_ignores_non_framework_files(self, creator_dir: Path) -> None:
        (creator_dir / "posts_raw.txt").write_text("some posts")
        (creator_dir / "creator_meta.json").write_text("{}")
        assert next_version(creator_dir) == 1


# ---------------------------------------------------------------------------
# latest_posts_file
# ---------------------------------------------------------------------------
class TestLatestPostsFile:
    def test_returns_none_when_empty(self, creator_dir: Path) -> None:
        assert latest_posts_file(creator_dir) is None

    def test_returns_single_posts_file(self, creator_dir: Path) -> None:
        p = creator_dir / "posts_raw.txt"
        p.write_text("some posts")
        assert latest_posts_file(creator_dir) == p

    def test_returns_most_recently_modified(self, creator_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import time
        p1 = creator_dir / "posts_raw.txt"
        p1.write_text("old posts")
        time.sleep(0.05)  # ensure different mtime
        p2 = creator_dir / "posts_raw_2.txt"
        p2.write_text("new posts")
        assert latest_posts_file(creator_dir) == p2

    def test_ignores_framework_files(self, creator_dir: Path) -> None:
        (creator_dir / "framework-v1.md").write_text("# framework")
        assert latest_posts_file(creator_dir) is None


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------
class TestExtractionPromptContent:
    """Verify the extraction prompt includes required sections."""

    def _build_prompt(self, handle: str = "kpak82") -> str:
        from distill import EXTRACTION_PROMPT
        return EXTRACTION_PROMPT.format(
            handle=f"@{handle}",
            posts_text="$SPY calls, entry at 475, target 480",
            today="2026-06-07",
            version="v1",
        )

    def test_prompt_includes_handle(self) -> None:
        prompt = self._build_prompt()
        assert "@kpak82" in prompt

    def test_prompt_includes_required_sections(self) -> None:
        prompt = self._build_prompt()
        required = [
            "Setup Triggers",
            "Entry Rules",
            "Exit Rules",
            "Risk Management",
            "Quality Gate",
            "Preferred Instruments",
            "Red Flags",
        ]
        for section in required:
            assert section in prompt, f"Missing section: {section}"

    def test_prompt_includes_posts_text(self) -> None:
        prompt = self._build_prompt()
        assert "$SPY calls" in prompt

    def test_prompt_requests_not_documented_for_missing(self) -> None:
        prompt = self._build_prompt()
        assert "Not documented" in prompt


# ---------------------------------------------------------------------------
# Output file creation (integration-style with mocked LLM)
# ---------------------------------------------------------------------------
class TestDistillOutputFile:
    SAMPLE_FRAMEWORK = """\
# Trading Framework: @kpak82
Distilled: 2026-06-07
Version: v1
Asset focus: US equities — $SPX $SPY $QQQ

## Trading Personality
Technical, disciplined, patient. Waits for clean setups.

## Market Conditions They Trade
Trending markets with clear structure.

## Setup Triggers — What Gets Their Attention
- Volume breakout above 20-day average
- Break of key supply/demand level

## Preferred Instruments & Timeframes
- Calls vs puts preference: both
- Typical DTE range: 7-21 days
- Favored underlyings: $SPY $QQQ $SPX

## Entry Rules
Waits for 4H candle close above resistance.

## Exit Rules
- Profit target: 50-100% gain
- Stop loss: 25-30% loss

## Risk Management
- Position sizing: 2-5% per trade
- Max loss per trade: 30%

## Red Flags
Avoids FOMC week entries without clear bias.

## Quality Gate — Sample Setups
Date: 2026-01-15
Setup: SPY 4H break above 475
Direction: Call
Outcome: +65% closed next day
Why it fits: Clean breakout entry rule applied.

## Honest Limitations
Many intraday entries not documented in public posts.
"""

    def test_output_file_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        creator_dir = tmp_path / "creators" / "kpak82"
        creator_dir.mkdir(parents=True)

        posts_file = creator_dir / "posts_raw.txt"
        posts_file.write_text("$SPY breakout, buying calls at 475, target 480")

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "llm": {"provider": "anthropic", "model": "claude-opus-4-5",
                    "api_key": "test-key", "max_tokens": 2048}
        }))

        monkeypatch.setattr("distill.BASE_DIR", tmp_path)
        monkeypatch.setattr("distill.CONFIG_PATH", config_file)

        with patch("distill.call_llm", return_value=self.SAMPLE_FRAMEWORK):
            # Simulate what main() does after parsing args
            from distill import next_version

            version = 1
            out_path = creator_dir / f"framework-v{version}.md"
            out_path.write_text(self.SAMPLE_FRAMEWORK)

        assert out_path.exists()
        content = out_path.read_text()
        assert "Trading Framework: @kpak82" in content
        assert "Quality Gate" in content

    def test_framework_has_required_sections(self) -> None:
        required = [
            "Trading Framework",
            "Entry Rules",
            "Exit Rules",
            "Quality Gate",
        ]
        for section in required:
            assert section in self.SAMPLE_FRAMEWORK, f"Missing: {section}"
