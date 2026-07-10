#!/usr/bin/env python3
"""
distill.py — Creator Trading Framework Distiller

Reads a creator's raw X posts and uses an LLM to extract a structured
trading framework document: setup triggers, preferred instruments, entry/exit
rules, risk management, and what they avoid.

Usage:
  python distill.py @kpak82
  python distill.py @kpak82 --posts-file ~/trading/creators/kpak82/posts_raw.txt
  python distill.py @kpak82 --refresh   # creates framework-v2.md from latest posts

Output:
  ~/trading/creators/{handle}/framework-v{n}.md
  (orchestrate.py always loads the latest version automatically)

Post collection:
  Run fetch_posts.py first, or manually paste posts into posts_raw.txt.
  See fetch_posts.py for instructions.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR    = Path.home() / "trading"
CONFIG_PATH = BASE_DIR / "config.json"
LOG_DIR     = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("distill")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(f"config.json not found at {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert at analyzing traders' X/Twitter posts to extract their
repeatable trading methodology. Your job is to distill posts from a specific
trader into a structured framework document that can be used by an AI system
to evaluate whether a given stock options setup matches how this trader thinks.

Be concrete and specific. Do not generalize. Use the trader's own language
where possible. If a piece of information is not present in the posts, say
"Not documented" rather than inventing it.

IMPORTANT: Focus on what makes this trader DIFFERENT from generic advice.
Generic statements like "cut losses quickly" are not useful. Specific patterns
like "exits half position at 50% gain, runs remainder to 2x" are useful.

Output ONLY valid markdown — no preamble, no explanation outside the document.
"""

EXTRACTION_PROMPT = """\
Here are X/Twitter posts from trader {handle}. Analyze them and extract a
structured trading framework document.

<posts>
{posts_text}
</posts>

Write the framework document using EXACTLY this structure (fill in each section
based only on what appears in the posts — mark "Not documented" for anything
not found):

# Trading Framework: {handle}
Distilled: {today}
Version: {version}
Asset focus: [what markets/assets they primarily trade]
Note: [any limitations — e.g. "posts contain parody content", "crypto-focused"]

## Trading Personality
[How they communicate — confident/cautious, contrarian/trend-following,
short-term/swing, technical/fundamental. 2-4 sentences max.]

## Market Conditions They Trade
[What macro or market environment they prefer — trending markets, high VIX,
low VIX, post-earnings, pre-Fed, etc.]

## Market Conditions They Avoid
[When they sit on their hands — what causes them to step aside]

## Setup Triggers — What Gets Their Attention
[Specific conditions they call out: volume spikes, chart patterns, relative
strength, sector rotation, news catalysts. Be specific.]

## Preferred Instruments & Timeframes
- Calls vs puts preference:
- Typical DTE range:
- Favored underlyings (specific tickers/ETFs they trade most):
- Timeframe (0DTE / weekly / monthly / LEAPS):

## Entry Rules
[What they wait for before entering — breakout confirmation, pullback entry,
specific indicator levels, time of day, price action signal. Be specific.]

## Exit Rules
- Profit target approach:
- Stop loss approach:
- Time stop (if any):
- Partial exits:

## Risk Management
- Position sizing:
- Max loss per trade:
- How they manage losing streaks:

## Red Flags — What They Explicitly Avoid
[Trades, setups, or conditions they warn against. Direct quotes if available.]

## Quality Gate — Sample Setups
[Extract 3-5 specific trade setups from the posts that best represent this
framework. For each, format as:
  Date: [date]
  Setup: [what triggered the trade]
  Direction: [call/put, long/short]
  Outcome: [if mentioned]
  Why it fits the framework: [which rules above this exemplifies]
]

## Honest Limitations
[What this framework CANNOT tell you — information gaps, things the trader
doesn't document, areas where the framework is thin or uncertain.]
"""


IMAGE_ANALYSIS_PROMPT = """\
This image was posted by trader {handle} on {date} with the caption:
"{tweet_text}"

You are analyzing this chart image to extract trading methodology.

Answer ALL of the following — mark "Not visible" if you can't determine it:

1. CHART TYPE: Is this a single chart, or two charts being compared side by side (historical analog)?

2. IF HISTORICAL ANALOG (two charts compared):
   - Asset in current chart:
   - Current time period shown:
   - Historical period being compared to:
   - What pattern is repeating? (e.g. "bull flag consolidation before breakout", "rounding bottom", "ascending triangle")
   - Predicted outcome based on the analog:
   - Any price targets or levels shown?

3. IF SINGLE CHART:
   - Asset:
   - Timeframe (1m, 5m, 1h, daily, weekly):
   - Pattern identified (describe what you see):
   - Key levels marked (support, resistance, targets):
   - Trend direction implied:
   - Any annotations, arrows, or text drawn on the chart?

4. TRADING SIGNAL: What trade does this chart image suggest? (calls/puts, long/short, wait, etc.)

5. TRANSFERABLE INSIGHT: In 1-2 sentences, what general trading methodology principle does this image demonstrate that could apply to any stock or asset?
"""

IMAGE_SUMMARY_PROMPT = """\
Below are insights extracted from {n} chart images posted by trader {handle}.
Synthesize these into a "Visual Chart Analysis" section for their trading framework.

Focus on:
- Recurring patterns they use (historical analogs, specific chart formations)
- How they identify and use historical pattern comparisons
- Their preferred timeframes for visual analysis
- Any stocks or assets they repeatedly chart (beyond crypto)
- The general methodology behind their visual pattern recognition

Image insights:
{insights_text}

Write a ## Visual Chart Analysis section with subsections:
### Historical Analog Method
[How they compare current charts to historical periods]

### Recurring Chart Patterns
[List the most common patterns found in their images]

### Transferable Methodology (Non-Crypto)
[What visual analysis techniques work for any asset, including US stocks]

### Sample Visual Setups (from images)
[2-3 specific examples from the image analysis with dates]
"""


OLLAMA_BASE_URL = "http://localhost:11434/v1"


def _ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


def call_llm(prompt: str, config: dict) -> str:
    llm_cfg  = config.get("llm", {})
    provider = llm_cfg.get("provider", "anthropic")
    api_key  = llm_cfg.get("api_key", "")
    model    = llm_cfg.get("model", "claude-opus-4-5")

    # Auto-route to Ollama if no API key is configured (non-Vertex providers)
    if provider == "ollama" or (
        provider != "vertex"
        and (not api_key or api_key == "PLACEHOLDER")
    ):
        ollama_model = llm_cfg.get("ollama_model", "qwen2.5:14b")
        if _ollama_running():
            logger.info(f"Using local Ollama ({ollama_model}) for distillation.")
            return _call_ollama(prompt, ollama_model, config)
        else:
            logger.error(
                "Ollama is not running and no API key is set.\n"
                "  Start Ollama: brew services start ollama\n"
                "  Or set an API key in config.json"
            )
            sys.exit(1)

    if provider == "anthropic":
        return _call_anthropic(prompt, api_key, model, config)
    elif provider == "openai":
        return _call_openai(prompt, api_key, model, config)
    elif provider == "mammouth":
        return _call_openai(prompt, api_key, model, config,
                            base_url="https://api.mammouth.ai/v1")
    elif provider == "openai_compatible":
        return _call_openai(prompt, api_key, model, config,
                            base_url=config.get("llm", {}).get("base_url"))
    elif provider == "vertex":
        from vertex_llm import is_vertex_configured, vertex_chat
        if not is_vertex_configured(llm_cfg):
            logger.error("Vertex AI: set llm.project_id and auth (adc or service account JSON)")
            sys.exit(1)
        logger.info(f"Calling Vertex AI ({llm_cfg.get('model', 'gemini-2.5-flash')})…")
        return vertex_chat(prompt, SYSTEM_PROMPT, llm_cfg, stream=True)
    else:
        logger.error(f"Unknown LLM provider: {provider}")
        sys.exit(1)


def _call_anthropic(prompt: str, api_key: str, model: str, config: dict) -> str:
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic not installed. Run: pip install anthropic")
        sys.exit(1)

    max_tokens = config.get("llm", {}).get("max_tokens", 4096)
    client = anthropic.Anthropic(api_key=api_key)

    logger.info(f"Calling Anthropic ({model})...")
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(prompt: str, api_key: str, model: str, config: dict, base_url: str | None = None) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai not installed. Run: pip install openai")
        sys.exit(1)

    max_tokens = config.get("llm", {}).get("max_tokens", 4096)
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    logger.info(f"Calling OpenAI-compatible API ({model}) via streaming…")
    # Use streaming to avoid server-side 60s timeout on long generations
    chunks = []
    with client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                chunks.append(delta)
    return "".join(chunks)


def _call_ollama(prompt: str, model: str, config: dict) -> str:
    """Call local Ollama via its OpenAI-compatible endpoint (no API key needed)."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai not installed. Run: pip install openai")
        sys.exit(1)

    max_tokens = config.get("llm", {}).get("max_tokens", 4096)
    client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

    logger.info(f"Calling Ollama ({model}) at {OLLAMA_BASE_URL}...")
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Image / Vision analysis
# ---------------------------------------------------------------------------

def _fetch_image_b64(url: str) -> tuple[str, str] | None:
    """Download an image URL and return (base64_data, media_type). Returns None on failure."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
            return base64.b64encode(data).decode(), content_type
    except Exception as exc:
        logger.warning(f"Could not fetch image {url}: {exc}")
        return None


def _call_vision_llm(tweet_text: str, image_b64: str, media_type: str,
                     date: str, handle: str, config: dict) -> str:
    """Send a tweet + chart image to Claude Vision and return the analysis."""
    llm_cfg  = config.get("llm", {})
    api_key  = llm_cfg.get("api_key", "")
    model    = llm_cfg.get("model", "claude-sonnet-4-5")
    provider = llm_cfg.get("provider", "anthropic")

    prompt_text = IMAGE_ANALYSIS_PROMPT.format(
        handle=f"@{handle}", date=date, tweet_text=tweet_text[:300]
    )

    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }],
            )
            return response.content[0].text
        elif provider == "vertex":
            from vertex_llm import vertex_vision
            return vertex_vision(prompt_text, image_b64, media_type, llm_cfg)
        else:
            # OpenAI-compatible: mammouth, openai, openai_compatible
            from openai import OpenAI
            base_url = None
            if provider == "mammouth":
                base_url = "https://api.mammouth.ai/v1"
            elif provider == "openai_compatible":
                base_url = llm_cfg.get("base_url")
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)
            response = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{media_type};base64,{image_b64}"
                        }},
                    ],
                }],
            )
            return response.choices[0].message.content
    except Exception as exc:
        logger.warning(f"Vision LLM call failed: {exc}")
        return ""


def _extract_image_lines(posts_text: str) -> list[dict]:
    """
    Parse posts_raw.txt and return list of dicts for posts with images:
    {date, tweet_text, image_urls, likes}
    Only returns posts with chart-related keywords to skip unrelated images.
    """
    CHART_KEYWORDS = re.compile(
        r"pattern|similar|analog|like 20\d\d|reminds|fractal|repeat|"
        r"looks like|compare|chart|setup|breakout|resistance|support|"
        r"target|level|rsi|ema|macd|bull|bear|wedge|triangle|channel|flag|"
        r"bounce|rejection|retest|consolidat",
        re.IGNORECASE
    )
    results = []
    lines   = posts_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Match a post header line like [2026-05-22] (likes:617 rt:63) text...
        m = re.match(r"^\[(\d{4}-\d{2}-\d{2})\]\s+\(likes:(\d+).*?\)\s+(.*)", line)
        if m:
            date  = m.group(1)
            likes = int(m.group(2))
            text  = m.group(3)
            # Check if next line is [IMAGES: ...]
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("[IMAGES:"):
                img_line = lines[i + 1].strip()
                urls_str = re.sub(r"^\[IMAGES:\s*", "", img_line).rstrip("]")
                image_urls = [u.strip() for u in urls_str.split(",") if u.strip()]
                # Only keep if tweet text has chart-related content
                if image_urls and (CHART_KEYWORDS.search(text) or likes >= 50):
                    results.append({
                        "date": date, "tweet_text": text,
                        "image_urls": image_urls, "likes": likes,
                    })
                i += 2
                continue
        i += 1

    # Sort by likes descending so we process the most impactful posts first
    results.sort(key=lambda x: x["likes"], reverse=True)
    return results


def analyze_chart_images(posts_text: str, handle: str, config: dict,
                         max_images: int = 40) -> str:
    """
    Find posts with chart images, analyze each via Claude Vision,
    then synthesize a visual methodology section.

    Returns the synthesized markdown section, or "" if no images found.
    """
    image_posts = _extract_image_lines(posts_text)
    if not image_posts:
        logger.info(f"No chart images found in posts for @{handle}")
        return ""

    total_images = sum(len(p["image_urls"]) for p in image_posts)
    logger.info(
        f"Found {len(image_posts)} posts with chart images "
        f"({total_images} total) for @{handle}. Analyzing top {max_images}…"
    )

    insights = []
    analyzed = 0
    for post in image_posts:
        if analyzed >= max_images:
            break
        for img_url in post["image_urls"]:
            if analyzed >= max_images:
                break
            logger.info(f"  Analyzing image {analyzed + 1}/{max_images}: {img_url[:60]}…")
            result = _fetch_image_b64(img_url)
            if not result:
                continue
            b64, media_type = result
            analysis = _call_vision_llm(
                tweet_text=post["tweet_text"],
                image_b64=b64,
                media_type=media_type,
                date=post["date"],
                handle=handle,
                config=config,
            )
            if analysis:
                insights.append(
                    f"--- Image from {post['date']} (likes:{post['likes']}) ---\n"
                    f"Tweet: {post['tweet_text'][:200]}\n"
                    f"Analysis:\n{analysis}\n"
                )
            analyzed += 1
            time.sleep(0.5)   # avoid rate limiting

    if not insights:
        logger.info(f"No successful image analyses for @{handle}")
        return ""

    logger.info(f"Synthesizing {len(insights)} image analyses for @{handle}…")
    summary_prompt = IMAGE_SUMMARY_PROMPT.format(
        handle=f"@{handle}",
        n=len(insights),
        insights_text="\n\n".join(insights[:30]),  # cap to avoid context overflow
    )
    try:
        # Re-use call_llm so synthesis always uses the configured provider
        return call_llm(summary_prompt, config)
    except Exception as exc:
        logger.warning(f"Image synthesis failed: {exc}")
        return "\n".join(insights)   # fallback: return raw insights


# ---------------------------------------------------------------------------
# Framework versioning
# ---------------------------------------------------------------------------
def next_version(creator_dir: Path) -> int:
    """Return the next framework version number."""
    existing = list(creator_dir.glob("framework-v*.md"))
    if not existing:
        return 1
    versions = []
    for f in existing:
        m = re.search(r"framework-v(\d+)\.md", f.name)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1 if versions else 1


def latest_posts_file(creator_dir: Path) -> Path | None:
    """Find the most recently modified posts file."""
    candidates = list(creator_dir.glob("posts_raw*.txt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Post cleaning
# ---------------------------------------------------------------------------
def clean_posts(raw: str) -> str:
    """
    Remove obvious noise from raw X page text or copy-pasted content.
    Keeps trading-relevant lines, drops UI chrome.
    """
    noise_patterns = [
        r"^(Home|Explore|Notifications|Messages|Grok|Premium|Profile|More)$",
        r"^(Follow|Following|Followed|Followers|Likes|Replies|Media|Views)$",
        r"^\d+\s*(repost|like|reply|view|bookmark)s?$",
        r"^@\w+$",
        r"^\d{1,3}[KMB]?$",          # follower counts
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+$",
    ]
    noise_re = [re.compile(p, re.IGNORECASE) for p in noise_patterns]

    lines = raw.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(p.match(line) for p in noise_re):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Distill creator trading framework from X posts")
    parser.add_argument("handle", help="X handle (e.g. @kpak82 or kpak82)")
    parser.add_argument("--posts-file", type=Path, default=None,
                        help="Path to file containing raw X posts (default: auto-detect posts_raw.txt)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force new version even if framework-v1.md already exists")
    parser.add_argument("--analyze-images", action="store_true",
                        help="Download and analyze chart images via Claude Vision (requires image URLs in posts_raw.txt)")
    parser.add_argument("--max-images", type=int, default=40,
                        help="Max chart images to analyze (default: 40)")
    parser.add_argument("--provider", default=None,
                        help="Override LLM provider: ollama | mammouth | anthropic | openai")
    args = parser.parse_args()

    handle       = args.handle.lstrip("@")
    creator_dir  = BASE_DIR / "creators" / handle
    creator_dir.mkdir(parents=True, exist_ok=True)

    # Find posts file
    posts_file = args.posts_file or latest_posts_file(creator_dir)
    if not posts_file or not posts_file.exists():
        print(f"""
No posts file found for @{handle}.

Run first:
  python fetch_posts.py @{handle}

Or paste posts manually into:
  {creator_dir / 'posts_raw.txt'}

Then re-run:
  python distill.py @{handle}
""")
        sys.exit(1)

    raw_text = posts_file.read_text(encoding="utf-8")
    if len(raw_text.strip()) < 100:
        logger.error(f"Posts file is too short ({len(raw_text)} chars). Add more post content.")
        sys.exit(1)

    logger.info(f"Posts file: {posts_file} ({len(raw_text):,} chars)")
    posts_text = clean_posts(raw_text)
    logger.info(f"After cleaning: {len(posts_text):,} chars")

    # Truncate if very long (LLM context limit)
    max_chars = 80_000
    if len(posts_text) > max_chars:
        posts_text = posts_text[:max_chars]
        logger.warning(f"Truncated to {max_chars:,} chars to fit LLM context window")

    config  = load_config()
    if args.provider:
        config.setdefault("llm", {})["provider"] = args.provider
        logger.info(f"LLM provider overridden to: {args.provider}")
    version = next_version(creator_dir) if args.refresh else (next_version(creator_dir) if next_version(creator_dir) == 1 else None)

    if version is None:
        existing = list(creator_dir.glob("framework-v*.md"))
        latest = max(existing, key=lambda p: p.stat().st_mtime)
        print(f"\nFramework already exists: {latest}")
        print("Use --refresh to create a new version from updated posts.")
        sys.exit(0)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # --- Optional: image analysis via Claude Vision ---
    visual_section = ""
    image_cache_path = creator_dir / "image_analysis_cache.md"
    if args.analyze_images:
        # Use cached result if it exists from a previous run today (avoids re-analyzing on retry)
        if image_cache_path.exists():
            cached = image_cache_path.read_text(encoding="utf-8")
            if cached.strip():
                logger.info(f"Using cached image analysis from {image_cache_path}")
                visual_section = cached
        if not visual_section:
            logger.info(f"Running image analysis for @{handle} (max {args.max_images} images)…")
            visual_section = analyze_chart_images(
                posts_text=raw_text,        # use raw_text (has [IMAGES: ...] lines)
                handle=handle,
                config=config,
                max_images=args.max_images,
            )
            if visual_section:
                image_cache_path.write_text(visual_section, encoding="utf-8")
                logger.info("Image analysis complete — cached and including visual section in framework.")
            else:
                logger.info("No image insights found. Continuing with text-only distillation.")

    prompt = EXTRACTION_PROMPT.format(
        handle=f"@{handle}",
        posts_text=posts_text,
        today=today,
        version=f"v{version}",
    )

    logger.info(f"Distilling @{handle} → framework-v{version}.md ...")
    framework_text = call_llm(prompt, config)

    # Append the visual section separately so Claude's token limit doesn't cut it off
    if visual_section and "## Visual Chart Analysis" not in framework_text:
        framework_text = framework_text.rstrip() + "\n\n## Visual Chart Analysis\n" + visual_section

    out_path = creator_dir / f"framework-v{version}.md"
    out_path.write_text(framework_text, encoding="utf-8")

    logger.info(f"Framework written: {out_path}")
    print(f"""
✓ Framework distilled: {out_path}

Next steps:
  1. Open {out_path} and read it
  2. Check the "Quality Gate — Sample Setups" section
  3. Find 5 recent @{handle} trade calls on X
  4. Verify each against the framework (target: ≥80% match)
  5. If below 80%, add more posts and run:
       python distill.py @{handle} --refresh
  6. Once verified, orchestrate.py will use it automatically
""")


if __name__ == "__main__":
    main()
