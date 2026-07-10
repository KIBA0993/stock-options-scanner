#!/usr/bin/env python3
"""
fetch_posts.py — Collect X/Twitter posts for creator distillation

Usage:
  python fetch_posts.py @kpak82
  python fetch_posts.py @kpak82 --limit 200

Output:
  ~/trading/creators/{handle}/posts_raw.txt

How X posts are fetched:
  - Uses the X API (requires X_BEARER_TOKEN in config.json)
  - Maximum 3,200 most recent posts (X API hard cap regardless of account age)
  - If no API key: prints manual collection instructions

Manual fallback (no API key needed):
  Open X in Chrome, scroll through creator's profile, copy posts into:
  ~/trading/creators/{handle}/posts_raw.txt
  One post per line, or paste full browser text — distill.py handles both formats.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
logger = logging.getLogger("fetch_posts")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        return json.load(f)


def fetch_via_api(handle: str, limit: int, bearer_token: str) -> list[str]:
    """Fetch posts via X API v2 (requires bearer token)."""
    try:
        import requests
    except ImportError:
        logger.error("requests not installed. Run: pip install requests")
        sys.exit(1)

    handle = handle.lstrip("@")
    headers = {"Authorization": f"Bearer {bearer_token}"}

    # Step 1: resolve handle → user ID
    user_url = f"https://api.twitter.com/2/users/by/username/{handle}"
    resp = requests.get(user_url, headers=headers, timeout=10)
    if resp.status_code != 200:
        logger.error(f"Could not resolve @{handle}: {resp.status_code} {resp.text[:200]}")
        sys.exit(1)
    user_id = resp.json()["data"]["id"]
    logger.info(f"@{handle} → user_id={user_id}")

    # Step 2: fetch timeline (max 100 per page, paginate up to `limit`)
    posts: list[str] = []
    url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    params = {
        "max_results":  min(100, limit),
        "tweet.fields": "created_at,text,public_metrics,attachments",
        "expansions":   "attachments.media_keys",
        "media.fields": "media_key,type,url,preview_image_url",
        "exclude":      "retweets",
    }
    next_token = None

    while len(posts) < limit:
        if next_token:
            params["pagination_token"] = next_token

        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            logger.warning("Rate limited — waiting 15 minutes...")
            time.sleep(900)
            continue
        if resp.status_code != 200:
            logger.error(f"Timeline API error: {resp.status_code} {resp.text[:200]}")
            break

        data = resp.json()

        # Build media_key → URL map from the includes block
        media_map: dict[str, str] = {}
        for m in data.get("includes", {}).get("media", []):
            key = m.get("media_key", "")
            # Photos have 'url'; videos/gifs have 'preview_image_url'
            img_url = m.get("url") or m.get("preview_image_url", "")
            if key and img_url and m.get("type") == "photo":
                media_map[key] = img_url

        for tweet in data.get("data", []):
            text    = tweet.get("text", "").strip()
            date    = tweet.get("created_at", "")[:10]
            metrics = tweet.get("public_metrics", {})
            likes   = metrics.get("like_count", 0)
            rt      = metrics.get("retweet_count", 0)

            # Collect image URLs attached to this tweet
            media_keys = tweet.get("attachments", {}).get("media_keys", [])
            image_urls = [media_map[k] for k in media_keys if k in media_map]

            line = f"[{date}] (likes:{likes} rt:{rt}) {text}"
            if image_urls:
                line += f"\n[IMAGES: {', '.join(image_urls)}]"
            posts.append(line)

        next_token = data.get("meta", {}).get("next_token")
        if not next_token:
            break

        time.sleep(1)  # be polite to the API

    images_found = sum(1 for p in posts if "[IMAGES:" in p)
    logger.info(f"Fetched {len(posts)} posts for @{handle} ({images_found} with chart images)")
    return posts


def print_manual_instructions(handle: str) -> None:
    handle = handle.lstrip("@")
    out_path = BASE_DIR / "creators" / handle / "posts_raw.txt"
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           MANUAL POST COLLECTION INSTRUCTIONS                 ║
╠══════════════════════════════════════════════════════════════╣
║  No X API key found. Use one of these options:               ║
╠══════════════════════════════════════════════════════════════╣

  OPTION A — Browser copy-paste (easiest):
  1. Open x.com/@{handle} in Chrome
  2. Scroll through their timeline (focus on trading posts)
  3. Copy the text of posts that show trade setups, entries,
     exits, reasoning, or risk management
  4. Paste into: {out_path}

  OPTION B — Browser "Select All" export:
  1. Open x.com/@{handle} in Chrome
  2. Press Ctrl+A then Ctrl+C on the page (gets visible text)
  3. Paste into: {out_path}
  4. distill.py will clean up the noise automatically

  OPTION C — Add X API key to config.json:
  Add to ~/trading/config.json:
    "x_api": {{
      "bearer_token": "YOUR_BEARER_TOKEN"
    }}
  Then rerun: python fetch_posts.py @{handle}

  OPTION D — Run with opencli (if installed):
    opencli twitter user --handle @{handle} --limit 200 -f json

╠══════════════════════════════════════════════════════════════╣
║  After collecting posts, run:                                 ║
║    python distill.py @{handle}                  ║
╚══════════════════════════════════════════════════════════════╝
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch X posts for creator distillation")
    parser.add_argument("handle", help="X handle (e.g. @kpak82 or kpak82)")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max posts to fetch (default 200, API cap 3200)")
    args = parser.parse_args()

    handle = args.handle.lstrip("@")
    out_dir = BASE_DIR / "creators" / handle
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "posts_raw.txt"

    config = load_config()
    bearer_token = config.get("x_api", {}).get("bearer_token", "")

    if not bearer_token:
        print_manual_instructions(handle)
        return

    posts = fetch_via_api(handle, args.limit, bearer_token)
    if not posts:
        logger.warning("No posts fetched. Try manual collection.")
        print_manual_instructions(handle)
        return

    out_path.write_text("\n".join(posts), encoding="utf-8")
    logger.info(f"Saved {len(posts)} posts → {out_path}")
    print(f"\n✓ {len(posts)} posts saved to {out_path}")
    print(f"→ Now run: python distill.py @{handle}")


if __name__ == "__main__":
    main()
