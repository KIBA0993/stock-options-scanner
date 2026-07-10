#!/usr/bin/env python3
"""Quick Vertex LLM smoke test — logs result for deploy validation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

def _trading_dir() -> Path:
    env = os.environ.get("TRADING_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    here = Path(__file__).resolve()
    # nas/scripts/vertex_smoke_test.py → repo root
    return here.parents[2]


TRADING = _trading_dir()
sys.path.insert(0, str(TRADING))

from vertex_llm import is_vertex_configured, vertex_chat  # noqa: E402


def _load_llm_cfg() -> dict:
    import json
    with open(TRADING / "config.json") as f:
        return json.load(f).get("llm", {})


def main() -> int:
    cfg = _load_llm_cfg()
    if cfg.get("provider") != "vertex":
        print(f"FAIL: llm.provider is {cfg.get('provider')!r}, expected vertex")
        return 1
    if not is_vertex_configured(cfg):
        print("FAIL: Vertex not configured (project_id + adc credentials)")
        return 1
    reply = vertex_chat("Reply with exactly: VERTEX_OK", "You are terse.", cfg, stream=False)
    print(f"provider=vertex model={cfg.get('model')} reply={reply.strip()!r}")
    if "VERTEX_OK" not in reply.upper():
        print("FAIL: unexpected reply")
        return 1
    print("PASS: Vertex smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
