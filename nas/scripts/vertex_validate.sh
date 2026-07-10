#!/bin/bash
# Run inside trading-scanner container (DSM Terminal) to validate Vertex.
set -euo pipefail

TRADING_DIR="/data/trading"
LOG="$TRADING_DIR/logs/vertex_startup.log"
mkdir -p "$TRADING_DIR/logs"

{
  echo "=== vertex_validate $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  cd "$TRADING_DIR"
  echo "provider=$(python3 -c 'import json; print(json.load(open("config.json"))["llm"]["provider"])')"
  pip install --quiet openai google-auth 2>&1 || pip install openai google-auth 2>&1 || true
  python3 nas/scripts/vertex_smoke_test.py
  echo "=== vertex_validate done ==="
} 2>&1 | tee -a "$LOG"
