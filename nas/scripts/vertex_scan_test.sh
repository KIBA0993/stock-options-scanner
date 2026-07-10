#!/bin/bash
# One-off Vertex validation: LLM score via orchestrate (dry-run, no alerts written).
set -euo pipefail

TRADING_DIR="/data/trading"
LOG_FILE="$TRADING_DIR/logs/vertex_scan_test.log"
PYTHON="python3"

mkdir -p "$TRADING_DIR/logs"
echo "=== vertex_scan_test started $(date '+%Y-%m-%d %H:%M:%S %Z') ===" | tee -a "$LOG_FILE"

cd "$TRADING_DIR"

echo "--- smoke ---" | tee -a "$LOG_FILE"
"$PYTHON" nas/scripts/vertex_smoke_test.py 2>&1 | tee -a "$LOG_FILE"

echo "--- orchestrate dry-run (vertex LLM) ---" | tee -a "$LOG_FILE"
"$PYTHON" orchestrate.py --dry-run 2>&1 | tee -a "$LOG_FILE"

echo "=== vertex_scan_test finished $(date '+%Y-%m-%d %H:%M:%S %Z') ===" | tee -a "$LOG_FILE"
