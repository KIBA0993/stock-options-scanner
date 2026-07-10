#!/bin/bash
# Daily framework backtest after 0DTE reflect (4:20 PM ET).
set -euo pipefail

TRADING_DIR="/data/trading"
LOG_FILE="$TRADING_DIR/logs/framework_backtest.log"
PYTHON="python3"

mkdir -p "$TRADING_DIR/logs"
echo "--- framework_backtest $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"
"$PYTHON" scripts/backtest_frameworks.py >> "$LOG_FILE" 2>&1
