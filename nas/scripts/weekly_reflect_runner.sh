#!/bin/bash
# Weekly swing-scan review — Friday after close (runs even on NYSE holidays).
# Scores emailed alerts from the week and sends summary email.

set -euo pipefail

TRADING_DIR="/data/trading"
LOG_FILE="$TRADING_DIR/logs/weekly_reflect.log"
PYTHON="python3"

mkdir -p "$TRADING_DIR/logs"
echo "--- weekly_reflect_runner started $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"
"$PYTHON" reflect.py --weekly >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "--- weekly_reflect_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"
exit $EXIT_CODE
