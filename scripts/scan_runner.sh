#!/bin/bash
# scan_runner.sh — intraday opportunity scanner
# Called by launchd at 12:00 PM and 2:30 PM ET.
# Runs scanner + orchestrate + notify. Only sends email if NEW alerts exist
# (deduplication ensures you don't get duplicate emails for the same signal).

set -euo pipefail

TRADING_DIR="$HOME/trading"
LOG_FILE="$TRADING_DIR/logs/scan.log"
PYTHON="$HOME/.pyenv/shims/python3"
[[ ! -x "$PYTHON" ]] && PYTHON="$(command -v python3)"

mkdir -p "$TRADING_DIR/logs"
echo "--- scan_runner started $(date '+%Y-%m-%d %H:%M:%S ET') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

"$PYTHON" scanner.py     >> "$LOG_FILE" 2>&1 || true
"$PYTHON" orchestrate.py >> "$LOG_FILE" 2>&1 || true
"$PYTHON" notify.py --channel email >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "--- scan_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S ET') ---" >> "$LOG_FILE"
exit $EXIT_CODE
