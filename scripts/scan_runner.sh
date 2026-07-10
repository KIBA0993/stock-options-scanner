#!/bin/bash
# scan_runner.sh — LOCAL DEV ONLY (production midday scan on NAS at 12:45 PM ET).

set -euo pipefail

TRADING_DIR="$HOME/trading"
LOG_FILE="$TRADING_DIR/logs/scan.log"
PYTHON="$HOME/.pyenv/shims/python3"
[[ ! -x "$PYTHON" ]] && PYTHON="$(command -v python3)"

mkdir -p "$TRADING_DIR/logs"
echo "--- scan_runner started $(date '+%Y-%m-%d %H:%M:%S ET') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

source "$TRADING_DIR/scripts/trading_day_guard.sh" >> "$LOG_FILE" 2>&1 || {
  echo "NYSE closed today — skipping scan (Mac)" >> "$LOG_FILE"
  exit 0
}

"$PYTHON" scanner.py     >> "$LOG_FILE" 2>&1 || true
"$PYTHON" orchestrate.py >> "$LOG_FILE" 2>&1 || true
"$PYTHON" notify.py --channel email >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "--- scan_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S ET') ---" >> "$LOG_FILE"
exit $EXIT_CODE
