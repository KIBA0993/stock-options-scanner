#!/bin/bash
# Morning digest — always sends email even if 0 trades qualify.
# Skips NYSE holidays and weekends.

set -euo pipefail

TRADING_DIR="/data/trading"
LOG_FILE="$TRADING_DIR/logs/morning_digest.log"
PYTHON="python3"

mkdir -p "$TRADING_DIR/logs"
echo "--- morning_digest_runner started $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

python3 -c "from market_calendar import is_trading_day; import sys; sys.exit(0 if is_trading_day() else 2)" >> "$LOG_FILE" 2>&1 || {
  echo "NYSE closed today — skipping morning digest" >> "$LOG_FILE"
  exit 0
}

HC="$TRADING_DIR/nas/scripts/healthcheck.sh"
[[ -x "$HC" ]] || HC="/scripts/healthcheck.sh"
"$HC" >> "$LOG_FILE" 2>&1 || {
  echo "ERROR: no internet — skipping morning digest" >> "$LOG_FILE"
  exit 1
}

"$PYTHON" scanner.py    >> "$LOG_FILE" 2>&1 || true
"$PYTHON" orchestrate.py >> "$LOG_FILE" 2>&1 || true
"$PYTHON" notify.py --morning-digest --channel email >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "--- morning_digest_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"
exit $EXIT_CODE
