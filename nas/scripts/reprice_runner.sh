#!/bin/bash
# Reprice backfill — fills live option contracts into any alert the morning scan
# archived before options pricing settled. Runs ~10:05 ET as a safety net after
# the 9:45 morning digest. Updates alerts.json + the archive validate.py reads.
# Skips NYSE holidays and weekends. Does not re-send email (notify dedups by day).

set -euo pipefail

TRADING_DIR="/data/trading"
LOG_FILE="$TRADING_DIR/logs/reprice.log"
PYTHON="python3"

mkdir -p "$TRADING_DIR/logs"
echo "--- reprice_runner started $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

python3 -c "from market_calendar import is_trading_day; import sys; sys.exit(0 if is_trading_day() else 2)" >> "$LOG_FILE" 2>&1 || {
  echo "NYSE closed today — skipping reprice" >> "$LOG_FILE"
  exit 0
}

"$PYTHON" reprice.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
echo "--- reprice_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"
exit $EXIT_CODE
