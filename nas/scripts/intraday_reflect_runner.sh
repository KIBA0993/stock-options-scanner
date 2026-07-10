#!/bin/bash
# Daily 0DTE performance review — after market close on trading days.

set -euo pipefail

TRADING_DIR="/data/trading"
LOG_FILE="$TRADING_DIR/logs/intraday_reflect.log"
PYTHON="python3"

mkdir -p "$TRADING_DIR/logs"
echo "--- intraday_reflect_runner started $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

/data/trading/nas/scripts/run_if_market_open.sh "$PYTHON" reflect.py --intraday-daily >> "$LOG_FILE" 2>&1 || {
  echo "Market closed — skipping intraday reflect" >> "$LOG_FILE"
  exit 0
}

EXIT_CODE=$?
echo "--- intraday_reflect_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"
exit $EXIT_CODE
