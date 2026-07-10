#!/bin/bash
# 0–1 DTE intraday rule scanner — runs inside NAS Docker during market hours.
# Logs alerts to data/intraday_0dte_alerts.jsonl for Friday reflect.py review.
# No email by default (high frequency); check logs or jsonl.

set -euo pipefail

TRADING_DIR="/data/trading"
LOG_FILE="$TRADING_DIR/logs/intraday_0dte.log"
PYTHON="python3"

mkdir -p "$TRADING_DIR/logs"
echo "--- intraday_0dte_runner started $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

HC="$TRADING_DIR/nas/scripts/healthcheck.sh"
[[ -x "$HC" ]] || HC="/scripts/healthcheck.sh"
"$HC" >> "$LOG_FILE" 2>&1 || {
  echo "ERROR: no internet — skipping intraday scan" >> "$LOG_FILE"
  exit 1
}

"$PYTHON" intraday_0dte.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "--- intraday_0dte_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"
exit $EXIT_CODE
