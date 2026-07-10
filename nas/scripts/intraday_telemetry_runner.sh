#!/bin/bash
# 5-minute telemetry snapshots (no alerts) — fills F2/F4/F5 backtest data.
set -euo pipefail

TRADING_DIR="/data/trading"
LOG_FILE="$TRADING_DIR/logs/intraday_telemetry.log"
PYTHON="python3"

mkdir -p "$TRADING_DIR/logs"
echo "--- telemetry_runner $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

HC="$TRADING_DIR/nas/scripts/healthcheck.sh"
[[ -x "$HC" ]] || HC="/scripts/healthcheck.sh"
"$HC" >> "$LOG_FILE" 2>&1 || {
  echo "ERROR: no internet — skipping telemetry" >> "$LOG_FILE"
  exit 1
}

"$PYTHON" intraday_telemetry.py --source scan_5m >> "$LOG_FILE" 2>&1
