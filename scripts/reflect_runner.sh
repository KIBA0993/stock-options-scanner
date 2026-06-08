#!/bin/bash
# reflect_runner.sh — launchd wrapper for the weekly self-reflection loop
#
# Called by: ~/Library/LaunchAgents/com.trading.reflect.plist
# Schedule:  Every Friday at 5:05 PM local time (after market close)
#
# Logs go to: ~/trading/logs/reflect.log
# Adjust Hour in the plist if your Mac is not in ET (UTC-4/UTC-5):
#   PT  → Hour: 14
#   CT  → Hour: 15
#   MT  → Hour: 16
#   ET  → Hour: 17

set -euo pipefail

TRADING_DIR="$HOME/trading"
LOG_FILE="$TRADING_DIR/logs/reflect.log"
PYTHON="$HOME/.pyenv/shims/python3"

# Fall back to system python if pyenv is not set up
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
fi

mkdir -p "$TRADING_DIR/logs"

echo "--- reflect_runner.sh started $(date '+%Y-%m-%d %H:%M:%S') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"
"$PYTHON" reflect.py --auto >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "--- reflect_runner.sh exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S') ---" >> "$LOG_FILE"
exit $EXIT_CODE
