#!/bin/bash
# reflect_runner.sh — LOCAL DEV ONLY (production weekly review on NAS, Fri 5:15 PM ET).

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
source "$TRADING_DIR/scripts/trading_day_guard.sh" >> "$LOG_FILE" 2>&1 || {
  echo "NYSE closed today — skipping reflect (Mac)" >> "$LOG_FILE"
  exit 0
}
"$PYTHON" reflect.py --auto >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "--- reflect_runner.sh exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S') ---" >> "$LOG_FILE"
exit $EXIT_CODE
