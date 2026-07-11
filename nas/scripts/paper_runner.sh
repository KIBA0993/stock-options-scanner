#!/bin/bash
# Alpaca PAPER execution runner.  Usage: paper_runner.sh submit|sync
# Submits the current alerts.json as paper option orders, or syncs fills.
# Skips NYSE holidays/weekends. No-ops unless alpaca.enabled=true in config.json.
# Paper-only is enforced inside paper_broker.py.

set -euo pipefail

ACTION="${1:-submit}"
TRADING_DIR="/data/trading"
LOG_FILE="$TRADING_DIR/logs/paper_broker.log"
PYTHON="python3"

mkdir -p "$TRADING_DIR/logs"
echo "--- paper_runner $ACTION started $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

python3 -c "from market_calendar import is_trading_day; import sys; sys.exit(0 if is_trading_day() else 2)" >> "$LOG_FILE" 2>&1 || {
  echo "NYSE closed today — skipping paper $ACTION" >> "$LOG_FILE"
  exit 0
}

"$PYTHON" paper_broker.py "$ACTION" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
echo "--- paper_runner $ACTION exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"
exit $EXIT_CODE
