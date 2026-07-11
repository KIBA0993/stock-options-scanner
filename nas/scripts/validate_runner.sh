#!/bin/bash
# Evening forward-validation: pull today's paper fills, then snapshot + mark +
# score the prediction ledger, and regenerate the HTML scorecard.
# Order matters: sync must land paper_fills.json before validate marks outcomes.
# Skips NYSE holidays/weekends.

set -euo pipefail

TRADING_DIR="/data/trading"
LOG_FILE="$TRADING_DIR/logs/validate.log"
PYTHON="python3"

mkdir -p "$TRADING_DIR/logs"
echo "--- validate_runner started $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

python3 -c "from market_calendar import is_trading_day; import sys; sys.exit(0 if is_trading_day() else 2)" >> "$LOG_FILE" 2>&1 || {
  echo "NYSE closed today — skipping validation" >> "$LOG_FILE"
  exit 0
}

"$PYTHON" paper_broker.py sync        >> "$LOG_FILE" 2>&1 || true   # fills → paper_fills.json
"$PYTHON" validate.py run             >> "$LOG_FILE" 2>&1           # snapshot + mark + report
"$PYTHON" validate.py report --html   >> "$LOG_FILE" 2>&1 || true   # refresh scorecard.html

EXIT_CODE=$?
echo "--- validate_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"
exit $EXIT_CODE
