#!/bin/bash
# Midday scan — email on new swing-trade alerts.
# Skips NYSE holidays and weekends.

set -euo pipefail

TRADING_DIR="/data/trading"
LOG_FILE="$TRADING_DIR/logs/scan.log"
PYTHON="python3"

mkdir -p "$TRADING_DIR/logs"
echo "--- scan_runner started $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

python3 -c "from market_calendar import is_trading_day; import sys; sys.exit(0 if is_trading_day() else 2)" >> "$LOG_FILE" 2>&1 || {
  echo "NYSE closed today — skipping midday scan" >> "$LOG_FILE"
  exit 0
}

HC="$TRADING_DIR/nas/scripts/healthcheck.sh"
[[ -x "$HC" ]] || HC="/scripts/healthcheck.sh"
"$HC" >> "$LOG_FILE" 2>&1 || {
  echo "ERROR: no internet — skipping scan" >> "$LOG_FILE"
  exit 1
}

# Ensure Vertex deps exist (image may predate openai/google-auth in requirements.txt)
pip install --quiet openai google-auth 2>/dev/null || pip install openai google-auth >> "$LOG_FILE" 2>&1 || true

"$PYTHON" scanner.py     >> "$LOG_FILE" 2>&1 || true
"$PYTHON" orchestrate.py >> "$LOG_FILE" 2>&1 || true
"$PYTHON" notify.py --channel email >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

# Submit midday paper orders immediately (Alpaca-priced fallback). No-op unless
# alpaca.enabled=true. The 12:48 cron submit remains an idempotent retry.
"$PYTHON" paper_broker.py submit >> "$TRADING_DIR/logs/paper_broker.log" 2>&1 || true

echo "--- scan_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"
echo "--- scan_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S %Z') ---" >> "$LOG_FILE"
exit $EXIT_CODE
