#!/bin/bash
# morning_digest_runner.sh — LOCAL DEV ONLY (production runs on NAS at 9:45 AM ET).
# LaunchAgents should stay in ~/Library/LaunchAgents/disabled/ to avoid duplicate emails.

set -euo pipefail

TRADING_DIR="$HOME/trading"
LOG_FILE="$TRADING_DIR/logs/morning_digest.log"
PYTHON="$HOME/.pyenv/shims/python3"
[[ ! -x "$PYTHON" ]] && PYTHON="$(command -v python3)"

mkdir -p "$TRADING_DIR/logs"
echo "--- morning_digest_runner started $(date '+%Y-%m-%d %H:%M:%S ET') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

source "$TRADING_DIR/scripts/trading_day_guard.sh" >> "$LOG_FILE" 2>&1 || {
  echo "NYSE closed today — skipping morning digest (Mac)" >> "$LOG_FILE"
  exit 0
}

# 1. Scan (get latest market data)
"$PYTHON" scanner.py    >> "$LOG_FILE" 2>&1 || true

# 2. Score (LLM / heuristic)
"$PYTHON" orchestrate.py >> "$LOG_FILE" 2>&1 || true

# 3. Send morning digest (always sends, even if 0 trades)
"$PYTHON" notify.py --morning-digest --channel email >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "--- morning_digest_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S ET') ---" >> "$LOG_FILE"
exit $EXIT_CODE
