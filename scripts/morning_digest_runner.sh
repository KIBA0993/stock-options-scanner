#!/bin/bash
# morning_digest_runner.sh
# Runs at 9:40 AM ET every market day (Mon-Fri).
# Runs the full scan pipeline first, then sends the morning digest email.
# Even if no trades qualify, an email is always sent summarising the outlook.

set -euo pipefail

TRADING_DIR="$HOME/trading"
LOG_FILE="$TRADING_DIR/logs/morning_digest.log"
PYTHON="$HOME/.pyenv/shims/python3"
[[ ! -x "$PYTHON" ]] && PYTHON="$(command -v python3)"

mkdir -p "$TRADING_DIR/logs"
echo "--- morning_digest_runner started $(date '+%Y-%m-%d %H:%M:%S ET') ---" >> "$LOG_FILE"

cd "$TRADING_DIR"

# 1. Scan (get latest market data)
"$PYTHON" scanner.py    >> "$LOG_FILE" 2>&1 || true

# 2. Score (LLM / heuristic)
"$PYTHON" orchestrate.py >> "$LOG_FILE" 2>&1 || true

# 3. Send morning digest (always sends, even if 0 trades)
"$PYTHON" notify.py --morning-digest --channel email >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "--- morning_digest_runner exited code=$EXIT_CODE $(date '+%Y-%m-%d %H:%M:%S ET') ---" >> "$LOG_FILE"
exit $EXIT_CODE
