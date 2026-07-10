#!/bin/bash
# Skip scheduled Mac jobs on weekends and NYSE holidays (exit 0 = OK, no run).
# NAS is the production scheduler; this guard is a safety net if launchd is re-enabled.

set -euo pipefail

TRADING_DIR="${TRADING_DIR:-$HOME/trading}"
PYTHON="${PYTHON:-$(command -v python3)}"

cd "$TRADING_DIR"
"$PYTHON" -c "from market_calendar import is_trading_day; import sys; sys.exit(0 if is_trading_day() else 2)"
