#!/bin/bash
# Skip job on weekends and NYSE holidays (exit 0 = cron OK, no run).

set -euo pipefail

cd /data/trading
python3 -c "from market_calendar import trading_day_or_exit; trading_day_or_exit()" || exit 0
exec "$@"
