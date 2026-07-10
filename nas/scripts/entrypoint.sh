#!/bin/bash
set -euo pipefail

LOG="/data/trading/logs/vertex_startup.log"
mkdir -p /data/trading/data /data/trading/logs
echo "=== entrypoint $(date '+%Y-%m-%d %H:%M:%S %Z') ===" >> "$LOG"

if [[ ! -f /data/trading/config.json ]]; then
  echo "ERROR: /data/trading/config.json missing." | tee -a "$LOG"
  echo "Copy config.json from your Mac before starting the container."
  exit 1
fi

echo "trading-scanner starting — TZ=${TZ:-unset} HOME=${HOME}" | tee -a "$LOG"
if [[ -f /data/trading/requirements.txt ]]; then
  echo "Installing/updating Python deps …" >> "$LOG"
  pip install --quiet -r /data/trading/requirements.txt >> "$LOG" 2>&1 \
    || pip install -r /data/trading/requirements.txt >> "$LOG" 2>&1 \
    || echo "WARN: pip install failed — scan_runner will retry" >> "$LOG"
fi

/scripts/healthcheck.sh >> "$LOG" 2>&1 \
  || echo "WARN: initial connectivity check failed; cron will retry" >> "$LOG"

echo "Running Vertex startup smoke test …" >> "$LOG"
(
  cd /data/trading
  python3 nas/scripts/vertex_smoke_test.py
) >> "$LOG" 2>&1 || echo "WARN: Vertex smoke test failed" >> "$LOG"

exec /usr/local/bin/supercronic -passthrough-logs /etc/crontab
