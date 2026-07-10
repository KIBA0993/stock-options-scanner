# Stock Scanner on Synology DS920+

Run the trading scan pipeline on your NAS over wired ethernet for stable, always-on internet. Your Mac stays free for distillation (`distill.py`) and journal work.

## What runs on the NAS

| Time (ET) | Job name | Script | What it does |
|-----------|----------|--------|--------------|
| 9:45 AM Mon–Fri | **morning digest** | `nas/scripts/morning_digest_runner.sh` | Swing scan + always email |
| 12:45 PM Mon–Fri | **midday swing scan** | `nas/scripts/scan_runner.sh` | Swing scan + email on new alerts |
| 9:40–3:50 Mon–Fri (every 10 min) | **0–1 DTE intraday scan** | `nas/scripts/intraday_0dte_runner.sh` | SPY/QQQ/IWM alerts → jsonl; email if `intraday_0dte.email_alerts_enabled` |
| 9:45–3:55 Mon–Fri (every 5 min) | **framework telemetry** | `nas/scripts/intraday_telemetry_runner.sh` | 5-framework backtest snapshots (no email) |
| 4:15 PM Mon–Fri | **0DTE daily reflect** | `nas/scripts/intraday_reflect_runner.sh` | End-of-day P&L summary email |
| 4:20 PM Mon–Fri | **framework backtest** | `nas/scripts/framework_backtest_runner.sh` | Daily framework replay report (no email) |
| 5:15 PM Friday | **weekly swing review** | `nas/scripts/weekly_reflect_runner.sh` | Past week's swing alerts + outcomes |

**Toggle 0–1 DTE real-time alert emails** (daily reflect email is separate):

```json
"intraday_0dte": { "email_alerts_enabled": false }
```

Redeploy after changing `config.json`: `./nas/deploy-to-nas.sh`

Disable Mac `com.trading.reflect.plist` if NAS weekly job is active (avoid duplicate Friday emails).

## One-time NAS setup

### 1. Network (stable internet)

1. Plug the DS920+ into your router with **Ethernet** (not Wi‑Fi).
2. DSM → **Control Panel → Network → General**:
   - Enable **Manually configure DNS**
   - Primary: `1.1.1.1`, Secondary: `8.8.8.8`
3. DSM → **Control Panel → Regional** → Time zone: `(GMT-05:00) Eastern Time` (or your market TZ; the container also sets `TZ=America/New_York`).

### 2. Enable SSH and Docker

1. **Control Panel → Terminal & SNMP** → Enable SSH (port 22).
2. **Package Center** → Install **Container Manager** (Docker).
3. Note your NAS LAN IP (e.g. `10.0.0.57` from DSM → Control Panel → Network).

### 3. Deploy from your Mac

```bash
cd ~/trading
cp nas/nas.env.example nas/nas.env
# Edit nas.env: NAS_HOST, NAS_USER (your DSM username), NAS_PATH

chmod +x nas/deploy-to-nas.sh nas/scripts/*.sh
./nas/deploy-to-nas.sh --test-scan
```

`--test-scan` runs one morning digest immediately so you can verify email + logs.

### 4. Disable Mac scans (after NAS test passes)

```bash
launchctl unload ~/Library/LaunchAgents/com.trading.scan-midday.plist
launchctl unload ~/Library/LaunchAgents/com.trading.scan-afternoon.plist
launchctl unload ~/Library/LaunchAgents/com.trading.morning-digest.plist
```

Keep `com.trading.reflect.plist` on Mac (Friday reflection) unless you move that too.

## Manual operations

```bash
# Live container logs
ssh user@NAS_IP "sudo docker logs -f trading-scanner"

# Scan logs on disk
ssh user@NAS_IP "tail -f /volume1/docker/trading/logs/scan.log"

# Run scan now
ssh user@NAS_IP "sudo docker exec trading-scanner /scripts/scan_runner.sh"

# Rebuild after code changes
./nas/deploy-to-nas.sh
```

## Container Manager UI (alternative)

1. DSM → **Container Manager → Project** → Create
2. Path: `/volume1/docker/trading/nas`
3. Use the included `docker-compose.yml`
4. Build and start

## What stays on Mac

- `distill.py` / `fetch_posts.py` — needs Chrome/X session
- `journal.py` — manual trade logging
- `reflect.py` — weekly framework review (Friday)

After updating creator frameworks on Mac, re-deploy so NAS picks them up:

```bash
./nas/deploy-to-nas.sh
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| SSH timeout | Mac must be on same LAN as NAS; verify IP in DSM |
| `config.json missing` | Ensure `~/trading/config.json` exists locally before deploy |
| Scan skips with "no internet" | Check NAS ethernet link; verify DNS 1.1.1.1 / 8.8.8.8 |
| Docker permission denied | Set `REMOTE_DOCKER="sudo docker"` in `nas.env` or add user to docker group |
| Duplicate emails | Unload Mac launchd scan plists (see above) |

## Architecture

```
DS920+ (wired ethernet)
└── Docker: trading-scanner
    ├── supercronic (ET schedule)
    ├── scanner.py → orchestrate.py → notify.py
    └── volume: /volume1/docker/trading
        ├── config.json
        ├── creators/*/framework-v*.md
        ├── data/
        └── logs/
```
