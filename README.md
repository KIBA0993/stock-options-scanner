# AI-Assisted Stock Options Scanner

An LLM-powered pipeline that scans US equity markets daily, identifies short-term call/put option opportunities (5–10 per week), and runs a weekly self-reflection loop to improve its own creator frameworks over time.

## Architecture

```
scanner.py → data/all_data.json
                  ↓
            orchestrate.py  (LLM scoring against creator frameworks)
                  ↓
            data/alerts.json
                  ↓
            notify.py  (Telegram / email)

Every Friday 5pm (launchd):
            reflect.py --auto  →  reflect_history.jsonl  →  creators/*/amendments/
```

## Features

- **Volume + TA scan** — tradingview-screener for RSI, MACD, EMA, relative volume across NASDAQ/NYSE
- **Creator knowledge distillation** — extracts trading frameworks from X posts by selected traders
- **LLM scoring** — Claude (via Mammouth AI) or local Ollama applies creator frameworks to each candidate
- **Trade journal** — CLI to log, close, and track P&L in R-multiples
- **Backtester** — validates historical signal quality via yfinance
- **Weekly self-reflection** — detects repeating miss patterns and drafts framework amendments for human review

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp config.json.template config.json
# Edit config.json and fill in your keys:
#   llm.api_key      → Mammouth AI key (https://mammouth.ai) or Anthropic key
#   x_api.bearer_token → X/Twitter bearer token (free tier works)
```

### 3. Distill creator frameworks

The system learns from four X creators out of the box. Refresh their content anytime:

```bash
python distill.py --handle kpak82
python distill.py --handle MasterPandaWu
python distill.py --handle CryptoKaleo
python distill.py --handle puppy_trades
```

Frameworks are stored in `creators/{handle}/framework-v1.md`.

### 4. Run the scanner

```bash
python scanner.py          # scan + save all_data.json
python orchestrate.py      # score candidates, write alerts.json
python notify.py           # send Telegram/email alerts
```

Or with the shell shortcut (add to `~/.zshrc`):

```bash
trade-scan                 # runs all three in sequence
```

### 5. Enable weekly self-reflection (macOS)

```bash
# Register the launchd agent (runs every Friday at 5:05 PM)
launchctl load ~/Library/LaunchAgents/com.trading.reflect.plist

# Or trigger manually any time:
python reflect.py --auto
```

## Shell shortcuts (add to `~/.zshrc`)

```bash
trade-scan()     { cd ~/trading && python scanner.py "$@" && python orchestrate.py && python notify.py; }
trade-scan-score() { cd ~/trading && python orchestrate.py "$@"; }
journal()        { cd ~/trading && python journal.py "$@"; }
backtest()       { cd ~/trading && python backtest.py "$@"; }
reflect()        { cd ~/trading && python reflect.py "$@"; }
```

## Usage

### Scanner

```bash
python scanner.py                          # default scan
python scanner.py --top-n 20              # expand candidate pool
python orchestrate.py --dry-run           # score without writing alerts
python orchestrate.py --no-llm            # force heuristic (no API call)
python orchestrate.py --ignore-budget     # bypass 10-trade/week cap
```

### Trade Journal

```bash
python journal.py log                     # log a new trade
python journal.py log --from-alert NVDA  # pre-fill from latest alert
python journal.py close NVDA --exit 4.20 # close trade, compute R-multiple
python journal.py status                 # open positions + budget
python journal.py summary --weeks 4      # 4-week win rate breakdown
python journal.py verify                 # cross-check vs yfinance prices
```

### Backtester

```bash
python backtest.py                        # validate current alerts
python backtest.py NVDA AAPL TSLA        # specific tickers
python backtest.py --from-journal        # all journal tickers
python backtest.py --period 2y           # extended lookback
```

### Self-Reflection Loop

```bash
python reflect.py report                  # this week's performance
python reflect.py patterns               # accumulated miss patterns
python reflect.py amendments             # pending framework drafts
python reflect.py apply kpak82          # merge draft → framework-v2.md
python reflect.py reject kpak82 --date 2026-06-07 --reason "one-off event"
python reflect.py --auto                 # full run (same as Friday launchd)
```

## LLM Providers

| Provider | Config | Cost |
|---|---|---|
| Mammouth AI (Claude) | `provider: mammouth`, key from [mammouth.ai](https://mammouth.ai) | Low |
| Anthropic (Claude) | `provider: anthropic`, key from console.anthropic.com | Moderate |
| Ollama (local) | `provider: ollama`, install from [ollama.ai](https://ollama.ai) | Free |
| Heuristic fallback | `--no-llm` flag or no key configured | Free |

## Data files (not committed)

| Path | Contents |
|---|---|
| `config.json` | API keys (gitignored — see template) |
| `data/all_data.json` | Latest scan results |
| `data/alerts.json` | Current trade alerts |
| `data/trade_journal.jsonl` | Append-only trade log |
| `data/reflect_history.jsonl` | Weekly reflection ledger |
| `data/archive/scored-*.json` | Per-scan full scored output (14-day retention) |
| `creators/*/posts_raw.txt` | Scraped X posts (regenerated via distill.py) |
| `creators/*/framework-v*.md` | Distilled creator trading frameworks |

## Running tests

```bash
pytest tests/ -q
```

## Disclaimer

This system is for **educational and research purposes only**. It does not constitute financial advice. All trades should be verified independently. Options trading involves substantial risk of loss.
