# Trading Framework: @CryptoKaleo
Distilled: 2026-06-07
Version: v1
Asset focus: CRYPTO ONLY — Bitcoin ($BTC) primary, Ethereum ($ETH), Solana ($SOL), Sui ($SUI), Hyperliquid ($HYPE), NFTs, altcoin memes
Note: ⚠️ CRYPTO-ONLY CREATOR. This framework does NOT apply to US equity options. The scanner surfaces US stocks — CryptoKaleo's specific signals (BTC cycle timing, altcoin rotation, NFT floor prices) will never match scanner output. Extract only general TA methodology and risk management philosophy for limited use as a secondary scoring weight.

## Trading Personality
Perma-bull on crypto with high conviction in multi-year cycles. Community-oriented, positive tone ("GOOD MORNING BULLS", "Be more bullish"). Patient accumulator who treats dips as opportunities, not threats. Transparent about positions and reasoning. Founder of Wonky Stonks NFT project — has financial stake in NFT ecosystem recovery. Uses account as a journal (per his bio). Rarely posts shorts; when he does, shares liquidation price publicly.

## Market Conditions They Trade (Crypto Context)
- Post-halving accumulation zones (BTC halving cycle: expects new ATHs ~536 days post-halving)
- Dips into major support zones (HTF support — "low 70s" retest before resuming higher)
- Momentum continuation in outperforming assets: "Don't fade momentum"
- Extreme fear sentiment: "Times of greatest fear are times of greatest opportunity"
- Rotation into early-stage ecosystems (Monad, Hyperliquid) before they go mainstream

## Market Conditions They Avoid
- Chasing laggards after momentum has shifted: "Don't chase the next best thing because it hasn't pumped yet"
- Selling into fear/capitulation events (MSTR collapse, ZEC exploit, SOL outages — treats these as buys)
- Trading on noise vs. signal — consistently says to "zoom out" rather than react to weekly candles

## Setup Triggers (Crypto — Limited Transferability to US Stocks)
1. **HTF support retest** — Price pulls back to a higher timeframe support zone after a breakout; expected bounce
2. **Bull flag pattern** — Tight consolidation after a strong move up; "should see solid acceleration once we get a clean break of resistance"
3. **Ecosystem adoption catalyst** — New platform integration, regulatory approval, or chain upgrade that drives fundamental demand
4. **Sentiment extremes** — When "crypto twitter is dead" and influencers quit = buying opportunity (referenced 2019 analog)
5. **Momentum continuation** — If an asset is leading the market and outperforming, don't fade it

## Preferred Instruments & Timeframes
- **Primary:** Spot BTC accumulation on dips; no leverage for long-term holds
- **Options/Derivatives:** Occasionally uses crypto perp futures (Kalshi); shares liquidation price publicly. One documented short: BTC short with $68.6K liquidation price
- **NFTs:** Active trader/holder; tracks floor prices, bull flags on NFT charts
- **Timeframe:** Multi-month to multi-year for BTC; shorter term for altcoin/meme rotations
- **US equity options:** Not applicable — does not trade US stocks

## Entry Rules (Crypto — General TA Principles Only)
1. Enters dips at HTF support zones: "Gameplan: retest low 70s and run it back higher"
2. Accumulates gradually ("stack high conviction plays on the dip")
3. Does not try to catch the exact bottom — "just keep stacking"
4. For momentum plays: enters early in an ecosystem before mainstream FOMO
5. When shorting: uses defined liquidation price as hard stop — publicly accountable

## Exit Rules
- Long-term BTC: "Never sell your Bitcoin" — effectively no exit for core position
- Altcoins/memes: sells into momentum / strength (implied by "stack on dip, ride the run")
- Not documented: specific profit targets or percentage-based exits for trading positions
- For shorts: liquidation price is the hard stop

## Risk Management
- Core position (BTC): never sells — conviction hold through all drawdowns
- Tactical positions (altcoins, NFTs, perps): smaller size, higher risk tolerance
- Does NOT use high leverage for long-term crypto holds
- Explicit: positions himself as contrarian buyer when sentiment is most bearish
- Transparency: posts actual positions and liquidation prices publicly

## Red Flags — What They Avoid
- Leverage without defined liquidation level
- Fading strong momentum: "it's not worth it to fade $HYPE because it's leading the market"
- Selling Bitcoin: "Never sell your Bitcoin" — hard rule
- Chasing coins that already ran: "Don't go after the next best thing"

## Quality Gate — Sample Setups

**Setup 1 — BTC HTF Support Play:**
- Date: 2026-05-22
- Setup: BTC loses $76K support; Kaleo calls for dip to low $70s zone before bounce
- Direction: Accumulate BTC spot on dip
- Thesis: "Retest low 70s, run it back to 80s/90s, range for summer, then $100K+ in fall/winter"
- Outcome: Still playing out (posts from June 7 reference $60K test)
- Transferable to US stocks: ❌ (crypto cycle timing, not applicable)

**Setup 2 — Bull Flag Break:**
- Date: 2026-05-17
- Setup: $MON (Monad token) "bouncing off HTF support as expected"
- Direction: Long; expects "solid acceleration once we get a clean break of resistance"
- Transferable to US stocks: ✅ partially (bull flag + HTF support = standard TA, applicable to any asset)

**Setup 3 — Momentum Continuation (Don't Fade Leaders):**
- Date: 2026-05-19
- Setup: $HYPE leading the market and outperforming; sentiment says to rotate elsewhere
- Direction: Hold/add to HYPE rather than rotate
- Quote: "It's not worth it to fade momentum — chasing after the next best thing doesn't work"
- Transferable to US stocks: ✅ (momentum continuation principle — applies to stocks with high relative strength)

**Setup 4 — Sentiment Extreme as Buy Signal:**
- Date: 2026-05-30
- Setup: References 2019 crypto bear market when influencers quit; those who held made 20x
- Direction: Buy during peak fear
- Transferable to US stocks: ✅ partially (contrarian sentiment signal — extreme fear = opportunity)

## What Transfers to US Stock Scanner (Limited)

| Framework Element | Transferable? | Notes |
|---|---|---|
| Bull flag pattern recognition | ✅ Yes | Standard TA, works on any asset |
| HTF support retest entry | ✅ Yes | Standard TA |
| Don't fade momentum leaders | ✅ Yes | Relative strength concept |
| Sentiment extremes as buy | ✅ Partially | VIX extremes as analog |
| BTC halving cycle timing | ❌ No | Crypto-specific |
| Altcoin rotation signals | ❌ No | Crypto-specific |
| NFT floor analysis | ❌ No | Not applicable |
| "Never sell" hold strategy | ❌ No | Incompatible with options (time decay) |

## Honest Limitations
- **Does not trade US stocks or equity options** — this framework has zero direct signal value for the scanner
- **Permanently bullish bias** — will not provide bearish setups for put opportunities
- **No DTE, no strike selection, no options-specific framework** — never discusses equity options mechanics
- **TA principles are generic** — the transferable elements (bull flags, HTF support) are standard knowledge already built into the scanner
- **Recommended weight in orchestrate.py scoring: 0 or very low** — do not let this framework influence put/call direction for individual US stocks. Could use the "don't fade momentum" principle only for filtering out stocks in strong uptrends when looking for put setups.
