# ML&DL Exam Project: Insider Trading Detection on Polymarket

## Handover Document
**Date:** 2026-04-18
**Status:** Research & data exploration complete. Data extraction pipeline partially built. Blocked on full trade data extraction.

---

## 1. Project Concept

**Research question:** Can we detect informed (insider) trading on Polymarket prediction markets using neural networks?

**Approach:**
- Pull all trades from resolved Iran geopolitical markets on Polymarket
- Each row = one trade. Target variable = was this bet correct (1/0), based on known market resolution
- Features = behavioural signals available at the time of the trade (no lookahead)
- Train an MLP to predict which bets are correct based on behavioural features
- Core insight: in an efficient market, behavioural features (wallet age, bet size, timing) should NOT predict correctness. Where they do, that's evidence of informed trading
- Compare against baselines: logistic regression, random forest, Isolation Forest
- Discussion: which features most predict correctness, and what does that imply about information leakage?

**Why this is interesting:**
- Novel: existing work (Columbia Law paper, pselamy/polymarket-insider-tracker) uses rule-based scoring. Nobody has applied ML/DL
- Documented insider cases exist for validation (Magamyman $553K, Iran ceasefire 50+ wallets)
- Fits course syllabus: MLP (L09), autoencoders (L11), anomaly detection (L07), class imbalance (L07), evaluation metrics (L02/L06)
- Ethics section writes itself: unregulated markets, information asymmetry, privacy of wallet data

---

## 2. Key References

### Academic
- **Mitts & Ofir (2026)** "From Iran to Taylor Swift: Informed Trading in Prediction Markets" (Columbia Law / SSRN)
  - SSRN: papers.ssrn.com/sol3/papers.cfm?abstract_id=6426778
  - Harvard blog summary: corpgov.law.harvard.edu/2026/03/25/from-iran-to-taylor-swift-informed-trading-in-prediction-markets/
  - 93K markets, 50K wallets, 210K flagged pairs, $143M anomalous profit
  - Five signals: cross-sectional bet size, within-trader bet size, profitability, pre-event timing, directional concentration
  - Key cases documented with amounts and timing (see section 5 below)
  - **Download the full paper via CBS SSRN access** for exact formulas

### Code / Tools
- **pselamy/polymarket-insider-tracker** (GitHub) -- rule-based detection. Useful for feature ideas. Not ML
- **warproxxx/poly_data** (GitHub) -- data scraper using Goldsky subgraph. Downloads ALL Polymarket trades globally. No per-market filter. Goldsky subgraph is currently stale (indexed only to Jan 5, 2026)
- **Polymarket API docs**: docs.polymarket.com

### News
- NPR: "Prediction market trader 'Magamyman' made $553,000 on death of Iran's supreme leader" (2026-03-01)
- NPR: "$300k Biden pardons trade" (2026-04-16)
- Bloomberg: "Polymarket implements new insider trading rules" (2026-03-23)

---

## 3. Data Sources

### Polymarket APIs (all free, no auth for reads)

| API | Base URL | What it gives | Limitation |
|-----|----------|---------------|------------|
| **Data API** | data-api.polymarket.com | Trades, positions, activity per wallet | **Hard cap ~3,100 trades per market** |
| **Gamma API** | gamma-api.polymarket.com | Market metadata, event info, token IDs | No trade data |
| **CLOB API** | clob.polymarket.com | Orderbook, prices. Trades need auth | 401 on trades endpoint without auth |

### Polymarket Data API -- tested endpoints

```
GET /trades?market={condition_id}&limit=100&offset={n}
```
- Returns trades for a market. Max 100 per request
- Offset pagination works up to ~3000 (then 400 error)
- `before`, `after`, `startTs` parameters exist but DO NOT reliably filter
- Always returns most recent trades first
- **This API cannot give us the full trade history for high-volume markets**

```
GET /activity?user={wallet_address}&limit=100&offset={n}
```
- Trade + yield history per wallet

```
GET /positions?user={wallet_address}&limit=100&sizeThreshold=0
```
- Current positions with: size, avgPrice, curPrice, initialValue, currentValue, cashPnl, percentPnl, totalBought, realizedPnl, percentRealizedPnl

### Polygonscan (RECOMMENDED for full trade data)
- CTF Exchange contract: `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`
- OrderFilled event topic0: `0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6`
- Free tier: 5 req/sec, 100K calls/day. Needs free API key (sign up at polygonscan.com)
- `getLogs` returns max 1,000 results per query, paginate by block range
- **Limitation:** `makerAssetId` is non-indexed (in data blob), so you must filter client-side by token ID after fetching
- Estimated extraction time: ~1 hour for one market at free tier rate limits

### Other free sources for enrichment

| Source | What it adds | Effort |
|--------|-------------|--------|
| **Polygonscan txlist** | Wallet age (first tx), tx count, gas patterns | Low |
| **GDELT** (api.gdeltproject.org) | News timestamps. Compute "minutes since last Iran news" | Low |
| **Moralis** (free tier) | Historical USDC balance, labeled tx types. 40K req/mo | Medium |

### SEC Form 3/4/5 data (for potential stock market transfer test)
- Location: `~/Downloads/2026q1_form345/` (Q1 2026, from EDGAR)
- 103K non-derivative transactions, 38K derivative transactions per quarter
- Data back to 2007 available from EDGAR
- Relevant if we extend the project to test whether Polymarket-learned patterns detect stock market insider trading

---

## 4. Target Markets

### Primary: "US strikes Iran by...?" (Event 114242)
- **Total event volume: $529M** across 31 sub-markets, all resolved
- Gamma API: `GET https://gamma-api.polymarket.com/events/114242`

**Key sub-market (the Magamyman market):**
- Question: "US strikes Iran by February 28, 2026?"
- Condition ID: `0x3488f31e6449f9803f99a8b5dd232c7ad883637f1c86e6953305a2ef19c77f20`
- YES token: `110790003121442365126855864076707686014650523258783405996925622264696084778807`
- NO token: `10832696757358093775468120009000761778513405247768868107262967513475277652998`
- Volume: $89.65M
- **Resolved YES** (outcome prices: YES=1, NO=0)
- Only 3,100 trades extractable via Data API (12.3% of volume)

**Other sub-markets in this event (all resolved, mix of outcomes):**

| Market | Condition ID | Volume | Resolved |
|--------|-------------|--------|----------|
| By Jan 11 | 0x843913ab... | $3M | NO |
| By Jan 14 | 0x64b14a09... | $14M | NO |
| By Jan 31 | 0xabb86b08... | $42M | NO |
| By Feb 20 | 0xe1c67f75... | $19M | NO |
| By Feb 28 | 0x3488f31e... | $90M | **YES** |
| By Mar 31 | 0x4b02efe5... | $22M | YES |
| By Jun 30 | 0x797d586a... | $9M | YES |

Full list of 31 condition IDs available via the Gamma API event endpoint above.

### Secondary: "Iran x Israel/US conflict ends by...?" (Event 236884)
- Total volume: $44.7M, 9 sub-markets (3 resolved NO, 6 active)
- Gamma API: `GET https://gamma-api.polymarket.com/events/236884`
- 3 resolved markets already extracted (see existing CSV)

### Additional Iran markets found (not yet explored)
- "Israel military action against Iran by...?" (Event 357509)
- "Which countries will conduct military action against Iran by April 30?"
- "Iran ceasefire talks?" variants
- "Will France, UK, or Germany strike Iran by June 30?"
- Full list available by searching Polymarket for "iran"

---

## 5. Documented Insider Trading Cases (from Columbia paper)

Use these for validation. If the model flags these wallets/trades, it's working.

| Case | Wallet/Name | Amount | Timing | Market |
|------|-------------|--------|--------|--------|
| Iran strike | "Magamyman" + 5 other new wallets | $553K profit (Magamyman alone) | 71 min before news, entry at 17% implied prob | US strikes Iran by Feb 28 |
| Iran ceasefire | 50+ brand-new accounts | $550K aggregate | Minutes before Trump ceasefire announcement | Iran ceasefire |
| Maduro removal | "Burdensome-Mix" | $485K from $38.5K (12.6x) | Hours before capture news | Maduro removal |
| Google Year in Search | Unnamed | $1M+ from $10K (100x) | Pre-announcement | Google search rankings |
| Taylor Swift engagement | "romanticpaul" | Undisclosed | Days before public announcement | Swift/Kelce engagement |
| Biden pardons | 2 linked accounts (shared Kraken wallet) | $316K from $64K | Final hours before pardons | Biden pardons |

---

## 6. Feature Set (all available at trade time, no lookahead)

### From trades CSV (no API calls needed)
- `market_volume_so_far` -- cumulative volume before this trade
- `market_trades_so_far` -- cumulative trade count
- `market_wallets_so_far` -- cumulative unique wallets
- `market_volume_last_1h` / `_last_24h` -- recent volume
- `market_trades_last_1h` -- recent trade intensity
- `market_avg_trade_size_so_far` -- running average
- `market_price_mean_1h` / `market_price_std_1h` -- recent price / volatility
- `market_buy_sell_ratio_1h` -- directional pressure
- `market_age_seconds` -- time since market creation
- `seconds_until_end_date` -- time to market close
- `pct_time_elapsed` -- position in market lifespan
- `size_vs_market_avg_so_far` -- this trade / running market avg
- `wallet_nth_trade_in_market` -- 1st, 2nd, 10th trade
- `wallet_cumulative_size_in_market` -- total invested so far
- `wallet_side_consistency_in_market` -- always same direction?
- `trade_value_usd` -- price * size

### From wallet history (Data API, 1 call per wallet)
- `wallet_prior_trades` / `wallet_prior_markets` -- experience level
- `wallet_market_diversity` -- entropy across market categories
- `wallet_is_new` -- first ever Polymarket trade
- `wallet_is_new_to_category` -- never traded geopolitics before
- `wallet_avg_bet_size` / `wallet_bet_size_std` / `wallet_max_bet_size`
- `wallet_days_since_last_trade` -- dormancy
- `wallet_trade_frequency` -- trades per day historically
- `wallet_directional_concentration` -- max(% BUY, % SELL)
- `wallet_win_rate` -- only from markets resolved BEFORE this trade
- `wallet_total_volume_usd` -- lifetime volume
- `size_vs_wallet_avg` / `size_vs_wallet_max` -- outsized for this person?

### From Polygonscan (1 call per wallet, free)
- `wallet_age_days` -- first on-chain tx to trade time
- `wallet_nonce` -- total on-chain transactions
- `wallet_usdc_balance` -- balance at trade time

### From GDELT (free, no key)
- `minutes_since_last_news` -- last Iran article before trade
- `news_count_last_1h` / `_last_24h` -- news activity

---

## 7. What's Been Built

### Scripts
- `scripts/extract_trades.py` -- extracts trades via Data API, writes to CSV
  - Works but capped at ~3,100 trades per market
  - Supports `--markets-only` (skip wallet enrichment) and `--resume`
  - Currently configured for 3 "conflict ends by" markets

### Data files
- `data/raw_trades.csv` -- 9,300 trades from 3 resolved "Iran conflict ends by" markets
  - All 3 resolved NO (outcome balance problem)
  - 2,997 unique wallets
  - Fields: market_label, condition_id, resolved_outcome, tx_hash, timestamp, wallet, name, pseudonym, side, outcome_token, price, size, asset_id, bet_correct

### Project structure
```
ml-exam-project/
  scripts/
    extract_trades.py
  data/
    raw_trades.csv
  outputs/
  handover.md          <-- you are here
```

---

## 8. Current Blocker

**The Polymarket Data API caps at ~3,100 trades per market.** For the Feb 28 market ($89.65M volume), that's only 12.3% of trades. We tested:

| Method | Result |
|--------|--------|
| Offset pagination (offset=0 to 3000) | Works, caps at 3,100 |
| `before`/`after` timestamp params | API ignores them |
| `startTs`/`endTs` params | API ignores them |
| Goldsky subgraph (warproxxx approach) | Stale, indexed only to Jan 5, 2026 |
| CLOB API `/trades` | 401 Unauthorized |

**Solution: Polygonscan event logs.**
- Contract: `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`
- Event: OrderFilled (topic0: `0xd0a08e8c...`)
- Free API key needed (sign up at polygonscan.com)
- Iterate by block range, filter client-side by YES/NO token ID
- ~1 hour for one market at free tier rate limits

---

## 9. Next Steps (in order)

1. **Get Polygonscan API key** -- free signup at polygonscan.com
2. **Write Polygonscan extraction script** -- query OrderFilled events by block range, filter by token IDs, parse amounts into price/size
3. **Extract full trade history** for the Feb 28 market (and ideally 3-5 more from the "US strikes Iran" event for outcome balance)
4. **Wallet enrichment** -- for each unique wallet, pull: trade history (Data API), age/nonce (Polygonscan), positions
5. **Feature engineering** -- compute all ~35 features from section 6
6. **GDELT enrichment** -- pull Iran news timestamps, compute news-to-trade lag
7. **Model training** -- MLP, with logistic regression / random forest / Isolation Forest baselines
8. **Validation** -- does the model flag Magamyman and other documented cases?
9. **Report writing** -- follows the prescribed template in exam.md

---

## 10. Exam Requirements (quick reference)

- Individual oral on written group product (2-4 members)
- Max 15 pages
- Custom dataset (few thousand rows, good number of columns)
- Must include: model complexity analysis vs baseline, ethical consideration, LLM usage disclosure
- Report template: Abstract (200 words) > Intro > Conceptual Framework > Methodology > Results > Ethics > Discussion > Conclusion
- Sample strong report: Face Mask Detection (22 pages, 4 models, Grad-CAM interpretability). Notes in course knowledge base
- Full exam details: `~/.claude/memories/uni/machine-learning-and-deep-learning/exam.md`
- Course knowledge base (all 16 lectures mapped): `~/.claude/memories/uni/machine-learning-and-deep-learning/`
