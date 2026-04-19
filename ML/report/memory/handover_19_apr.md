# Handover - 19 April 2026

**Session focus:** Data collection for RQ1 (MLP predicting `bet_correct` on Polymarket Iran markets).

## 1. What was built

`report/fetch_polymarket.py` — single-file fetcher for events 114242 and 236884, pulls from Polymarket's public APIs:

- Gamma API: event and market metadata, including `outcomePrices` used to derive `resolved` and `winning_outcome_index`.
- CLOB API: per-outcome-token price history via `/prices-history`.
- Data API: paginated trade history per market via `/trades`.

Outputs to `report/data/`:

- `markets.csv` — one row per market (74 total, 67 resolved).
- `prices.csv` — CLOB mid-price time series per outcome token.
- `trades.csv` — raw trade log.
- `trades_enriched.csv` — merged and feature-engineered training table.

## 2. Cap workaround

Polymarket Data API hard-caps pagination offset at ~3000. The fetcher now falls back to `side=BUY` and `side=SELL` pulls when the unfiltered pass hits the cap, dedupes on `transactionHash + asset + side + size + price + timestamp`. Lifts per-market ceiling from ~3,500 to ~10,500 trades.

## 3. Derived columns on `trades_enriched.csv`

| Column | Definition |
|---|---|
| `bet_correct` | `1` when `(outcomeIndex == winning_outcome_index) == (side == BUY)`. Target for RQ1. |
| `market_implied_prob` | Last CLOB mid-price for the trade's token strictly before trade timestamp (no-lookahead, via `pd.merge_asof`). Falls back to the trade's own `price` field when no prior sample exists. |
| `trade_value_usd` | `size * price`. |
| `settlement_minus_trade_sec` | `end_date - timestamp` in seconds. |
| `wallet_first_minus_trade_sec` | `wallet's first observed trade - current trade` in seconds. Proxy for wallet age within the dataset. |

## 4. Dataset snapshot

| Metric | Value |
|---|---|
| Rows | 332,258 |
| Markets | 67 resolved |
| Unique wallets | 70,470 |
| Date range | 2025-12-22 to 2026-04-03 |
| `bet_correct` rate | 0.518 |
| `market_implied_prob` coverage | 1.000 |
| Median `trade_value_usd` | $11.00 |
| Max `trade_value_usd` | $1,023,518 |

## 5. Known issues to address before modelling

### 5.1 Recency bias from offset cap
Date range is only 3.5 months. The Data API returns newest-first within the offset window, so for any market with more than ~10,500 trades the earliest trades are truncated. Long-lived markets under event 236884 (Iran x Israel/US conflict, resolved 2025) are missing their earliest activity. This matters because early informed trades carry the strongest pre-news signal. Fix: time-windowed pagination using a decreasing `before` timestamp per market, or Polygonscan on-chain extraction for capped markets.

### 5.2 `end_date` is not the true resolution timestamp
The Gamma `endDate` field is a scheduled end, not the actual resolution moment. Evidence: 25th percentile of `settlement_minus_trade_sec` is negative 14 days and median is near zero, meaning half the trades occur after `end_date`. Replace with the CLOB close time or the timestamp at which the token price first locks to 0 or 1 before using this column as a feature.

### 5.3 Wallet age is dataset-local
`wallet_first_minus_trade_sec` only captures wallet age within the pulled dataset. True on-chain wallet age requires Polygonscan and is still open per the project plan (Deliverable 3).

## 6. Next steps

| # | Task | Status |
|---|---|---|
| 1 | Time-windowed pagination for capped markets | open |
| 2 | Replace `end_date` with true resolution timestamp | open |
| 3 | Polygonscan enrichment for true wallet age and on-chain identity | open |
| 4 | GDELT news-timing enrichment (project plan Deliverable 4) | open |
| 5 | Behavioural running features (volume-so-far, trade-count-so-far, recent volatility) | open |
| 6 | Wallet running features (prior trades, prior volume, prior win rate) | open |
| 7 | Temporal train/val/test split by `end_date` quantiles | open |
| 8 | EDA notebook | open |
| 9 | MLP training and baselines | open |
