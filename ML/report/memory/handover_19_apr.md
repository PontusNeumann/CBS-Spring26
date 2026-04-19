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

## 6. Update — 19 April 2026 evening session

### 6.1 Scope widened
`TARGET_EVENT_IDS` in `fetch_polymarket.py` now also includes **355299** ("Trump announces US x Iran ceasefire end by...?") and **357625** ("US x Iran ceasefire extended by...?"). Full refetch produced:

| Metric | Value |
|---|---|
| Rows | 346,898 |
| Markets (resolved) | 74 |
| Unique wallets | 73,839 |
| Date range | 2025-12-22 to 2026-04-19 |
| `bet_correct` rate | 0.518 |
| Markets capped near ~7k | 21 |

### 6.2 Offset cap — initially accepted, now being lifted via HuggingFace
Polymarket's Data API `/trades` has no time-filter parameter and hard-errors at `offset>=5000`. Side-splitting lifts the per-market ceiling to ~7,000. Twenty-one markets of the 74 resolved are truncated at this ceiling.

**Resolution (19 April, later in the evening):** the `SII-WANGZJ/Polymarket_data` HuggingFace mirror carries the complete on-chain trade history for 67 of the 74 resolved markets (events 114242 and 236884). A new script `scripts/build_iran_dataset.py` streams that mirror's `trades.parquet` (38.7 GB) over HTTPS via duckdb's httpfs extension, filters server-side to the target condition_ids, and writes only the matching rows to `data/trades_iran_subset.parquet` (~hundreds of MB). The disk-limited design matters: the full parquet does not fit on a laptop with 23 GB free. The remaining 7 ceasefire markets (events 355299, 357625) are post-HF-cutoff and fetched via the Polymarket Data API; they are all under 5k trades so the ~7k ceiling is not reached.

Once the hybrid build runs, the 21-capped-markets limitation is mitigated. The only residual constraint is the HF snapshot cutoff (2026-03-31) which applies to the 7 API-fetched ceasefire markets — unaffected because those markets fit under the API cap.

### 6.3 True resolution timestamp implemented
New `derive_resolution_timestamps()` in `fetch_polymarket.py`. Strategy: find the earliest timestamp at which the winning-outcome token's price first locks to ≥0.995 and never falls back below 0.9. CLOB history is the primary source; trade-execution prices are the fallback. Fallback matters — CLOB `/prices-history` returns empty for 66 of 67 already-resolved markets (the service drops history after resolution). Derived for 98.0% of rows. Used in `settlement_minus_trade_sec` with `end_date` as final fallback. Note: 16.5% of trades now show negative values (post-resolution close-outs). Filter `settlement_minus_trade_sec > 0` before predictive modelling.

### 6.4 Running features added to `trades_enriched.csv`
Market-level (per `condition_id`, strictly prior to each trade):
- `market_trade_count_so_far`
- `market_volume_so_far_usd`
- `market_price_vol_last_1h` (rolling 1-hour std of `market_implied_prob`, excludes current row)

Wallet-level (per `proxyWallet`, strictly prior):
- `wallet_prior_trades`
- `wallet_prior_volume_usd`
- `wallet_prior_win_rate` (mean of `bet_correct` over prior trades; NaN on first trade)

### 6.5 Skipped this session — still open
| # | Task | Why deferred |
|---|---|---|
| A | Polygonscan enrichment (true wallet age, USDC inflow) | Needs API key signup; user chose free-data-only |
| B | GDELT news-timing enrichment | Free API but needs design conversation on features (count, tone, window) |
| C | Trade-offset-cap resolution via subgraph | Full rewrite, separate project |
| D | Temporal train/val/test split | Needs decision on split boundaries |
| E | EDA notebook | Pending |
| F | MLP training and baselines | Pending |

## 7. Pre-modelling filtering reminder
1. Drop post-resolution trades: `settlement_minus_trade_sec > 0`.
2. Optional: drop markets with more than ~6,500 trades if the recency-bias risk is unacceptable for the modelled subset.
3. For `wallet_prior_win_rate`, decide whether to impute first-trade NaN (e.g., global mean of 0.518) or use `wallet_prior_trades == 0` as a categorical signal.

## 7b. Alex-adoption pass (19 April evening, second update)

Adopted the substantive improvements from `ML/report/mldp-project-overview.md` without narrowing scope:

- **Trade-timestamp temporal split.** `split` column added to `trades_enriched.csv` with train/val/test at quantiles 0.70/0.85 of the trade timestamp distribution. Current counts: 242,828 train / 52,035 val / 52,035 test. Replaces the settlement-date split.
- **Price deliberately excluded from feature set.** `market_implied_prob` stays in the CSV only as the trading-rule benchmark; it should not be fed to the MLP.
- **Expanded six-layer feature taxonomy.** 19 new columns added via `expand_features()` in `fetch_polymarket.py`:
  - Trade-local: `log_size`
  - Time: `time_to_settlement_s`, `log_time_to_settlement`, `pct_time_elapsed`
  - Wallet-in-market bursting: `wallet_trades_in_market_last_1min/10min/60min`, `wallet_is_burst`
  - Wallet-in-market directional: `wallet_directional_purity_in_market`, `wallet_has_both_sides_in_market`, `wallet_spread_ratio`
  - Wallet-in-market position-aware: `wallet_position_size_before_trade`, `trade_size_vs_position_pct`, `is_position_exit`, `is_position_flip`, `wallet_is_whale_in_market`
  - Interactions: `size_vs_wallet_avg`, `size_x_time_to_settlement`
- **Two-strategy trading rule.** Plan and docx updated: general +EV (edge > 0.02, flat $100) and home-run (edge > 0.20, time_to_settlement < 6h, price < 0.30, larger stake). Cutoff-date sweep over N in {14, 7, 3, 1} days.
- **Magamyman sanity check** added as an illustrative Discussion anchor (not a Results acceptance target).
- **Ethics rewritten** around the Coplan November 2025 quote versus the 23 March 2026 rule change, the documented cases, and the enforcement gap.

Deliberately not adopted from Alex:
- Narrowing scope to 7 sub-markets of event 114242 only. Kept our 74 markets across 4 events so the ceasefire events stay in.
- Polygonscan on-chain enrichment. Free-data-only scope still holds.
- HuggingFace data source migration. Flagged as a pending decision in Section 8 of the plan; would eliminate the ~7k cap but costs a 28 GB download and a fetcher rewrite.

Expected null rates on the new features (all explainable):
- `wallet_directional_purity_in_market`, `wallet_spread_ratio`: ~48% NaN on first trade per (wallet, market) — by definition.
- `size_vs_wallet_avg`: ~21% NaN on first trade per wallet — by definition.
- `pct_time_elapsed`: ~2% NaN on markets with missing life_total (no resolution_ts nor end_date). Impute with 1.0 or drop as modelling chooses.

## 8. Backup
Previous `data/*.csv` snapshot saved to `data/_backup_20260419/` before the refetch.
