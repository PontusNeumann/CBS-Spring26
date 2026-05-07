# Feature Selection — Decisions and Audit Trail

KAN-CDSCO2004U Machine Learning and Deep Learning · Polymarket mispricing detection

This document records, for the submission's modeling stage, every column from `consolidated_modeling_data.parquet` and whether it was kept as a feature or excluded under one of the foundation-doc decisions (P0-2, P0-8, P0-9, P0-11, P0-12) or under the original v5 causality / future-looking exclusion set.

## Summary

- Modeling parquet ships **87 columns** total (5 meta, 1 target, 1 row-id, 80 candidate features).
- After exclusions: **43 features** kept for `X`.
- **41 columns** explicitly excluded by `FORBIDDEN_LEAKY_COLS` in `01_data_prep.py`.

The exclusion principle mirrors the v5 keep-list documented in `archive/foundation.md`: drop everything that (a) leaks the future, (b) jointly determines the target via a sign-flipping formula, (c) encodes direction in a way that the train/test cohort split cannot generalize, or (d) carries absolute scale that lets the model identify which market it is in.

## Kept features (43)

Grouped by source layer.

### Trade-Local  (1)

| Feature | Description |
|---|---|
| `log_size` | log of trade USD; archive Layer 1 keep |

### Interaction  (3)

| Feature | Description |
|---|---|
| `log_size_vs_taker_avg` | log size relative to taker's recent average |
| `trade_size_to_recent_volume_ratio` | trade size relative to market recent volume |
| `trade_size_vs_recent_avg` | trade size relative to recent average |

### Market Microstructure  (7)

| Feature | Description |
|---|---|
| `avg_trade_size_recent_1h` | average trade size in last 1h |
| `log_recent_volume_1h` | log recent volume, 1h window |
| `log_recent_volume_24h` | log recent volume, 24h window |
| `log_recent_volume_5min` | log recent volume, 5-min window |
| `log_trade_count_1h` | log recent trade count, 1h |
| `log_trade_count_24h` | log recent trade count, 24h |
| `log_trade_count_5min` | log recent trade count, 5min |

### Market Context (Normalised)  (4)

| Feature | Description |
|---|---|
| `implied_variance` | model-free implied variance |
| `jump_component_1h` | jump component decomposition (Bardorff-Nielsen-Shephard) |
| `market_price_vol_last_1h` | 1h realised price vol; archive Layer 2 keep |
| `realized_vol_1h` | 1h realised vol |

### Market History  (1)

| Feature | Description |
|---|---|
| `log_n_trades_to_date` | log count of trades in this market so far |

### Market Timing  (1)

| Feature | Description |
|---|---|
| `log_time_since_last_trade` | log seconds since previous trade in this market |

### Time (Relative)  (1)

| Feature | Description |
|---|---|
| `pct_time_elapsed` | fraction of [creation, deadline] elapsed at trade time; archive Layer 3 keep |

### Time (Cyclic)  (3)

| Feature | Description |
|---|---|
| `day_of_week_cos` | cyclic encoding |
| `day_of_week_sin` | cyclic encoding |
| `hour_of_day_sin` | cyclic encoding |

### Taker History  (5)

| Feature | Description |
|---|---|
| `log_taker_cumvol_in_market` | log cumulative volume by this taker in this market |
| `log_taker_first_minutes_ago_in_market` | log minutes since taker's first trade in this market |
| `log_taker_prior_trades_in_market` | log prior trade count in this market |
| `log_taker_prior_trades_total` | log lifetime trade count (prior to t) |
| `log_taker_prior_volume_total_usd` | log lifetime traded USD (prior to t) |

### Taker Behaviour  (1)

| Feature | Description |
|---|---|
| `log_taker_burst_5min` | log taker activity in last 5min |

### Taker Indicator  (2)

| Feature | Description |
|---|---|
| `taker_first_trade_in_market` | binary: taker's first trade in this market |
| `taker_traded_in_event_id_before` | binary: taker traded the same event before |

### Counterparty  (1)

| Feature | Description |
|---|---|
| `log_maker_prior_trades_in_market` | log prior trade count of the maker (counterparty) |

### Cross-Market Diversity  (1)

| Feature | Description |
|---|---|
| `log_taker_unique_markets_traded` | log unique markets traded prior to t |

### Wallet On-Chain  (10)

| Feature | Description |
|---|---|
| `days_from_first_usdc_to_t` | days from wallet's first USDC inflow to t |
| `wallet_cex_usdc_cumulative_at_t` | cumulative CEX-funded USDC at t |
| `wallet_funded_by_cex_scoped` | 1 iff wallet's first CEX inflow was strictly before t (causal variant) |
| `wallet_log_cex_usdc_cum` | log cumulative CEX-funded USDC at t |
| `wallet_log_n_inbound_at_t` | log inbound transfers count at t |
| `wallet_log_polygon_nonce_at_t` | log Polygon nonce at t |
| `wallet_n_cex_deposits_at_t` | count of CEX-funded inflows at t |
| `wallet_n_inbound_at_t` | inbound transfers count at t |
| `wallet_polygon_age_at_t_days` | wallet's Polygon address age in days at t |
| `wallet_polygon_nonce_at_t` | Polygon nonce at t (transactions sent so far) |

### Indicator  (1)

| Feature | Description |
|---|---|
| `wallet_enriched` | 1 if wallet enrichment fetched successfully (else features default to 0) |

## Excluded columns (41)

Grouped by decision.

### causality  (3)

| Column | Why excluded |
|---|---|
| `kyle_lambda_market_static` | fit on first half of each market then broadcast back — uses future trades for the early-half rows |
| `n_tokentx` | lifetime tx count — peeks past trade time |
| `wallet_funded_by_cex` | lifetime flag, true if wallet ever got a CEX deposit, even after t |

### P0-9 causality  (1)

| Column | Why excluded |
|---|---|
| `wallet_prior_win_rate` | naive version that pools across all priors regardless of resolution timing — carries +0.13 leak-driven correlation with target |

### P0-11 direction-encoding pair  (2)

| Column | Why excluded |
|---|---|
| `outcome_yes` | same as side_buy; together they reconstruct bet_correct deterministically per market |
| `side_buy` | jointly determines bet_correct via XOR formula whose mapping flips across market resolution types — catastrophic test-set inversion on single-resolution cohorts (test set is all-NO ceasefires) |

### P0-12 direction-encoding aggregate  (16)

| Column | Why excluded |
|---|---|
| `consensus_strength` | directional aggregate of consensus formation |
| `contrarian_score` | signed contrarian metric |
| `contrarian_strength` | magnitude of contrarian deviation |
| `is_long_shot_buy` | binary direction flag |
| `market_buy_share_running` | running share of BUY trades — direction-encoding aggregate with 30-pt train-vs-test cohort shift |
| `order_flow_imbalance_1h` | signed order-flow imbalance, 1h |
| `order_flow_imbalance_24h` | signed order-flow imbalance, 24h |
| `order_flow_imbalance_5min` | signed order-flow imbalance, 5min |
| `signed_oi_autocorr_1h` | signed open-interest autocorrelation |
| `taker_directional_purity_in_market` | fraction of taker's trades on one side |
| `taker_position_size_before_trade` | signed position size leaks taker's directional view |
| `taker_yes_share_global` | global YES share for the taker |
| `token_side_skew_5min` | directional token-side skew |
| `yes_buy_pressure_5min` | directional buy pressure on YES side |
| `yes_volume_share_recent_1h` | share of recent volume on YES side |
| `yes_volume_share_recent_5min` | 5-min YES volume share |

### P0-8 absolute scalar  (6)

| Column | Why excluded |
|---|---|
| `is_within_1h_of_deadline` | same |
| `is_within_24h_of_deadline` | same |
| `is_within_5min_of_deadline` | binary deadline-proximity flag, derived from the absolute time scalar |
| `log_time_to_deadline_hours` | absolute time-to-deadline in log hours; scales with each market's deadline distance and so leaks market identity |
| `market_price_vol_last_24h` | absolute-window vol; archive keeps only 1h normalised vol |
| `market_price_vol_last_5min` | absolute-window vol; archive keeps only 1h normalised vol |

### P0-8 absolute price level  (12)

| Column | Why excluded |
|---|---|
| `distance_from_boundary` | min(price, 1-price) — derived from absolute price |
| `log_payoff_if_correct` | log(1/cost) — derived from price and direction-aware |
| `pre_trade_price_change_1h` | signed price change |
| `pre_trade_price_change_24h` | signed price change |
| `pre_trade_price_change_5min` | signed price change, leaks both level and direction |
| `recent_price_high_1h` | absolute price extremum, leaks market price benchmark |
| `recent_price_low_1h` | absolute price extremum |
| `recent_price_mean_1h` | absolute mean price, 1h |
| `recent_price_mean_24h` | absolute mean price, 24h |
| `recent_price_mean_5min` | absolute mean price, 5min |
| `recent_price_range_1h` | absolute high-minus-low range |
| `risk_reward_ratio_pre` | cost/(1-cost) ratio derived from absolute price |

### market benchmark  (1)

| Column | Why excluded |
|---|---|
| `pre_trade_price` | explicitly excluded from features so p_hat is independent of the market's own belief; retained in the parquet only as the trading-rule benchmark for the backtest |

## References

- `archive/foundation.md` — full provenance of decisions P0-2, P0-8, P0-9, P0-11, P0-12.
- `submission/scripts/01_data_prep.py` — `FORBIDDEN_LEAKY_COLS` set and the eight checks that enforce it (S1, S2, C1, F3, D1, D4, W1, B1).
- Pre-exclusion permutation importance attributed 0.385 of test AUC drop to `outcome_yes` and 0.277 to `side_buy` (P0-11). Excluding both was the single largest driver of the post-fix model behaviour.
