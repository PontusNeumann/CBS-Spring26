# Feature exclusion list — reference

*As of 22 Apr (late): the leaky / buggy / market-identity columns that used to live in this exclusion list have been **physically dropped** from `data/03_consolidated_dataset.csv` by `scripts/20_finalize_dataset.py`. The pre-drop snapshot at `data/03_consolidated_dataset.pre_dropped_variables.csv` preserves the audit trail. This file now covers only the remaining non-feature columns (IDs, labels, filter, benchmark) that live in the finalised CSV for reproducibility but must not be fed to a model.*

The finalised CSV has **54 columns**. After the exclusions below, **~38 features** remain for modelling.

## Class 1 — Identifiers and raw metadata (still in the CSV)

Not features by construction — these identify rows/markets/wallets or support the §4 filter and the trading-rule benchmark. Bloat metadata columns (`conditionId`, `title`, `slug_x`, `slug_y`, `icon`, `eventSlug`, `outcome`, `name`, `pseudonym`, `bio`, `profileImage`, `profileImageOptimized`, `outcomes`) have been **physically dropped** and are no longer in the CSV.

```
proxyWallet, asset, transactionHash, condition_id,
source, question,
end_date, resolution_ts, deadline_ts,
winning_outcome_index, resolved,
is_yes
```

(`is_yes` is a per-market label derived from `winning_outcome_index`, not a feature.)

## Class 2 — Raw columns superseded by derived features (3 cols)

```
size            # replaced by log_size
price           # duplicates market_implied_prob
timestamp       # only used to construct the split column and for deadline_ts joins
```

## Class 3 — Filter / label / benchmark (3 cols)

```
settlement_minus_trade_sec    # filtering column (drop post-resolution) — §4
bet_correct                   # THE TARGET, not a feature
market_implied_prob           # benchmark for trading rule, not a feature (§4)
```

`split` was dropped — cohort assignment now lives in `data/experiments/{train,val,test}.parquet`, not in the main CSV.

## Class 4 — Formerly leaky / buggy / market-identity (now physically dropped)

Kept here as historical record. If any of these reappears in a future rebuild, it is a regression — `scripts/20_finalize_dataset.py` would drop them again, but the correct fix is upstream in `01_polymarket_api.py`.

- **P0-1 `wallet_is_whale_in_market`**: was end-of-market p95; fixed in source to expanding causal p95, not dropped.
- **P0-2 `is_position_exit`, `is_position_flip`**: denominator bug; dropped pending source fix.
- **P0-8 market-identity absolute-scale**: `time_to_settlement_s`, `log_time_to_settlement`, `market_volume_so_far_usd`, `market_vol_1h_log`, `market_vol_24h_log`, `market_trade_count_so_far`, `size_x_time_to_settlement` — dropped.
- **P0-9 `wallet_prior_win_rate` (naive)**: future-info peek; dropped. Replaced by `wallet_prior_win_rate_causal` + `wallet_has_resolved_priors` (both kept in the CSV and both safe to feed).
- **`wallet_funded_by_cex` (static/unscoped)**: structurally leaky; dropped. `wallet_funded_by_cex_scoped` is retained.
- **P0-11 direction determinism**: `side`, `outcomeIndex` — dropped. Their pair perfectly determines `bet_correct` within any market and the mapping flips between YES- and NO-resolved markets, producing test-set inversion on single-resolution cohorts.
- **P0-12 indirect direction-dependent features**: `wallet_position_size_before_trade`, `trade_size_vs_position_pct`, `wallet_cumvol_same_side_last_10min`, `wallet_directional_purity_in_market`, `wallet_has_both_sides_in_market`, `market_buy_share_running` — dropped. Each re-opens the P0-11 channel through signed position, same-side filter, or outcomeIndex-share aggregate.

Bounded / normalised substitutes that replaced the P0-8 drops are in the CSV:
- `pct_time_elapsed` (0-1, market-normalised)
- `market_price_vol_last_1h` (price-based, not scale)

## The full set (copy-paste into your script)

```python
NON_FEATURE_COLS = {
    # Identifiers / metadata still in the CSV
    "proxyWallet", "asset", "transactionHash", "condition_id",
    "source", "question",
    "end_date", "resolution_ts", "deadline_ts",
    "winning_outcome_index", "resolved", "is_yes",
    # Raw cols superseded by derived features
    "size", "price", "timestamp",
    # Filter / label / benchmark
    "settlement_minus_trade_sec", "bet_correct", "market_implied_prob",
}
```

## What's left after all drops (the modelling feature set)

Approximately these groups — verify from the actual parquet when you write the driver:

| Group | Example features | No-lookahead? |
|---|---|---|
| **Layer 6 (on-chain identity)** | `wallet_polygon_age_at_t_days`, `wallet_funded_by_cex`, `wallet_cex_usdc_cumulative_at_t` (+ log variants) | ✓ per-trade bisect |
| **Layer 7 (cross-market)** | `wallet_market_category_entropy` | ✓ expanding entropy, strictly prior |
| **Market context (normalised)** | `pct_time_elapsed`, `market_buy_share_running`, `market_price_vol_last_1h`, `market_timing_known` | ✓ |
| **Wallet global** | `wallet_prior_trades`, `wallet_prior_volume_usd`, `wallet_prior_win_rate`, `wallet_first_minus_trade_sec` | ✓ |
| **Wallet-in-market bursting** | `wallet_trades_in_market_last_{1,10,60}min`, `wallet_is_burst`, `wallet_median_gap_in_market` | ✓ |
| **Wallet-in-market directionality** | `wallet_directional_purity_in_market`, `wallet_has_both_sides_in_market`, `wallet_spread_ratio` | ✓ |
| **Position awareness** | `wallet_position_size_before_trade`, `trade_size_vs_position_pct`, `is_position_flip` | ✓ |
| **Depth** | `wallet_prior_trades_in_market`, `wallet_cumvol_same_side_last_10min` | ✓ |
| **Sizing / interactions** | `log_size`, `size_vs_wallet_avg`, `size_x_time_to_settlement`, `size_vs_market_cumvol_pct`, `size_vs_market_avg` | ✓ |
| **Trade-local** | `side`, `outcomeIndex`, `trade_value_usd` | ✓ (per-trade) |
| **Wallet global (causal)** | `wallet_prior_win_rate_causal` (priors with `resolution_ts < t` only) | ✓ non-leaky |
| **Missingness indicators** | `wallet_has_prior_trades`, `wallet_has_prior_trades_in_market`, `wallet_has_cross_market_history`, `market_timing_known`, `wallet_enriched`, `wallet_has_resolved_priors` | ✓ |

## How to use in an MLP driver

```python
import pandas as pd

train = pd.read_parquet("../../data/experiments/train.parquet")
feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]
# ~45 columns, all numeric (after encoding `side` as 0/1)
X_train = train[feature_cols]
y_train = train["bet_correct"].astype(int)
```

The val and test parquets have the same columns — apply the same filter.

## When this list changes

Update both this file AND `data-pipeline-issues.md` if:
- New leaky feature is found → add to Class 4a, new P0-n entry.
- Upstream fix lands for a drop → can remove the exclusion + note in issues log.
- New feature is added to the pipeline → decide whether it joins the feature set or the exclusion list before first training run.
