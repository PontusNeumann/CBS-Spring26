# Feature exclusion list — reference

*Canonical list of columns that must NOT be fed into models trained on `data/experiments/{train,val,test}.parquet`. Matches `NON_FEATURE_COLS` you should configure in any modelling script under `alex/scripts/`.*

The training parquet has 84 columns. After these exclusions, **~45 features** remain for modelling.

## Class 1 — Identifiers and raw metadata (22 cols)

Not features by construction — these identify rows/markets/wallets.

```
proxyWallet, asset, transactionHash, condition_id, conditionId,
source, title, slug_x, slug_y, icon, eventSlug, outcome,
name, pseudonym, bio, profileImage, profileImageOptimized,
question, end_date, winning_outcome_index, resolved, resolution_ts,
outcomes, is_yes
```

(`is_yes` is a per-market label derived from `winning_outcome_index`, not a feature.)

## Class 2 — Raw columns superseded by derived features (3 cols)

```
size            # replaced by log_size
price           # duplicates market_implied_prob
timestamp       # only used to construct the split column
```

## Class 3 — Filter / label / benchmark / split (4 cols)

```
settlement_minus_trade_sec    # filtering column (drop post-resolution)
bet_correct                   # THE TARGET, not a feature
market_implied_prob           # benchmark for trading rule, not a feature (§4)
split                         # legacy from old quantile split, unused
```

## Class 4 — Leakage mitigations (per `data-pipeline-issues.md`)

### P0-1 — `wallet_is_whale_in_market`

Threshold `p95_by_market` is computed from end-of-market wallet totals → future knowledge. Drop until upstream fix lands.

### P0-2 — `is_position_exit`

Denominator uses current trade size → fresh SELLs always flagged as exits. Drop.

### P0-9 — `wallet_prior_win_rate`

Temporal leak. Computed via `cumsum()` on `bet_correct` of prior trades — but `bet_correct` of a prior trade is only KNOWN once that trade's market has resolved, which may be AFTER the current trade's timestamp. Formula peeks at future outcomes.

Strongest linear correlate with target (+0.226) in training, likely inflated by the leak. Drop until correctly recomputed (per-row filter on `resolution_ts < current_timestamp`).

### P0-8 — Market-identifying absolute-scale features (v3 drops from PR #5)

These let the model memorise which sub-market each trade belongs to:

```
time_to_settlement_s, log_time_to_settlement,
market_volume_so_far_usd, market_vol_1h_log, market_vol_24h_log,
market_trade_count_so_far
```

**Plus the interaction feature that bleeds market-identity through the back door** (uses raw `time_to_settlement_s`):

```
size_x_time_to_settlement
```

Bounded / normalised substitutes are **retained** (safe across markets):
- `pct_time_elapsed` (0-1, market-normalised)
- `market_buy_share_running` (0-1)
- `market_price_vol_last_1h` (price-based, not scale)

## The full set (copy-paste into your script)

```python
NON_FEATURE_COLS = {
    # Class 1: identifiers / metadata
    "proxyWallet", "asset", "transactionHash", "condition_id", "conditionId",
    "source", "title", "slug_x", "slug_y", "icon", "eventSlug", "outcome",
    "name", "pseudonym", "bio", "profileImage", "profileImageOptimized",
    "question", "end_date", "winning_outcome_index", "resolved", "resolution_ts",
    "outcomes", "is_yes",
    # Class 2: raw cols superseded by derived features
    "size", "price", "timestamp",
    # Class 3: filter / label / benchmark / split
    "settlement_minus_trade_sec", "bet_correct", "market_implied_prob", "split",
    # Class 4a: leaky features (P0-1, P0-2 in issues log)
    "wallet_is_whale_in_market",
    "is_position_exit",
    # Class 4b: market-identifying absolute-scale (P0-8 in issues log — v3 drops)
    "time_to_settlement_s",
    "log_time_to_settlement",
    "market_volume_so_far_usd",
    "market_vol_1h_log",
    "market_vol_24h_log",
    "market_trade_count_so_far",
    # Interaction feature that uses raw time_to_settlement_s (same P0-8 reasoning)
    "size_x_time_to_settlement",
    # Temporal leak via bet_correct cumsum on prior trades (P0-9)
    "wallet_prior_win_rate",
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
| **Missingness indicators** | `wallet_has_prior_trades`, `wallet_has_prior_trades_in_market`, `wallet_has_cross_market_history`, `wallet_enriched` | ✓ |

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
