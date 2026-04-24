# Feature inventory — full definitional audit

*Every column ever present in `03_consolidated_dataset.csv` (including the
25 physically dropped by `scripts/20_finalize_dataset.py`), with its
definition, formula, and whether the value is knowable at the trade's
`timestamp` from strictly-prior information only.*

The purpose of this table is to catch **definitional** leakage — features
whose formula references information that would not be observable in a
live deployment at the trade's timestamp, independent of whether empirical
tests caught it.

Status legend:
- **kept** — present in the finalised 54-col CSV
- **drop1** — Tier 1, confirmed leak / bug
- **drop2** — Tier 2, market-identifying absolute-scale (P0-8)
- **drop3** — Tier 3, metadata bloat
- **drop4** — Tier 4, obsolete
- **drop5** — Tier 5, direction encoding (P0-11 / P0-12)

Causality legend:
- **✓** — formula uses only information with timestamp `< t` or time-
  invariant facts published before `t` (e.g. deadline in question text).
- **±** — technically causal but the formula embeds market-identifying
  information that could be a back-door shortcut.
- **✗** — formula references information unavailable at `t`.

---

## A. IDs and trade primitives (always known at t by definition)

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `proxyWallet` | kept | 0x-prefixed 40-char Polygon address of the taker who initiated the trade. Polymarket settles via smart-wallet proxies; this is the on-chain taker. | ✓ |
| `asset` | kept | 77-digit decimal ERC-1155 token id for the outcome token being traded. One token id per (market, outcomeIndex) pair. | ✓ |
| `transactionHash` | kept | Polygon transaction hash of the order fill. Unique per (trade) event. | ✓ |
| `condition_id` | kept | Polymarket market id (0x-prefixed 64-char hash). Joins trade to market metadata. | ✓ |
| `conditionId` | drop3 | Duplicate of `condition_id` (camelCase variant Polymarket returns on some API paths). Same string. | ✓ |
| `source` | kept | `"hf"` (HuggingFace mirror path) or `"api"` (Polymarket Data API path). Data-provenance flag. | ✓ |
| `timestamp` | kept | Trade execution time (UTC, `datetime64[ns, UTC]` on parquet, ISO on CSV). | ✓ |
| `size` | kept | Trade size in token units (shares). `BUY` = shares acquired; `SELL` = shares released. | ✓ |
| `price` | kept | Trade execution price ∈ [0, 1]. For a binary outcome token this is the implied probability of that token paying out 1.0 at resolution. | ✓ |
| `outcomeIndex` | drop5 | 0 or 1 — which of the two outcome tokens the trade is on. **Forms P0-11 with side.** | ✓* |
| `side` | drop5 | `"BUY"` or `"SELL"`. **Forms P0-11 with outcomeIndex:** `bet_correct = (outcomeIndex == winning_outcome_index) == (side == "BUY")`, so the pair deterministically encodes the label conditional on `winning_outcome_index`. | ✓* |

`outcomeIndex` / `side` are individually known at trade time (✓), but in
combination they leak the target given `winning_outcome_index` — which
flips per-market between YES- and NO-resolved cohorts (P0-11). Dropped
from the feature set to prevent cross-resolution inversion.

---

## B. Market metadata and labels

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `question` | kept | Market's human-readable question string (e.g. "US strikes Iran by February 27, 2026?"). Known at market creation; parsed for `deadline_ts`. | ✓ |
| `title` | drop3 | Duplicate of the event-level title. Polymarket exposes this as a separate API field. | ✓ |
| `slug_x`, `slug_y` | drop3 | URL slug of the market (two variants from overlapping API joins). | ✓ |
| `icon`, `profileImage`, `profileImageOptimized` | drop3 | Image URLs (market icon, user avatar). Irrelevant for modelling. | ✓ |
| `eventSlug` | drop3 | URL slug of the parent event. | ✓ |
| `name`, `pseudonym`, `bio` | drop3 | User-profile fields from the API join. Self-reported, only populated for API-path rows. | ✓ |
| `outcome` | drop3 | API-path only: 1 if the trade is on the YES-side outcomeIndex. Ambiguously named (not the resolved outcome); dropped to avoid confusion with `is_yes`. | ✓ |
| `outcomes` | drop3 | Raw per-market outcomes array, e.g. `"Yes;No"` or `"No;Yes"`. Used to derive `is_yes`; redundant afterwards. | ✓ |
| `end_date` | kept | Gamma API `endDate` field — market's advertised deadline at the time of the API snapshot. **Stale on 34 of 74 markets** (returns placeholder `2026-01-31`), so feature code uses `deadline_ts` parsed from `question` instead. Retained as metadata only. | ✓ (published at creation) |
| `resolution_ts` | kept | Derived: first timestamp at which the winning token's price locks ≥ 0.995 and stays ≥ 0.9 thereafter. **Post-hoc**: observable only after the market finalises, so it is *not* a feature. Used only in the §4 settlement filter. | ✗ (post-hoc) |
| `deadline_ts` | kept | Parsed from `question` via regex: `"by <Month> <Day>, <Year>"`; fallback to `end_date.year` for year-less patterns. Published at market creation, knowable at any subsequent `t`. | ✓ |
| `winning_outcome_index` | kept | 0 or 1 — which outcomeIndex resolved YES. Determined by market settlement. **Target-side**, not a feature. | ✗ (post-hoc) |
| `resolved` | kept | Boolean, market reached settlement. Always True in this cohort by construction (we only pulled resolved markets). Not a feature. | ✗ (post-hoc) |
| `is_yes` | kept | `1` iff `outcomes[winning_outcome_index] == "Yes"`. Per-market YES/NO label derived from the outcomes array. Not a feature. | ✗ (post-hoc) |
| `bet_correct` | kept | **TARGET.** `(outcomeIndex == winning_outcome_index) == (side == "BUY")`. 1 iff the trade ends up on the winning side. | ✗ (post-hoc — this IS the label) |
| `market_implied_prob` | kept | The market's implied probability at trade time. Computed as a `merge_asof(..., direction="backward", allow_exact_matches=False)` against CLOB mid-price history for the trade's `asset`; falls back to `price` (the trade's own execution price) when CLOB history is missing. HF-path markets (67/74) have no CLOB history so they all fall back to `price`. **Benchmark for the trading rule, excluded from feature set** (§4). | ✓ (contemporaneous market state) |
| `settlement_minus_trade_sec` | kept | `resolution_ts − timestamp` in seconds (falls back to `end_date − timestamp` when `resolution_ts` is NaT). Negative for post-resolution trades. **Filter input** (§4 drops rows with this ≤ 0). Not a feature. | ✗ (uses post-hoc `resolution_ts`) |
| `split` | drop4 | Legacy trade-timestamp quantile assignment (0.70/0.85). Replaced by market-cohort parquets under `data/experiments/`. | ✓ (if re-computed) |

---

## C. Trade-local derived features

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `trade_value_usd` | kept | `size × price` — USD notional of this specific trade. | ✓ |
| `log_size` | kept | `log1p(max(0, size))`. | ✓ |

---

## D. Market running features (all per-market, timestamp-ordered)

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `market_trade_count_so_far` | drop2 | `groupby(condition_id).cumcount()` — trades in this market strictly before `t`. **Drop rationale (P0-8):** absolute scale scales with each market's deadline and total activity, letting a model memorise "this is the Feb 28 market" and shortcut to its resolution. | ✓* |
| `market_volume_so_far_usd` | drop2 | `tv.groupby(condition_id).cumsum() − tv` — USD volume in this market strictly before `t`. Same P0-8 memorisation risk. | ✓* |
| `market_vol_1h_log` | drop2 | `log1p(rolling_sum(tv, 1h, closed="left"))` on per-market trades. Absolute-scale, same P0-8 reasoning. | ✓* |
| `market_vol_24h_log` | drop2 | Same with 24h window. | ✓* |
| `market_price_vol_last_1h` | kept | `rolling_std(market_implied_prob, 1h, closed="left")` per market. Bounded (price ∈ [0, 1]) and therefore market-normalised, so retained as a market regime indicator. | ✓ |
| `market_buy_share_running` | drop5 | Share of BUY trades in the market strictly before `t`: `prior_buys / n_prior`. **Drop rationale (P0-12):** direction-dependent aggregate reopens the P0-11 inversion channel (0.38 train vs 0.67 test shift). | ✓* |

✓* = strictly causal mechanically, but embeds market-identity (P0-8) or
direction (P0-12) information that produces generalisation shortcuts.

---

## E. Wallet-global running features

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `wallet_prior_trades` | kept | `groupby(wallet).cumcount()` — trades by this wallet strictly before `t`. | ✓ |
| `wallet_prior_volume_usd` | kept | `tv.groupby(wallet).cumsum() − tv` — USD notional by this wallet strictly before `t`. | ✓ |
| `wallet_first_minus_trade_sec` | kept | `(first_trade_of_wallet − timestamp).dt.total_seconds()`. Always ≤ 0; equals 0 on the wallet's very first trade. `min(timestamps)` is always in the past, so not leaky. | ✓ |
| `wallet_prior_win_rate` | drop1 | Old: `cumsum(bet_correct)/cumsum(labeled) − current`. **Peeks future outcomes** — at time `t`, a prior trade's `bet_correct` is only observable once that trade's market has resolved. 83.6% of rows had at least one unresolved prior at trade time. Leak-driven r(target) = +0.13. | ✗ |
| `wallet_prior_win_rate_causal` | kept | Causal replacement: cumulative mean of `bet_correct` restricted to prior trades whose `resolution_ts < t`. Implemented via a per-wallet walk with `np.asarray(hist_rts) < ts[i]` mask. 40% NaN rate (wallets with no resolved priors yet). | ✓ |
| `wallet_has_resolved_priors` | kept | 1 iff at least one prior trade with `resolution_ts < t` exists. Matching missingness indicator for the causal win rate. | ✓ |

---

## F. Wallet-in-market bursting features

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `wallet_trades_in_market_last_1min` | kept | `rolling_count((wallet, condition_id), "60s", closed="left")`. Trade count by this wallet in this market in the prior minute. | ✓ |
| `wallet_trades_in_market_last_10min` | kept | Same with 600s window. | ✓ |
| `wallet_trades_in_market_last_60min` | kept | Same with 3600s window. | ✓ |
| `wallet_is_burst` | kept | `1` iff `wallet_trades_in_market_last_10min >= 3`. Captures the Mitts & Ofir "bursty informed flow" signature. | ✓ |
| `wallet_median_gap_in_market` | kept | `expanding().median().shift(1)` over consecutive-trade gaps per (wallet, market). Shift of 1 makes it strictly prior. | ✓ |

---

## G. Wallet-in-market directional features (all P0-12 except spread_ratio)

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `wallet_directional_purity_in_market` | drop5 | `p_0² + p_1²` where `p_i = cum_i / (cum_0 + cum_1)` and `cum_i` is the wallet's prior trades in this market on `outcomeIndex == i`. Simpson-style diversity on outcomeIndex distribution. P0-12: outcomeIndex share aggregate → indirectly direction-dependent. | ✓* |
| `wallet_has_both_sides_in_market` | drop5 | `1` iff both `cum_0 > 0` and `cum_1 > 0` strictly before `t`. Indicator on outcomeIndex distribution (P0-12). | ✓* |
| `wallet_spread_ratio` | kept | `min(cum_0, cum_1) / max(cum_0, cum_1)` strictly before `t`. Symmetric under swapping outcomes 0 and 1, so it cannot identify WHICH side the wallet concentrates on — only HOW concentrated. Retained. | ✓ |

---

## H. Wallet-in-market position-aware features (P0-12 except whale_flag)

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `wallet_position_size_before_trade` | drop5 | Signed cumsum per (wallet, condition_id, outcomeIndex): `cumsum(where(side=="BUY", +size, -size)) − signed_current`. Encodes direction (sign = net BUY/SELL). P0-12. | ✓* |
| `trade_size_vs_position_pct` | drop5 | `size / max(|pos_before|, size)`, clipped [0, 1]. Uses signed position → P0-12. | ✓* |
| `is_position_exit` | drop1 | `1` iff `side == "SELL"` AND `trade_size_vs_position_pct >= 0.9`. **Buggy (P0-2):** denominator uses the current trade's size, so a wallet's first-ever SELL always satisfies `ratio = size/size = 1.0` and fires even though the trade is opening a short, not exiting. Dropped. | ✓ (if formula were correct) |
| `is_position_flip` | drop1 | `1` iff sign change of signed position before vs. after the trade. Same signed-size family as `is_position_exit`, inherits the denominator assumption. Dropped pending verification. | ✓ (if correct) |
| `wallet_is_whale_in_market` | kept | **Causal expanding p95 (fixed upstream).** At each trade in market M at time t, the threshold is the 95th percentile of per-wallet running volumes using only trades with timestamp strictly less than t. 20-wallet warmup before the flag can fire. Implemented via `sortedcontainers.SortedList` for O(N log W). | ✓ |

---

## I. Wallet-in-market depth

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `wallet_prior_trades_in_market` | kept | `groupby([wallet, condition_id]).cumcount()` strictly before t. | ✓ |
| `wallet_cumvol_same_side_last_10min` | drop5 | `rolling_sum(tv, 10min, closed="left")` over groupby (wallet, condition_id, outcomeIndex). **Same-side filter** makes it direction-dependent → P0-12. | ✓* |

---

## J. Interactions

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `size_vs_wallet_avg` | kept | `tv / (wallet_prior_volume_usd / wallet_prior_trades)`. Trade size relative to the wallet's historical average trade size. Undefined on wallet's first trade (NaN). | ✓ |
| `size_x_time_to_settlement` | drop2 | `log_size × log_time_to_settlement`. Inherits market-identity leak via `log_time_to_settlement` (P0-8). | ✓* |
| `size_vs_market_cumvol_pct` | kept | `tv / market_volume_so_far_usd`. Trade size as fraction of the market's cumulative USD volume. Denominator was dropped from features (Tier 2) but this RATIO is bounded [0, ∞) and not market-identifying — retained. **Top permutation importance feature** at Δroc +0.049. | ✓ |
| `size_vs_market_avg` | kept | `tv / (market_volume_so_far_usd / market_trade_count_so_far)`. Trade size relative to market's average trade size. Bounded ratio. | ✓ |

---

## K. Time features (all from `deadline_ts`, not `resolution_ts`)

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `time_to_settlement_s` | drop2 | `deadline_ts − timestamp` in seconds. **Drop rationale (P0-8):** absolute scale varies per market. Used only in the trading rule at rule-time (not as a model feature). | ✓* |
| `log_time_to_settlement` | drop2 | `log1p(max(0, time_to_settlement_s))`. Same P0-8. | ✓* |
| `pct_time_elapsed` | kept | `(timestamp − market_start) / (deadline_ts − market_start)`, clipped [0, 1]. `market_start = groupby(condition_id).timestamp.transform("min")` — always in the past for any row. Bounded and market-normalised, so retained. | ✓ |
| `market_timing_known` | kept | Missingness indicator: 1 iff `pct_time_elapsed` is defined. | ✓ |

---

## L. Layer 6 — on-chain identity (all via bisect on Etherscan timestamp arrays)

All computed in `11_add_layer6.py` via `np.searchsorted(..., side="left")`
against per-wallet sorted timestamp arrays, so they use only events with
block timestamp strictly before the trade's `timestamp`.

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `wallet_enriched` | kept | 1 iff Etherscan V2 `tokentx` returned non-empty data for this wallet (i.e. the wallet has any ERC-20 history on Polygon). | ✓ (time-invariant presence) |
| `wallet_polygon_age_at_t_days` | kept | `max(0, timestamp − polygon_first_tx_ts) / 86400`. Age of the wallet on Polygon at trade time. | ✓ |
| `wallet_polygon_nonce_at_t` | kept | `np.searchsorted(outbound_ts, t_trade, side="left")`. Count of outbound ERC-20 transfers strictly before `t`. | ✓ |
| `wallet_log_polygon_nonce_at_t` | kept | `log1p(wallet_polygon_nonce_at_t)`. | ✓ |
| `wallet_n_inbound_at_t` | kept | Count of inbound ERC-20 transfers strictly before `t`. | ✓ |
| `wallet_log_n_inbound_at_t` | kept | `log1p(n_inbound)`. | ✓ |
| `wallet_n_cex_deposits_at_t` | kept | Count of inbound USDC transfers from a known CEX hot-wallet strictly before `t`. | ✓ |
| `wallet_cex_usdc_cumulative_at_t` | kept | Cumulative USD value of those CEX inflows strictly before `t`. | ✓ |
| `wallet_log_cex_usdc_cum` | kept | `log1p(cex_usdc_cumulative)`. | ✓ |
| `days_from_first_usdc_to_t` | kept | `(timestamp − first_usdc_inbound_ts) / 86400` iff `first_usdc_inbound_ts < t`; else NaN. | ✓ |
| `wallet_funded_by_cex` | drop1 | Lifetime flag: 1 iff the wallet's **first-ever** USDC inflow came from a known CEX. **Structurally leaky** — if the first CEX-sourced USDC arrives after `t`, the flag nonetheless shows 1 at `t`. 0 rows in the current dataset empirically exhibit this (every flagged wallet was CEX-funded before its first Iran trade), but dropped as defence-in-depth. | ✗ (in principle) |
| `wallet_funded_by_cex_scoped` | kept | Causal version: `wallet_funded_by_cex AND first_usdc_inbound_ts < timestamp`. Retained. | ✓ |

---

## M. Layer 7 — cross-market diversity

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `wallet_market_category_entropy` | kept | Shannon entropy (nats) over an 8-category distribution of the wallet's prior **distinct** markets across the full Polymarket universe, using the HF mirror `SII-WANGZJ/Polymarket_data`. Streamed compute in `10_wallet_category_entropy.py`: for each row (wallet, t), count distinct prior markets by coarse category (geopolitics, us_politics, crypto, macro_finance, sports, entertainment, tech_business, other), compute `H = −Σ p_i log p_i` where `p_i = n_i / Σ n_j`. Entropy is written BEFORE the current trade's market is added to the counts — strictly prior. NaN when the wallet has fewer than 2 distinct prior markets. | ✓ |
| `wallet_has_cross_market_history` | kept | 1 iff `wallet_market_category_entropy` is defined (i.e. wallet has ≥ 2 prior distinct markets AND is present in the HF mirror). | ✓ |

---

## N. Missingness indicators

| Variable | Status | Definition / formula | Causal |
|---|---|---|---|
| `wallet_has_prior_trades` | kept | 1 iff `wallet_prior_trades > 0`. | ✓ |
| `wallet_has_prior_trades_in_market` | kept | 1 iff `wallet_prior_trades_in_market > 0`. | ✓ |
| `wallet_has_cross_market_history` | kept | Duplicate row in M above. | ✓ |
| `market_timing_known` | kept | In K above. | ✓ |
| `wallet_enriched` | kept | In L above. | ✓ |
| `wallet_has_resolved_priors` | kept | In E above. | ✓ |

---

## Definitional-leakage audit: is any kept feature actually leaky?

After walking the table, the kept feature set is **definitionally clean**.
The only remaining ✗ columns in the CSV are `resolution_ts`,
`winning_outcome_index`, `resolved`, `is_yes`, `bet_correct`, and
`settlement_minus_trade_sec` — none are fed to models. `market_implied_prob`
is a contemporaneous state (merge_asof backward, exact-match disallowed)
and is excluded from the feature set anyway (it is the trading-rule
benchmark).

The ✓* cells (causally mechanical but with shortcut risk) are all drops.

One subtle point worth noting, not a leak but a scope-of-claim nuance:
`wallet_market_category_entropy` uses the HF mirror snapshot from
2026-03-31. In a live deployment the entropy must be computed from a
rolling index of Polymarket's full trade history up to `t`. The HF mirror
is a convenient source here because all our training data is already in
scope; live deployment needs its own pipeline (see the realism review
already in the `pontus_adventure.md` companion).
