# Feature review — 2026-04-29 release (v4)

*Pre-pipeline-run review of `pontus-modeling-data-2026-04-29` (87 cols / 1.37M rows). Compared against [`feature-exclusion-list.md`](feature-exclusion-list.md) and the [v4 pipeline contract](../v4_final_ml_pipeline/README.md). Source feature builders: [`alex/scripts/06b_engineer_features.py`](../scripts/06b_engineer_features.py) and [`pontus/scripts/build_walletjoined_features.py`](../../pontus/scripts/build_walletjoined_features.py).*

Verdict: **don't train as-is.** Three blockers, several smaller items.

---

## TL;DR

| # | Item | Severity | Action |
|---|---|---|---|
| 1 | `kyle_lambda_market_static` is **fit on each market's first half** and used on rows in that first half — peeks at future trades within the same market | **leak — drop** | drop column |
| 2 | `side_buy` + `outcome_yes` together encode `bet_correct` (P0-11 inversion). Reintroduced from v3.5 | **shortcut + cohort flip risk** | drop both, OR replace with `is_yes_bet = side_buy XNOR outcome_yes` (loses no info) and verify train/test parity |
| 3 | Static `wallet_funded_by_cex` is in the parquet despite the v4 contract saying "no static `wallet_funded_by_cex`" | **spec violation** | drop column at load time; only use `wallet_funded_by_cex_scoped` |
| 4 | P0-12 direction-dependent aggregates reintroduced under new names (`taker_yes_share_global`, `taker_directional_purity_in_market`, `yes_volume_share_recent_*`, `yes_buy_pressure_5min`, `token_side_skew_5min`, `taker_position_size_before_trade`, `is_long_shot_buy`, `contrarian_score`, `market_buy_share_running`) | **cohort flip risk** | run causality guard (Stage 2) before training; if it falsifies train→test, drop |
| 5 | P0-8 absolute-scale features back (`log_time_to_deadline_hours`, `log_n_trades_to_date`, `log_recent_volume_*`, `log_trade_count_*`, `avg_trade_size_recent_1h`) | **market-identity shortcut** | same — guard catches |
| 6 | `engineer()` calls `fillna(0)` at the end — wipes missingness on log-features and pre_trade_price | **modelling distortion** | if not refactoring, add an explicit `*_isna` indicator before fillna |
| 7 | Per-cohort engineering — train and test are processed in isolated `engineer()` calls, so a taker's "global" priors reset to 0 at test boundary | **feature semantics** | acknowledge; rename `*_total` → `*_in_cohort` if shipping |
| 8 | `derive_winning_token` falls back to deadline-vs-event heuristic when `outcome_prices` ≠ {1, 0} | **target correctness** | verify all 75 markets settle cleanly via `outcome_prices`; halt if any fall through |

---

## 1. `kyle_lambda_market_static` — definitional leak

`alex/scripts/06b_engineer_features.py:411-433`. For each market, OLS slope is fit on **the first half of that market's life**, and the same slope is broadcast to every trade in the market.

```python
for mid, g in df.groupby("market_id"):
    g_sorted = g.sort_values("ts_dt")
    cutoff_idx = len(g_sorted) // 2
    first_half = g_sorted.iloc[:cutoff_idx]
    ...
    lambdas[mid] = float(slope)
df["kyle_lambda_market_static"] = df["market_id"].map(lambdas)
```

For trades in the first half, the lambda value uses **other trades in that same first half that occur AFTER the trade's timestamp**. That is post-trade information at trade time. For trades in the second half it is causal but the "static" framing assumes the modeller knows the market's first-half slope at trade-time, which is true only after the midpoint.

Either drop the column, or refit causally as expanding-window slope at each trade (much more expensive — would need rolling OLS per market). Default action: drop.

---

## 2. `side_buy` + `outcome_yes` — P0-11 label determinism reintroduced

The bet-correctness function (`derive_bet_correct`, lines 119-129):

```python
correct = np.where(
    is_buy,
    (side == winning).astype(int),    # BUY correct iff token side matches winner
    (side != winning).astype(int),    # SELL correct iff token side ≠ winner
)
```

Equivalently: `bet_correct = (side_buy == 1) XNOR (outcome_yes == is_winning_token1)`.

In a single-resolution cohort (Iran-strike train markets nearly all settle one way; ceasefire test markets the other), `is_winning_token1` is a near-constant per cohort. So given `outcome_yes`, the label is determined by `side_buy`:
- Train (most markets resolve YES = token1 winner): `bet_correct ≈ side_buy XNOR outcome_yes`
- Test (resolution flips): inverted

A model that learns this on train will systematically invert on test. **This is exactly what P0-11 in [`feature-exclusion-list.md`](feature-exclusion-list.md) flagged** — and the column-level fix back then was to drop `side` and `outcomeIndex`. The renamed `side_buy` and `outcome_yes` reintroduce the same channel.

Two options:
1. Drop both columns from the feature set. Loses information.
2. Replace with a single derived column `is_yes_bet = side_buy XNOR outcome_yes` (= 1 iff this trade is a "long-YES economic position"). Carries the same info from each individual feature alone (zero) but no longer encodes the label given resolution.

Either way, must rerun causality guard.

---

## 3. Static `wallet_funded_by_cex` violates v4 contract

[`v4_final_ml_pipeline/README.md`](../v4_final_ml_pipeline/README.md) line 86 explicit: *"Same 70 v3.5 features + exactly 6 new wallet columns (no `n_tokentx`, no static `wallet_funded_by_cex`, no `wallet_prior_win_rate`)"*.

Released parquet has both `wallet_funded_by_cex` (static, lifetime flag) and `wallet_funded_by_cex_scoped` (causal). The static version is structurally leaky for any wallet whose first CEX-sourced USDC arrives after `t` (it reads 1 even though that fact is unobservable at `t`). Empirically zero rows trigger this in the current dataset, but defence-in-depth says drop.

Action: load parquet, drop `wallet_funded_by_cex`, keep `wallet_funded_by_cex_scoped`. Either patch in a load helper or update Stage 1 schema validation to reject it.

(Sidenote: the info.json lists 12 "wallet features", but several pairs are raw + log variants of the same underlying bisect — `wallet_polygon_nonce_at_t` / `wallet_log_polygon_nonce_at_t`, etc. Effectively six independent signals + log mirrors + one indicator + the static flag. The "6 new wallet columns" claim in the v4 contract is undercounted; the meaningful set after drops is closer to 11.)

---

## 4. P0-12 direction-dependent aggregates reintroduced under new names

The 2026-04-22 audit dropped a class of features whose values shift sign or share when the resolution flips between cohorts. Several are back in v3.5 with renames:

| Old name (dropped) | New name (in v4) | What it encodes |
|---|---|---|
| `wallet_directional_purity_in_market` | `taker_directional_purity_in_market` | share of prior trades on same `(side_buy, outcome_yes)` code |
| `wallet_position_size_before_trade` | `taker_position_size_before_trade` (tanh-bounded) | signed YES-equivalent position |
| `market_buy_share_running` | unchanged, kept | running BUY share by volume |
| (new) | `taker_yes_share_global` | running mean of `outcome_yes` per taker |
| (new) | `yes_volume_share_recent_5min/1h` | recent share of volume on YES-side token |
| (new) | `yes_buy_pressure_5min` | recent share of volume that's BUY of YES |
| (new) | `token_side_skew_5min` | `yes_share_5min − 0.5` |
| (new) | `is_long_shot_buy` | flag: BUY at price < 0.20 |
| (new) | `contrarian_score`, `contrarian_strength` | uses `(2*side_buy − 1) × (0.5 − pre_trade_price)` |

The bounded-and-tanh framing of `taker_position_size_before_trade` is better than the old raw signed-cumsum, but the sign still flips with cohort resolution. None of these are per-trade leaks — all are strictly causal — but each opens a cohort-flip channel. The mechanism is the same as P0-12: a feature that says "this wallet leans YES" predicts `bet_correct = 1` on train (where YES wins) and `bet_correct = 0` on test (where it doesn't).

Don't drop on suspicion. **Run [`pressure_tests/phase2_falsification.py`](../scripts/pressure_tests/phase2_falsification.py) before the sweep**: if train→test AUC collapses or inverts on the v4 feature set, this is why.

---

## 5. P0-8 absolute-scale features reintroduced

| Old name (dropped) | New name (in v4) | Issue |
|---|---|---|
| `time_to_settlement_s`, `log_time_to_settlement` | `log_time_to_deadline_hours` | Absolute scale — per-market deadline encoded |
| `market_trade_count_so_far` | `log_n_trades_to_date` | Absolute, market-identity |
| `market_vol_1h_log`, `market_vol_24h_log` | `log_recent_volume_{5min,1h,24h}` | Absolute USD |
| (new) | `log_trade_count_{5min,1h,24h}` | Absolute count |
| (new) | `avg_trade_size_recent_1h` | Absolute USD |

`pct_time_elapsed` is the bounded substitute and is in the set, good. Bounded ratios (`trade_size_to_recent_volume_ratio`, `trade_size_vs_recent_avg`) are fine. The absolute-scale columns identify markets and let a tree memorise: "log_time_to_deadline ∈ [k, k+ε] ∧ market_id-implied scale → this is the Feb 28 strike market". Test markets have different deadlines so the memorisation does not transfer; same shortcut, different rotation.

Same recommendation: causality guard catches it. If guard passes, keep; if it falsifies, drop the absolute-scale family and keep only bounded/normalised ratios.

---

## 6. `engineer()` ends with `df.fillna(0)`

Line 854: `out = out.fillna(0)`.

This collapses missingness signals across all 70 features:
- `pre_trade_price` defaults to `0.5` *before* fillna (line 296), so 0 is wrong default — 0.5 is the agnostic prior. After fillna, any subsequent NaN would become 0 (= certainty market is no), distorting downstream features built from it.
- `log_size_vs_taker_avg`: NaN when wallet has no prior trade. After fillna, "no prior trade" reads identical to "log ratio = 0" (= same as average), which is a meaningful but different state.
- All `*_recent_*` rolling features reset to 0 on the wallet's/market's first row. 0 means "no prior activity" but in log-space it means `log(1) = 0` = exactly 1 prior unit. Reading is ambiguous.

For trees with missing-value handling this is OK-ish but information-lossy. For the MLP and LogReg in the sweep it can shift coefficients meaningfully.

Cheapest fix: at the top of `engineer()`, snapshot a `*_isna` indicator for the price/wallet-size/recent-volume features before the final fillna; emit them alongside. More expensive: refactor to explicit imputation per family.

---

## 7. Per-cohort engineering — "global" features reset at test boundary

`main()` (lines 859-889) calls `engineer()` once per cohort. Every running feature (`log_taker_prior_trades_total`, `log_taker_prior_volume_total_usd`, `log_taker_unique_markets_traded`, `taker_yes_share_global`, `log_size_vs_taker_avg`) is computed over only that cohort's rows, sorted by timestamp.

Consequence: a taker who made 200 trades in train markets, then their first test trade, sees `taker_prior_trades_total = 0` at that test row. The "global" name is misleading.

This is **not** a leak (test rows can't see train rows is the safe direction). But it strips wallet history at the cohort boundary, so `taker_yes_share_global = 0.5` (default) for many test rows that should have a meaningful prior. Either:
- Concatenate train + test, sort globally, compute running features, then split. Risk: a test row's running mean uses the train cohort's history. That's allowed (train is older) — but verify timestamps actually order this way.
- Or rename the columns `*_in_cohort` to be honest about scope.

Cheap path: rename. Right path: cross-cohort compute with a no-lookahead assert.

---

## 8. Target derivation falls back to deadline heuristic

`derive_winning_token` (lines 80-116):

```python
if abs(p1 - 1.0) < 0.01 and abs(p2 - 0.0) < 0.01:
    winning.append("token1"); continue
if abs(p1 - 0.0) < 0.01 and abs(p2 - 1.0) < 0.01:
    winning.append("token2"); continue
# else fall through to:
if cohort == "train":
    winning.append("token1" if deadline >= STRIKE_EVENT_UTC else "token2")
elif cohort == "test":
    winning.append("token1" if deadline >= CEASEFIRE_ANNOUNCEMENT_UTC else "token2")
```

Markets settling cleanly via `outcome_prices` ∈ {0, 1} are correctly labelled. Anything else falls through to a hardcoded "after the event = YES, before = NO" heuristic. If a market settled NO with deadline after the event date, the heuristic mislabels it.

Action before training: print the count of markets that hit each branch. Verify the fallback set is empty or consistent. Add an explicit halt if not.

---

## Recommended pre-flight (15 minutes)

Before invoking Stage 4 (`03_sweep.py`), run a short script that:

1. Loads `consolidated_modeling_data.parquet`
2. Drops `wallet_funded_by_cex` (static), `kyle_lambda_market_static`, and either drops or recodes `side_buy` + `outcome_yes` to the XNOR collapse
3. Asserts `derive_winning_token` produced no fallback rows (re-derive from `markets_subset.parquet` and compare)
4. Prints train/test parity on `bet_correct` mean (should both be ~50.3%)
5. Runs phase2 falsification on the residual feature set
6. Saves a clean `consolidated_modeling_data_clean.parquet` with the post-drop schema

Then Stage 1 schema validation has a clean target to validate.

---

## What is fine

- All 12 wallet enrichment columns from `build_walletjoined_features.py` use `np.searchsorted` against per-wallet lifetime arrays — strictly bisected at trade time. Confirmed against [Pontus wallet features review](../../../../Downloads/pontus-wallet-features-review.md). The only structural concern is the static `wallet_funded_by_cex` flagged above.
- Bounded ratios: `pct_time_elapsed`, `trade_size_to_recent_volume_ratio`, `trade_size_vs_recent_avg`, `log_size_vs_taker_avg`, `consensus_strength`, `distance_from_boundary`, `implied_variance`, `risk_reward_ratio_pre`, `log_payoff_if_correct` — all bounded or wallet-relative, not market-identifying.
- Pre-trade-only price family: `pre_trade_price` (lag-1) and `recent_price_*` (closed='left' rolling) — no current-trade peek.
- Time / cyclical: `hour_of_day_sin`, `day_of_week_{sin,cos}`, `is_within_*` deadline flags — clean.
- Microstructure: `realized_vol_1h`, `jump_component_1h`, `signed_oi_autocorr_1h` — closed='left', causal.
- On-chain: `log_same_block_trade_count` — per-block count is contemporaneous but block-level (not single-trade). Fine.

---

## What's missing relative to v4 spec

- `wallet_prior_win_rate_causal` — not present. Marked as v5 stretch in v4 README; expected absence.
- The "exactly 6 new wallet columns" line in the v4 contract is loose vs the released 12. Either tighten the contract to enumerate the 12, or rebuild the parquet to match the contract. Don't paper over the gap silently.
