# Data Pipeline Issues — Log

*Created 2026-04-22 during pre-training audit. Each entry records a finding, the current mitigation applied to unblock training, and the upstream fix that should land after the deadline.*

Severity legend:
- **P0** — silently corrupts training target or features; must mitigate before training.
- **P1** — affects reporting accuracy / defensibility; mitigate if cheap, document otherwise.
- **P2** — code quality, performance, or minor correctness drift.

---

## 2026-04-29 — Audit of `consolidated_modeling_data.parquet` against the prior exclusion list

After the 2026-04-29 data-folder consolidation, the team's single modeling
file is `data/consolidated_modeling_data.parquet` (87 cols, built from Alex's
`alex/scripts/06b_engineer_features.py` joined with the wallet enrichment).
Its column set differs from the post-`scripts/20_finalize_dataset.py`
exclusion in `alex/notes/feature-exclusion-list.md`. Audit findings:

**Aligned (correctly absent):** P0-1 `wallet_is_whale_in_market`; P0-2
`is_position_exit` / `is_position_flip`; P0-8 absolute-scale market-identity
columns (`time_to_settlement_s`, `market_volume_so_far_usd`,
`market_trade_count_so_far`, `size_x_time_to_settlement`, etc. — Alex's
pipeline uses windowed 5min/1h/24h variants instead); P0-9 naive
`wallet_prior_win_rate`; `trade_size_vs_position_pct`,
`wallet_cumvol_same_side_last_10min`, `wallet_has_both_sides_in_market`.

**Causality verified for the `taker_*` analogues** (read 06b directly):
`taker_position_size_before_trade`, `taker_directional_purity_in_market`,
and `market_buy_share_running` all use `cumsum().shift(1).fillna(0)` (or the
strict `codes[:i] == codes[i]` count for purity) with first-in-market reset
to 0 — strictly prior, no temporal leak. `side_buy` and `outcome_yes` are
per-row derivations from `taker_direction` / `nonusdc_side`, no temporal
issue.

**Reintroduced features (present in the parquet, previously dropped):**

| Feature in parquet | Previously dropped as | Audit ID | Risk |
|---|---|---|---|
| `side_buy`, `outcome_yes` | `side`, `outcomeIndex` (renamed) | P0-11 | Within-market direction determinism: the pair partitions trades into 4 categories that map deterministically to `bet_correct` once the market's resolution is known. If the split is by market ID (each market fully in train or fully in test), the shortcut does not transfer to test; if the split shares market IDs across train/test, the model can memorise per-market resolution. |
| `taker_directional_purity_in_market` | `wallet_directional_purity_in_market` | P0-12 | Built from `_side_code = side_buy*2 + outcome_yes`, so re-opens the P0-11 channel via aggregation. Causal across time, but encodes the leaky pair. |
| `taker_position_size_before_trade` | `wallet_position_size_before_trade` | P0-12 | Computed as `tanh(cumsum(sign_yes_equiv * tokens) / 1000)` where `sign_yes_equiv = (2*side_buy − 1) * (2*outcome_yes − 1)`. Same channel as P0-12. |
| `market_buy_share_running` | same name | P0-12 | Aggregate of `side_buy` over prior trades in the market — direction-channel aggregate. |
| `wallet_funded_by_cex` | structurally leaky lifetime flag | feature-exclusion §4 | Lifetime CEX-funding flag, not point-in-time. Redundant: `wallet_funded_by_cex_scoped` (the safe per-trade-time version) is also in the parquet. |

**Recommendation at modeling time:** decide per model family whether to
include the reintroduced features. Conservative default that matches the
prior exclusion list: build a `LEAKY_REINTRO` set and exclude it after
loading.

```python
LEAKY_REINTRO = {
    "side_buy", "outcome_yes",
    "taker_directional_purity_in_market",
    "taker_position_size_before_trade",
    "market_buy_share_running",
    "wallet_funded_by_cex",   # keep wallet_funded_by_cex_scoped instead
}
META_COLS = ["split", "market_id", "ts_dt", "timestamp"]
TARGET    = "bet_correct"
feature_cols = [
    c for c in df.columns
    if c not in META_COLS + [TARGET] and c not in LEAKY_REINTRO
]
```

**Status:** parquet kept as-is (no rebuild). Decision deferred to modeling
stage so each script can run an inclusion/exclusion sensitivity check and
report the deltas.

---

## P0 — Mitigated for this session, upstream fix still needed

### P0-1. `wallet_is_whale_in_market` uses end-of-market p95 as threshold

- **File / line:** `scripts/01_polymarket_api.py:682-695`
- **What's wrong:** the threshold `p95_by_market` is computed from each market's **final** wallet-volume distribution (`groupby([wallet, cid]).sum().groupby(cid).quantile(0.95)`). The running `cum_vol_wm` that is compared to it is correctly no-lookahead (`cumsum() - tv`), but the threshold itself is future information. A wallet's whale designation is therefore calibrated against what the market will eventually look like at settlement, not what the model could have known in real time.
- **Impact severity:** modest-to-moderate leak. The feature still carries real signal (running volume does climb correctly), but the trigger point is informed by future state. MLP may latch onto it.
- **Mitigation applied:** dropped from the feature set at training time (added to `NON_FEATURE_COLS` in `scripts/12_train_mlp.py`). Column is retained in the parquet for audit.
- **Upstream fix (recommended):** replace the end-of-market p95 threshold with one of: (a) a fixed absolute USD threshold (e.g. `$10k` running volume), (b) the wallet's historical p95 across their prior markets, or (c) a rolling per-market p95 of wallets-seen-so-far (complex, cold-start issues).

### P0-2. `is_position_exit` misfires on first-ever SELL

- **File / line:** `scripts/01_polymarket_api.py:667-674`
- **What's wrong:** denominator `max(|pos_before|, size_num)` uses the CURRENT trade size rather than just the prior position. A wallet's first-ever SELL (where `pos_before = 0`) always yields `ratio = size/size = 1.0` → fires `is_position_exit = 1` even though the trade is an opening short, not an exit.
- **Impact severity:** feature is wrong for every wallet's first SELL in every market. For bursty new entrants this biases the `whale-exit` cluster. `is_position_flip` may inherit the issue (same signed-size logic).
- **Mitigation applied:** dropped from feature set at training time (added to `NON_FEATURE_COLS`). Kept in parquet.
- **Upstream fix (recommended):** denominator should be `|pos_before|` only, with a guard returning 0 when `pos_before == 0` (i.e. "fraction of prior position this trade would close"). Re-verify `is_position_flip` logic under the corrected denominator.

### P0-9. `wallet_prior_win_rate` leaks future market-resolution outcomes

- **File / line:** `scripts/01_polymarket_api.py:323-339` (`_add_running_wallet_features`).
- **What's wrong:** the feature is computed as `cumsum(bet_correct) / cumsum(labeled) - current` per wallet. The code respects chronological ordering of trades, but `bet_correct` of each prior trade is only KNOWN once that trade's market has resolved. A wallet's trade on Jan 15 in a "by-Feb-28" market has `bet_correct` determined by the Feb 28 settlement. If the same wallet trades on Jan 20, `wallet_prior_win_rate` at Jan 20 includes the Jan 15 trade's bet_correct — but that value wasn't observable at Jan 20 (Feb 28 market hasn't resolved). Classic future-info peek via `resolution_ts > current_trade_timestamp`.
- **Impact severity:** large. The feature has the strongest linear correlation with the target in our training frame (+0.226 Pearson, 2× the next feature). Much of that correlation is likely leak, not genuine wallet-skill signal.
- **Mitigation applied:** dropped from `NON_FEATURE_COLS` at training time. Column retained in parquet for audit.
- **Upstream fix (recommended):** recompute with a per-row filter — for each trade at time `t`, only include prior trades where `resolution_ts < t`. Requires a per-wallet iteration (can't be pure `cumsum()` because the "resolved-by-now" set varies per row). O(n × k) naive but can be optimised via sorted-per-wallet iteration.

### P0-10. HF-path rows are per-fill; API-path rows are per-order (granularity asymmetry)

- **Files affected:** `scripts/02_build_dataset.py` (HF stream ingest) and the Polymarket Data API path.
- **What's wrong:** a single large order that fills against N maker orders produces:
  - **HF path:** N separate rows in the consolidated CSV (one per on-chain OrderFilled event).
  - **API path:** 1 row (Polymarket's `/trades` endpoint aggregates per-order).
- **Evidence:** train (HF-dominant): 202,082 rows / 129,595 unique `transactionHash` = **1.56 rows/tx**. Test (API, ceasefire markets): 13,414 rows / 13,414 tx = **1.00 rows/tx**.
- **Impact severity:** train / test representation mismatch. Training sees highly correlated per-fill data (same wallet, same prices, same bet_correct across dozens of rows in one tx), effectively over-weighting specific wallet behaviors. Test sees aggregated per-order data. Metrics reported on test don't cleanly compare to train distribution.
- **Mitigation applied:** none yet. Documented. Viable options: (a) deduplicate HF to one canonical row per tx at ingestion (rebuild features), (b) aggregate HF rows to per-tx summary matching API granularity, or (c) switch API to per-fill too (if possible).
- **Upstream fix (recommended):** (b) is the cleanest — aggregate HF rows per `(transactionHash, proxyWallet, side, outcomeIndex)` tuple with size-sum / price-VWAP before feature computation. Matches API semantics and preserves the taker-side action as one event.

### P0-8. Market-identifying absolute-scale features let the MLP memorise markets

- **Context:** Pontus's PR #5 v3 fix dropped 6 features from `train_mlp_reframed.py` because they functioned as implicit market identifiers (absolute-scale features that let the model look up "which market is this" and cheat on bet_correct prediction). The v3 drop was never applied to the current `scripts/12_train_mlp.py` scaffold — those features are still active in the feature set.
- **Features to drop:**
  - `time_to_settlement_s` (scales with market deadline distance)
  - `log_time_to_settlement` (same, log variant)
  - `market_volume_so_far_usd` (absolute USD volume — Feb 28 ≈ 10× Feb 25)
  - `market_vol_1h_log` (absolute rolling volume)
  - `market_vol_24h_log` (same)
  - `market_trade_count_so_far` (absolute trade count)
  - `size_x_time_to_settlement` (interaction feature that uses raw `time_to_settlement_s` — same leak via back door)
- **What's wrong:** at any wall-clock timestamp, these features take very different values per sub-market (because markets have different deadlines and volumes). In a 4-market training cohort with 1 YES + 3 NO, the model can learn "high volume + long time_to_settlement → this is Feb 28 (the YES market) → predict bet_correct accordingly." Training signal collapses to market identity.
- **Impact severity:** **critical shortcut risk.** Without this drop, the MLP can reach near-perfect training loss by memorising which sub-market each trade belongs to. Test generalisation collapses because Apr 18 ceasefire-extended has neither high volume nor long time-to-settlement distribution.
- **Mitigation applied:** not in the shared pipeline yet. For the Alex-workspace MLP training, add to the training script's `NON_FEATURE_COLS` (or equivalent feature-exclusion mechanism). Bounded / normalised substitutes are retained (`pct_time_elapsed`, `market_buy_share_running`, `market_price_vol_last_1h`).
- **Upstream fix (recommended):** apply the same drops to `scripts/12_train_mlp.py` for the shared pipeline, or carry forward the v3 feature-pruning logic from the archived `train_mlp_reframed.py`.

### P0-3. `resolution_ts = NaT` for 3 markets, breaking the settlement filter

- **File / line:** `scripts/01_polymarket_api.py:195-212` (`_first_lock_timestamp`) — strict `.any() < LOCK_UNLOCK_FLOOR` rejected valid locks that had outlier trades after settlement.
- **Markets affected:** `February 27, 2026?` (46,657 trades, in training cohort), `February 8, 2026?` (5,389), `March 3, 2026?` (8,675). All fell back to the placeholder `end_date = 2026-01-31`, causing the `settlement_minus_trade_sec > 0` filter to drop 100% of Feb 27 and Feb 8 trades.
- **Impact severity:** Feb 27 is 46k trades in the training cohort; losing it to the filter removes a key signal market.
- **Mitigation applied:**
  - Upstream patch in `_first_lock_timestamp`: robust median check (`np.median(subsequent) >= LOCK_UNLOCK_FLOOR`) replaces `.any() < LOCK_UNLOCK_FLOOR`. Tolerates outliers, rejects false spikes.
  - One-shot CSV backfill: `scripts/16_patch_resolution_ts.py` applies the same logic to the 3 NaT markets in the existing consolidated CSV, recomputing `resolution_ts` and `settlement_minus_trade_sec`. Applied 22 Apr.
- **Upstream fix:** landed in this branch; ships as part of the PR.

### P0-4. `market_implied_prob` is the trade-execution price for 67/74 markets

- **File / line:** `scripts/02_build_dataset.py:449-466`; feature contract in `scripts/01_polymarket_api.py:enrich_trades`.
- **What's wrong:** the HF-path build only fetches `api_prices` for the 7 ceasefire markets. For the 67 strike + conflict-end markets, `market_implied_prob` falls back to the trade's own execution price (`df.get("price")`). Plan §4 claims "CLOB mid-price where available, otherwise price field" — silently, "otherwise" is the default.
- **Impact severity:**
  - MLP training target: **unaffected** — `market_implied_prob` is already excluded from features (`NON_FEATURE_COLS`).
  - RQ1b trading-rule evaluation: **contaminated** on 95% of rows. The edge `p_hat − market_implied_prob` uses the trade's own execution price as the benchmark, which is biased by the trade itself.
- **Mitigation applied:** none yet. For RQ1b, we will restrict the PnL evaluation to rows with `source == "api"` (the 7 ceasefire markets, ~14.6k trades) where the CLOB mid is real. Strike-market PnL reported separately with the caveat.
- **Upstream fix (recommended):** fetch `fetch_price_history(token_id)` for HF markets' winning and losing tokens during the `02_build_dataset.py` run. One-time cost; adds ~1-2 hours for 67 markets × 2 tokens.

---

## P0 — Time-series-specific concerns (no silent bug, but affect interpretation)

### P0-5. Val cohort is temporally EARLIER than train

- **Context:** Train = Feb 25-28 strike markets. Val = Jan 23 strike market. Test = Apr 18 ceasefire-extended.
- **What's wrong:** training trades have access to richer wallet histories than val trades. `wallet_prior_trades`, `wallet_prior_volume_usd`, `wallet_cumvol_same_side_last_10min`, and all running-features are systematically thinner at val time than at train time.
- **Impact severity:** val-time metrics are pessimistic relative to train. MLP early-stopping may fire early due to noisy val BCE rather than true overfitting.
- **Mitigation applied:** none yet; documented. Increase `EARLY_STOP_PATIENCE` from 5 to 8-10 in `12_train_mlp.py` if val BCE proves unstable.
- **Upstream fix (recommended):** val cohort ideally sits between train and test chronologically. In this dataset there's no post-Feb-28 NO strike market (all post-strike markets resolved YES), so the only candidates are ceasefire or conflict-end markets — both all-NO and in test territory. Documenting as a methodology limitation is the right call.

### P0-6. Wallet group leakage across train / val / test

- **Context:** `proxyWallet` is the same identifier across the whole dataset. A wallet that traded in a Feb 28 strike market (train) and in an Apr 18 ceasefire (test) is the same wallet with the same behavioural fingerprint.
- **What's wrong:** test metrics on "seen" wallets reflect "I've seen this wallet before" in addition to genuine cross-market generalisation. Not a per-row leak (features are no-lookahead within-market), but the generalisation claim is weaker than a pure hold-out would suggest.
- **Impact severity:** overall test ROC-AUC likely inflated relative to novel-wallet generalisation. Affects RQ1 headline claim.
- **Mitigation applied:** planned — split test metrics by `wallet_in_training_set` vs novel-wallet, report both. Pending test run.
- **Upstream fix (recommended):** `GroupKFold(proxyWallet)` for within-training CV. Not applied for this run to avoid scope creep.

### P0-7. Correlated errors within wallet bursts (effective sample size < n_trades)

- **Context:** consecutive trades by the same wallet in the same market are highly correlated (burst patterns, spread-building). Standard ROC-AUC CIs assume iid samples.
- **Impact severity:** reported confidence intervals on test metrics are too narrow.
- **Mitigation applied:** planned — bootstrap 95% CI at the `proxyWallet` level (resample wallets, not trades). Pending test run.
- **Upstream fix:** not an upstream pipeline issue — this is a reporting convention.

---

## P1 — Deferred; document in Methodology → Known Limitations

### P1-1. Layer 7 entropy emits NaN for wallets with exactly one prior market

- **File / line:** `scripts/10_wallet_category_entropy.py:504-515`
- **What's wrong:** emits `wallet_market_category_entropy = NaN` when `total_markets < 2`. Entropy of a degenerate (single-bin) distribution is 0, not undefined.
- **Impact severity:** `wallet_has_cross_market_history` (derived from `.notna()` in `11b_add_missingness_flags.py`) flags single-prior-market wallets as "no cross-market history", which is informationally wrong.
- **Mitigation applied:** none. For the quick-fix training run we accept the NaN (imputed at training time via `SimpleImputer`).
- **Upstream fix (recommended):** emit 0 when `total_markets == 1`; emit NaN only when `total_markets == 0` (no prior distinct markets at all).

### P1-2. `pct_time_elapsed` computed against heuristic `resolution_ts` for 3 patched markets

- **Context:** the 3 markets patched by `16_patch_resolution_ts.py` have a resolution timestamp that is a price-lock estimate, not an on-chain confirmed resolution.
- **Impact severity:** minor. Our heuristic is within hours of the true resolution for Feb 27 and Feb 8; Mar 3 is outside current cohorts.
- **Mitigation applied:** documented.
- **Upstream fix:** not urgent — the 3 markets' resolution_ts values could be reconciled against an on-chain resolution oracle post-deadline.

### P1-3. `market_implied_prob` CLOB asymmetry (see P0-4)

Already logged as P0-4 since it affects RQ1b defensibility. Also documented in the methodology limitations.

### P1-4. Feature distribution shift between train (Feb 25-28) and test (Apr 8-19)

- **Context:** train covers a 4-day peak-informed-flow window; test covers April ceasefire markets with different microstructure (lower trade counts, different wallet mix).
- **Impact severity:** expected and part of the RQ1 design. Not a bug — it's the out-of-sample generalisation test itself. Worth measuring feature-distribution KL divergence between train and test before reading into results.
- **Mitigation applied:** documented.

### P1-5. Time-to-settlement nearly uniform across training markets

- **Context:** all 4 training markets resolve within a 4-day span, so `time_to_settlement_s` distribution in training is dominated by that pattern. Test markets have different deadline spacings.
- **Impact severity:** if `time_to_settlement_s` emerges as a top feature via permutation importance, flag it explicitly in Discussion — it may not transfer cleanly.
- **Mitigation applied:** none; monitor via permutation importance.

---

## P2 — Code quality, defer

- `add_timestamp_split` in `01_polymarket_api.py:725` is dead code under the new market-cohort split (`scripts/14_build_experiment_splits.py`). The emitted `split` column is retained in the CSV but not used.
- Python per-row loops in `01_polymarket_api.py:289-299` (rolling windows), `11_add_layer6.py:146-191` (bisect on timestamp arrays), and `10_wallet_category_entropy.py:496-515` (entropy per row) can be vectorised for 10-20× speedup.
- `11_add_layer6.py:68` loads the 84-col CSV without dtype hints; `proxyWallet` hex strings are robust but a `dtype={"proxyWallet": "string"}` would be more deterministic.
- `09_patch_new_features.py:49-53` vs `01_polymarket_api.py:275-285` use subtly different sort orders; works correctly but not obviously equivalent by inspection.

---

## Shipped on 2026-04-22 (late) — upstream fixes + physical drops

The shared pipeline now physically drops the leak / bug / market-identity
columns in `scripts/20_finalize_dataset.py` rather than relying on a
training-time exclusion list. The pre-drop snapshot is preserved at
`data/03_consolidated_dataset.pre_dropped_variables.csv`. Status by issue:

- **P0-1 `wallet_is_whale_in_market` — FIXED upstream.** `_causal_whale_flag`
  in `01_polymarket_api.py` uses an expanding per-market p95 (sortedcontainers,
  20-wallet warm-up); `17_fix_leakage.py` patched existing CSVs. 0 first-(wallet,
  market) trades flagged (was 3,624); non-whales-by-end flag rate 0.004 (was 0.000
  in the leaky definition). Feature retained.
- **P0-2 `is_position_exit`, `is_position_flip` — PHYSICALLY DROPPED** by
  `20_finalize_dataset.py`. Denominator bug not yet fixed upstream; revisit if
  reinstated.
- **P0-3 `resolution_ts = NaT` for 3 markets — FIXED.** Robust-median lock detector
  landed. `16_patch_resolution_ts.py` backfilled existing CSVs. 74 / 74 markets
  now carry resolution_ts.
- **P0-4 `market_implied_prob` HF-path contamination — OPEN.** Documented known
  limitation; RQ1b backtest should restrict to `source == "api"` or fetch CLOB
  for HF markets (~1 h network).
- **P0-5 val chronology — FIXED** (cohort builder moved val to Mar 15).
- **P0-8 market-identity absolute-scale features — PHYSICALLY DROPPED** by
  `20_finalize_dataset.py` (7 cols: `time_to_settlement_s`,
  `log_time_to_settlement`, `market_volume_so_far_usd`, `market_vol_1h_log`,
  `market_vol_24h_log`, `market_trade_count_so_far`, `size_x_time_to_settlement`).
- **P0-9 `wallet_prior_win_rate` temporal leak — FIXED upstream + original
  DROPPED.** `_add_running_wallet_features` now emits `wallet_prior_win_rate_causal`
  (cumulative mean over priors with `resolution_ts < t`) plus
  `wallet_has_resolved_priors` indicator. Naive leaky version physically dropped
  from the CSV. Empirical: leaky r = 0.367 vs causal r = 0.236 (leak component
  +0.131). `19_add_causal_win_rate.py` backfilled existing CSVs.
- **Data-quality follow-up — end_date placeholder on 34 / 74 markets — FIXED.**
  Gamma `endDate` is stale / `2026-01-31` for 34 strike markets. New
  `parse_deadline_from_question` helper extracts the "by Month Day, Year" string
  from the question and stores `deadline_ts`; all time features now derive from
  `deadline_ts`. `18_fix_deadline.py` backfilled existing CSVs.
- **Metadata bloat (13 cols) — PHYSICALLY DROPPED.**
  `conditionId, title, slug_x, slug_y, icon, eventSlug, outcome, name, pseudonym,
  bio, profileImage, profileImageOptimized, outcomes`. Strictly cosmetic +
  disk savings; none were ever features.
- **`split` column obsolete — PHYSICALLY DROPPED.** Cohort assignment lives
  exclusively in `data/experiments/{train,val,test}.parquet` now.

## P0 — Still open; found after the physical drop (Alex's diagnostic session)

### P0-11. `(side, outcomeIndex)` deterministically encodes `bet_correct`, mapping flips by market resolution

- **File / line:** the pair appears as features anywhere `side` and `outcomeIndex`
  are both fed to a model. Still present in the finalised CSV.
- **What's wrong:** `bet_correct = (outcomeIndex == winning_outcome_index) == (side == "BUY")`. Within any single market, `(side, outcomeIndex)` perfectly
  determines the label. Crucially, the label mapping *flips* between YES-resolved
  and NO-resolved markets. A model trained on a mixed cohort learns the average
  association across resolution types, which then inverts when tested on a
  single-resolution cohort. Alex observed tree models reaching ROC 0.00-0.04 on
  the all-NO ceasefire test set driven by this mechanism.
- **Impact severity:** catastrophic for any model that includes both columns.
  Not a future-info leak in the classical sense (both values are known at trade
  time), but a structural generalisation failure that looks like leakage from the
  model's behaviour.
- **Mitigation applied:** `alex/scripts/*.py` adds `side` and `outcomeIndex` to
  `NON_FEATURE_COLS`.
- **Recommendation (not yet applied to shared CSV):** physically drop `side` and
  `outcomeIndex` from the finalised CSV via `20_finalize_dataset.py`, OR keep them
  only as encoding-for-inference metadata and explicitly document "never feature."

### P0-12. Indirect direction-dependent features inherit P0-11

- **Files / lines:** `01_polymarket_api.py:expand_features` — several
  wallet-in-market features use signed position, same-side filtering, or
  outcomeIndex-share aggregates.
- **Columns affected (Alex's list):** `wallet_position_size_before_trade` (signed
  cumsum of BUY minus SELL); `trade_size_vs_position_pct` (uses signed position);
  `wallet_cumvol_same_side_last_10min` (same-side filter); `wallet_directional_purity_in_market` and `wallet_has_both_sides_in_market`
  (outcomeIndex distribution aggregates); `market_buy_share_running` (running
  share of BUY — per Alex's diagnostic, 0.38 train vs 0.67 test, a 30-point
  cohort shift that re-opens the P0-11 channel).
- **What's wrong:** each feature encodes side / outcomeIndex information
  indirectly. Even with `side` and `outcomeIndex` themselves excluded, these
  features re-introduce the direction-dependent contamination Alex's P0-11
  surfaces.
- **Impact severity:** significant — these were the "safe" features I retained
  after the 22-Apr physical-drop pass. Alex's diagnostic shows they still carry
  the inversion risk.
- **Mitigation applied:** `alex/scripts/*.py` excludes them via NON_FEATURE_COLS.
- **Recommendation (not yet applied to shared CSV):** extend
  `20_finalize_dataset.py` to physically drop them, then regenerate cohort
  parquets.

---

## Confirmed clean (verified during audit)

- No-lookahead enforcement within each market via `cumsum() - tv`, `cumcount()`, and `rolling(..., closed='left')` is correct throughout `_add_running_market_features`, `_add_running_wallet_features`, and `expand_features`.
- `bet_correct` invariant holds for all 74 markets (BUY trades on winning outcomeIndex all have bet_correct=1, other side all 0). Zero label-corruption markets.
- `winning_outcome_index` agrees with price-tail convergence (`outcomeIndex=wi` prices converge to ~1.0, other side to ~0.0) for all 74 markets.
- `is_yes` derivation (PR #6) matches real-world event timeline (strike markets: deadline < Feb 28 → NO, deadline ≥ Feb 28 → YES; all 7 ceasefires → NO).
- Layer 6 bisect uses `side="left"` correctly — a trade exactly at a boundary timestamp does not count its own event.
- `_rolling_count_by_group` / `_rolling_sum_by_group` use `closed='left'` — no current-row leak.
- `wallet_first_minus_trade_sec` is min over full wallet history, always ≤ current timestamp — no future-leak.
- `11b_add_missingness_flags.py` indicator logic consistent with source-column semantics.
- `15_backfill_is_yes.py` merge + winning_outcome_index sanity check correct.
- `03_enrich_wallets.py` per-key rate limiter + deterministic key assignment is correct.
