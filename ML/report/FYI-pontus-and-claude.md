# FYI — Pontus + your Claude

*Short brief so you can get oriented fast without re-deriving everything. Alex + his Claude pushed a lot of pipeline-correctness work on 2026-04-22. Key items below. Longer form in `data-pipeline-issues.md`.*

## TL;DR — what changed

1. **`is_yes` column added** at trade level (PR #6). Derived from `outcomes[winning_outcome_index] == "Yes"` inside `enrich_trades`. `project_plan.md` §2 previously said "55 YES / 19 NO" — that was flipped; actual is **19 YES / 55 NO**. All 7 ceasefire markets resolved **NO**.
2. **`resolution_ts` fixed for 3 markets** (PR #8). `_first_lock_timestamp` in `01_polymarket_api.py` was too strict; rejected valid locks on any outlier below `LOCK_UNLOCK_FLOOR`. Now uses median-based robust check. Scripts 16 backfills the current CSV; no full rebuild required.
3. **Market-cohort split design** (PR #7) replaces the old trade-timestamp quantile split. Full rationale in `project_plan.md` §4 → Split row.
4. **Cohort builder `scripts/14_build_experiment_splits.py`** produces `data/experiments/{train,val,test}.parquet`. Train = 4 strike markets (Feb 25-28, peak informed-flow window, ~202k rows). Val = 1 conflict-end market (Mar 15, ~13k rows, post-training, cross-family). Test = all 7 ceasefire markets (~13k rows, Apr 8-19, cross-family, all NO).
5. **Data-pipeline issues log** at `data-pipeline-issues.md` — 10 P0 items, 5 P1 items. Each has file:line, what's wrong, current mitigation, and recommended upstream fix.
6. **Alex has his own workspace** at `alex/` for his personal MLP experiments. Pontus's shared pipeline scripts (`scripts/`) are untouched from that folder.

## The P0 items that affect modelling most

**If you're about to run `12_train_mlp.py`, these matter:**

| Ref | Issue | Current state | Action |
|---|---|---|---|
| P0-1 | `wallet_is_whale_in_market` uses end-of-market p95 threshold → leak | Dropped from `NON_FEATURE_COLS` in 12_train_mlp.py | Keep dropped until upstream fix |
| P0-2 | `is_position_exit` fires on fresh SELLs (denominator bug) | Dropped | Keep dropped |
| P0-8 | 6 market-identifying absolute-scale features + `size_x_time_to_settlement` let model memorise sub-markets | Dropped | The v3 feature-pruning fix from PR #5 never migrated to 12_train_mlp.py — we've migrated it now |
| **P0-9** | **`wallet_prior_win_rate` peeks at future market resolutions** — `bet_correct` cumsum on priors includes outcomes not yet known at trade time | **Just dropped (2026-04-22)** | **Strongest linear correlate (+0.226). Likely mostly leak. Correctly recomputed version would filter priors by `resolution_ts < t`** |
| P0-10 | HF-path is per-fill (~1.56 rows/tx), API-path is per-order (1 row/tx) | Documented, not fixed | Train/test granularity mismatch; documented as limitation |

Full `NON_FEATURE_COLS` set to copy into any modelling script: `alex/notes/feature-exclusion-list.md`.

## The P0 items that are already shipped / handled

- P0-3 `resolution_ts = NaT` for Feb 27 / Feb 8 / Mar 3 → **fixed** in `01_polymarket_api.py` and backfilled to CSV via `scripts/16_patch_resolution_ts.py`.
- P0-4 `market_implied_prob` contaminated by trade price on HF markets — Not used as feature (already in NON_FEATURE_COLS). Only affects RQ1b PnL benchmark — restrict PnL to `source=='api'` rows.
- P0-5 Val temporally before train → **fixed**: val now = Mar 15 conflict-end (post-training).
- P0-6 Wallet group leakage across cohorts — analytical mitigation (split test metrics by seen vs novel wallet); no code change.
- P0-7 Correlated errors within wallet bursts — analytical (bootstrap test CIs at wallet level).

## Cohort specs as of 2026-04-22

| Cohort | File | Rows (post-filter) | Markets | Resolution | Time span |
|---|---|---:|---|---|---|
| Train | `data/experiments/train.parquet` | 202,082 | 4 strike (Feb 25-28) | 1 YES + 3 NO | through Feb 28 |
| Val | `data/experiments/val.parquet` | 13,154 | 1 conflict-end (Mar 15) | NO | Feb 28 → Mar 17 |
| Test | `data/experiments/test.parquet` | 13,414 | 7 ceasefires (Apr 8-18) | all NO | Apr 8 → Apr 19 |

Chronology: train → val → test, strictly. ~3 weeks between val end and test start.

## Feature set

- 84 columns in parquet.
- 40 excluded (identifiers, raw-superseded, label/benchmark/filter, 4 leaky/memorising features, 1 temporal-leak feature).
- **44 features** remain for modelling.
- Target: `bet_correct`.

Feature groups surviving: Layer 6 on-chain identity (12), wallet-in-market behavioural (13), wallet global (3 — was 4 before `wallet_prior_win_rate` drop), sizing/interactions (4), trade-local (4), market context normalised (3), missingness indicators (3), pct_time_elapsed (1), Layer 7 cross-market entropy (1).

## Things to know if you inspect the work

- **Alex's folder (`alex/`)** is his own playground. Pontus's `scripts/` is shared, unmodified from Alex's side. If you want to see his modelling approach, look at `alex/scripts/` and `alex/outputs/`.
- **If `NON_FEATURE_COLS` differs between Pontus's and Alex's scripts**, treat `alex/notes/feature-exclusion-list.md` as the canonical current drop set. Pontus's `scripts/12_train_mlp.py` has been updated to match (as of 2026-04-22).
- **Open decision — sister-market features.** `alex/notes/sister-market-features.md` documents a deferred feature-engineering idea (cross-market price injection per plan §8). Not implemented yet; trigger conditions spelled out.
- **The test set is all-NO resolution.** This is a validity concern flagged and then re-evaluated in `alex/notes/test-cohort-no-bias.md`. Per-trade `bet_correct` is ~50/50 within every market regardless of market resolution, so this isn't a blocking issue — but feature-importance interpretation should acknowledge it.

## Where to read next

- `data-pipeline-issues.md` — canonical issues list with P0/P1 severities and line refs.
- `project_plan.md` — §4 Split row is the current authoritative split spec. §11 Open Decisions has resolved + open items.
- `alex/README.md` — workspace overview, current focus.
- `alex/notes/feature-exclusion-list.md` — copy-paste `NON_FEATURE_COLS` for any new modelling script.

Ping Alex if anything here conflicts with what you're seeing.
