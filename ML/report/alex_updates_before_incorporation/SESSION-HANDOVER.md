# Session Handover — 2026-04-20

## TL;DR for next session

Enrichment is 100% complete (55,329 / 55,329). Layer 6 is integrated. We found a memorisation issue and a partial fix that produced the first positive backtest result, but we've been doing **exploration**, not a finalised ML pipeline. Next session: lock the feature set, rerun baselines + MLP (5 seeds) apples-to-apples, do ablations, and only then report final PnL.

## State at end of session

| Stage | Status |
|---|---|
| Enrichment 55,329 / 55,329 wallets, zero failures | ✅ |
| `data/wallet_enrichment.parquet` (186 MB) — per-wallet scalars + timestamp arrays | ✅ |
| `scripts/add_layer6.py` — joins enrichment onto labeled v1 via bisect | ✅ |
| `data/iran_strike_labeled_v2.parquet` (91 MB, 451,933 × 75 cols) | ✅ |
| Reframed MLP v2 with Layer 6 trained | ✅ test target ROC 0.66 / test derived ROC 0.31 / backtest 77% |
| **Memorisation diagnosed** — train loss → 0.005 in 1 epoch, model learning market fingerprints | ✅ locked finding |
| **v3 fix — dropped market-identifying absolute-scale features** | ✅ test target ROC 0.73 / derived 0.34 / **backtest 116.4%** (single seed) |
| Formal ML pipeline — finalised features, apples-to-apples baselines, multi-seed variance, ablations | ⏭ next session |
| Report drafting | ⏭ next session |

## Key new findings from today

1. **Memorisation via market fingerprints** — with only 7 markets in the dataset, absolute-scale market-level features (`time_to_settlement_s`, `market_cumvol_log`, `market_vol_*_log`, `log_time_to_settlement`) let the model identify each specific market from its feature values and look up the known outcome. Training loss collapses to 0.005 after 1 epoch. This is the single most important methodology finding of the project.

2. **Fix by feature pruning** — dropping those features forced the model to rely on per-trade and per-wallet signals that generalise across markets. Train loss epoch 1 rose from 0.08 → 0.36 (healthy). Test target ROC improved 0.66 → 0.73. Backtest general-+EV strategy went from underperforming random (77%) to beating random by +23.6 pp (**116.4% return, Sharpe 83.5, hit rate 0.549 vs naive 0.524**).

3. **BUT — single seed, ad-hoc feature set.** We haven't run a proper finalised ML pipeline. The 116% is an exciting preliminary signal, not a reportable result. Next session must confirm with 5-seed variance, apples-to-apples baseline comparison, and feature-group ablations.

4. **Home-run strategy consistently fails** (−100% across v1, v2, v3). The (edge > 0.20, TTS < 6h, price < 0.30) filter picks losers in this test market. Two candidate explanations — filter is mis-specified, or model's extreme-confidence picks are genuine contrarian signals. Open.

## Scripts written today

- `scripts/add_layer6.py` — bisect-based per-trade features from per-wallet timestamp arrays
- `scripts/train_mlp_reframed.py` updated twice: (a) Layer 6 feature set v2, (b) market-identifying features dropped v3
- `scripts/enrich_onchain.py` re-run as resumable retry to clean up 861 → 0 failures

## New output directories

```
data/
  wallet_enrichment.parquet          186 MB — NOT pushed (>100MB limit)
  iran_strike_labeled_v2.parquet      91 MB — pushed (over warning, under hard limit)
  mlp_reframed_v2_outputs/            with Layer 6
  mlp_reframed_v3_outputs/            with Layer 6 + market-identifying features dropped
  backtest_v2_outputs/                corresponding backtests
  backtest_v3_outputs/
  enrichment_progress.json            final snapshot
```

## Next session — first 30 minutes

### Step 1 (collaborative): finalise the feature set

Produce a keep/drop/replace table for every one of the 75 columns in `iran_strike_labeled_v2.parquet`. Decision criteria:

- **Drop** features that uniquely identify individual markets (absolute-scale market-level)
- **Keep** per-trade features bounded by construction (price-based, 0-1 percentages, flags)
- **Keep** per-wallet features that don't encode market identity
- **Keep** Layer 6 identity features (`polygon_age`, `polygon_nonce_log`, `funded_by_cex_scoped`, etc.)
- **Replace** absolute-scale market features with market-agnostic versions (percentile within market OR ratio to market's own median) if we want to preserve the signal they carry

Target: ~25-35 final features, each documented.

### Step 2 (automated): build `iran_strike_labeled_final.parquet`

Small script that takes v2, applies the decisions from Step 1, emits the final dataset.

### Step 3 (automated): apples-to-apples baselines on final features

Rerun `baselines.py` with the new feature set. Logreg L1/L2, RF, IF, naive market.

### Step 4 (automated): MLP training, 5 seeds

Train the reframed MLP with seeds 42, 1, 2, 3, 4. Report mean ± std for val/test ROC, PR-AUC, gap AUC, and backtest returns. Confirm 116% isn't a single-seed fluke.

### Step 5 (automated): feature-group ablations

Drop each of {time-features / market-state / wallet-global / wallet-in-market / Layer 6 / interactions} in turn, retrain, measure Δ test ROC and Δ backtest return. Identifies which groups carry signal.

### Step 6 (automated): proper backtest with seed variance

Run the 5 MLP seeds through `backtest.py`, report distribution of general +EV and home-run returns.

### Step 7: report drafting

With genuine pipeline results in hand, start the 15-page report per the MLDP exam brief template (Abstract → Intro → Conceptual Framework → Methodology → Results → Ethics → Discussion → Conclusion).

## Current best result (provisional, unreliable)

- MLP reframed v3 (Layer 6, market-identifying features dropped), single seed 42:
  - Val target P(YES) ROC: 0.80
  - Test target P(YES) ROC: 0.73
  - Val derived bet_correct ROC: 0.87
  - Test derived bet_correct ROC: 0.34 (still below 0.5 — inverts)
  - Test gap_roc: 0.52
  - Backtest MLP general +EV: **116.4% return, hit rate 0.549, Sharpe 83.5** (vs random 92.9%, naive 92.5%)

**This is exploration, not confirmation.** Next session must reproduce under proper conditions.

## Files NOT pushed (size or privacy)

- `wallet_enrichment.parquet` (186 MB) — needs Git LFS or external hosting. Can regenerate from `scripts/enrich_onchain.py` + `.env` keys.
- `.env` (never pushed) — 3 Etherscan keys + 1 Alchemy key.

## PRs shipped in the project so far

- #1 dataset + initial docs (merged)
- #2 enrichment design + scripts (merged)
- #3 EDA fixes + outputs + `report.html` (merged)
- #4 full modelling pipeline — baselines, MLP v1/reframed, autoencoder, HPO, permutation importance, backtest (state unknown; likely still open)
- **#5 (this session)** — Layer 6, memorisation diagnosis + fix, v2/v3 variants, handover update

## Resources still in memory

- `data/wallet_enrichment.parquet` — don't delete; this is the expensive artefact
- `data/iran_strike_labeled_v2.parquet` — don't delete; source of v3 etc
- `data/iran_strike_trades.parquet` — raw; don't delete
- `data/iran_strike_markets.parquet` — tiny; don't delete
- `data/hf_raw/trades.parquet` + `markets.parquet` — 28 GB + 116 MB; can delete if disk pressure, regenerate via `hf download` as needed
