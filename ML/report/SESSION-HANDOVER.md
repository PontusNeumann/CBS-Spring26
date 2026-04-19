# Session Handover — 2026-04-19 evening

## TL;DR for next session

Enrichment is running overnight (~7h remaining). Everything upstream of it is done. Next session picks up by:
1. Verifying enrichment completed (`cat data/enrichment_progress.json`)
2. Adding Layer 6 to `build_dataset.py` (bisect-based per-trade features)
3. Rebuilding labeled dataset with 8–10 new features
4. Retraining both MLPs (reframed + original `bet_correct` target)
5. Re-running backtest, comparing against today's baseline numbers
6. Drafting the report

## What's currently running

| Process | PID | Purpose | ETA |
|---|---|---|---|
| `enrich_onchain.py` | 18451 | Fetching Etherscan tokentx for 55,329 wallets | ~7h 16m |
| `render_dashboard.py` | 18551 | Live HTML dashboard rendering | — |

**Do not kill these.** Enrichment writes to `data/wallet_enrichment.parquet` every 500 wallets (checkpointed, resumable).

## State of the pipeline

| Stage | Status |
|---|---|
| Dataset build (v1, no Layer 6) | ✅ 451,933 × 63 cols, zero nulls, label 50.2%, saved |
| EDA (8 plots + narrative report) | ✅ `data/eda_outputs/`, pushed to repo (PR #3) |
| Baselines (logreg L1/L2, RF, IF, naive market) | ✅ `data/baseline_outputs/` |
| MLP v1 on `bet_correct` | ✅ val ROC 0.62, test ROC 0.23 (inverts) |
| Autoencoder arm (L11) | ✅ `data/ae_outputs/` |
| HPO (L13, 6 configs) | ✅ all invert on test |
| Permutation importance (L14) | ✅ `data/mlp_outputs/permutation_importance.csv` |
| Reframed MLP (target = P(YES)) | ✅ val ROC 0.94, test ROC 0.50 |
| Streaming backtest | ✅ `data/backtest_outputs/` |
| Layer 6 enrichment | 🟡 running overnight |
| Layer 6 integrated into `build_dataset.py` | ⏭ next session |
| Retrain with Layer 6 + re-backtest | ⏭ next session |
| Report drafting | ⏭ next session |

## Key empirical findings from today

1. **Original framing (target=`bet_correct`) inverts on test across ALL models** — RF test ROC 0.15, MLP test ROC 0.23 across 6 HPO configs, consistent across architectures. Root cause: training is 70% NO-outcome markets, test is 99% YES — direction features (`is_buy`, `is_token1`) dominate predictions and reverse sign between regimes.

2. **Reframing to target=`market_resolves_yes` fixes inversion** — val ROC on `bet_correct` jumped 0.62 → 0.94, gap AUC 0.58 → 0.82. Test still weak (0.50) but no longer inverted — just structurally dominated by single-market outcome homogeneity (99% Feb 28 YES).

3. **Severe overfitting on 7 markets** — train loss 0.005 vs val 5.9. Only 7 effective outcome samples. Model memorises market trajectories. Honest Limitations section material.

4. **Backtest — MLP underperforms random** — MLP general +EV returns 78% vs random 93%, MLP home-run 0% hit rate on 179 triggers. Test period is a high-return regime (market priced Feb 28 at $0.45 avg, YES won) so every strategy makes money, but our model's confidence signal actively deselects the winning trades. Rich Discussion material, honest result.

5. **Autoencoder cross-check** — top-decile recon error × top-decile |gap| overlap: 2.78× vs null on val, 1.19× on test. Signals agree where regime is stable, diverge in the pressure window. Consistent with the MLP inversion story.

## New scripts written today (all in `scripts/`)

- `build_dataset.py` — unchanged from PR #2, produces labeled.parquet
- `baselines.py` — 4 models + naive market
- `train_mlp.py` — original bet_correct MLP
- `train_mlp_reframed.py` — target=P(YES) MLP
- `train_autoencoder.py` — L11 arm
- `interpret_mlp.py` — permutation importance
- `mlp_hpo.py` — 6-config grid
- `backtest.py` — streaming event-replay
- `enrich_onchain.py` — multithreaded Etherscan fetcher (running)
- `render_dashboard.py` — live HTML progress dashboard (running)
- `eda.py` — 8-section EDA with narrative HTML report

## New outputs (data/)

```
data/
  iran_strike_markets.parquet       7 rows, metadata
  iran_strike_trades.parquet        451,933 raw trades
  iran_strike_labeled.parquet       451,933 × 63 cols (v1, no Layer 6)
  wallet_enrichment.parquet         IN PROGRESS — Layer 6 raw per-wallet
  enrichment_progress.json          live stats
  enrichment_dashboard.html         live dashboard
  eda_outputs/                      8 PNGs + summary.txt + report.html
  baseline_outputs/                 metrics.csv + rf_feature_importance.csv + lasso_coefs.csv + gap_evaluation.csv
  mlp_outputs/                      model.pt + scaler + calibrator + predictions + metrics + loss_curve + permutation_importance + hpo_results
  mlp_reframed_outputs/             model.pt + scaler + calibrator + predictions + metrics + loss_curve
  ae_outputs/                       ae_model.pt + recon_errors.parquet + overlap_analysis.json + loss_curve
  backtest_outputs/                 strategy_metrics.csv + cutoff_sweep.csv + pnl_curves.png
```

## GitHub state

Pontus's repo `PontusNeumann/CBS-Spring26`:
- `main` — has PR #1 (initial data + docs) merged, PR #2 (enrichment design + scripts) merged, PR #3 (EDA fixes + outputs + report.html) open
- Today's new scripts + outputs — **NOT yet pushed**. Next session or end-of-session should push an update PR.

## Next session — first 10 minutes

```bash
# 1. Check enrichment state
cat "/Users/alex/Documents/Claude Projects/ml-exam-project/data/enrichment_progress.json"
# expect pct_done near 100, failures < 500

# 2. Check the output file exists and has ~55k rows
.venv/bin/python -c "import pandas as pd; df = pd.read_parquet('data/wallet_enrichment.parquet'); print(df.shape); print(df.head())"

# 3. Kill the background processes
pkill -f enrich_onchain.py
pkill -f render_dashboard.py

# 4. Start the Layer 6 integration work:
#    - Write a function in build_dataset.py that joins wallet_enrichment.parquet
#      and bisects its timestamp arrays at each trade's timestamp
#    - Re-run build_dataset.py to produce iran_strike_labeled_v2.parquet with
#      the new columns
```

## Open questions / decisions for next session

- **Do we retrain BOTH the original `bet_correct` MLP and the reframed `P(YES)` MLP with Layer 6?** Proposal: yes — direct comparison is the cleanest empirical question
- **Does Layer 6 close the train/test generalization gap?** Hypothesis: CEX-funded + brand-new-wallet features carry enough trade-specific signal that the model is less dependent on market-identifying features → less overfitting to per-market trajectory
- **Do we open a cross-event-family or multi-market-pooled robustness check if Layer 6 fails too?** Big scope bump — only do if exam deadline allows

## Report status

Not yet started. For next session, rough outline:

```
1. Abstract (200 words)
2. Intro — prediction markets, Iran event, Magamyman case, why ML?
3. Conceptual Framework — asymmetric info, efficient markets, insider-trading literature
4. Methodology
   - Data (7 markets, 451k trades, HF source, causal features)
   - Feature engineering (6+1 layers, causal guarantees, improvements over syllabus)
   - Model architecture (MLP + baselines + autoencoder, calibration)
   - Split strategy + target framing (reframed to P(YES))
5. Results
   - Classification metrics val + test
   - Autoencoder cross-check
   - Trading-signal backtest
   - Cutoff sweep
6. Ethics — privacy, dual-use, Polymarket's stance
7. Discussion
   - Inversion finding + why reframing helped on val
   - Overfitting to 7 markets (honest limitation)
   - Backtest result — MLP vs random on test window
   - Layer 6 impact (filled in next session)
8. Conclusion
9. Appendix — Magamyman anecdote, prompt / LLM disclosure
```

## Good night

Enrichment will be done around 03:30 local time if current rate holds. First thing to check in the morning is `enrichment_progress.json` — if `completed >= 55000` and `failures < 500`, we're golden.
