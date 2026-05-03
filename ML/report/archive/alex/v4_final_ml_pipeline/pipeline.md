# v4 final ML pipeline — runbook

Detailed per-stage runbook. For high-level explanation, decision tree, and how-to-invoke, see [`README.md`](README.md).

---

## ⚠️ Feature engineering happens BEFORE this pipeline

This pipeline starts from already-built feature parquets. The upstream feature-engineering scripts live in `alex/scripts/06b_engineer_features.py` (v3.5) and Pontus's wallet-join script (delivers v4). They run **before** Stage 0.

---

## Stage 0 — Pre-flight (5 min)

Pontus delivers our cohort + his wallet features tacked on. Two parquets:
- `data/train_features_v4.parquet` — 1,114,003 × 76 (70 v3.5 features + 6 wallet)
- `data/test_features_v4.parquet` — 257,177 × 76

Same row order as v3.5 (validated in Stage 1).

```bash
cd ML/report
git stash                    # save local work
git pull origin main         # get Pontus's latest
git stash pop                # reapply local
# Place his v4 parquets at data/{train,test}_features_v4.parquet
# (download command depends on Pontus's release tag — TBD)
```

---

## Stage 1 — Schema validation (5 min)

We don't write a join script. We just verify Pontus's delivery matches the contract before piping into modelling.

```bash
.venv/bin/python alex/v4_final_ml_pipeline/scripts/01_validate_schema.py
```

The script asserts:
- Row counts match v3.5 (1,114,003 train / 257,177 test)
- All 70 v3.5 features still present
- Exactly 6 expected new wallet columns added (no `n_tokentx`, no static `wallet_funded_by_cex`, no `wallet_prior_win_rate`)
- Row order matches v3.5 (validated via `(market_id, timestamp, taker)` key)
- Wallet-feature NaN rate < 0.1% (Pontus v2 claims 100% coverage)
- Auto-updates `data/feature_cols.json` with the 6 new columns on success

If anything fails: halt, ask Pontus to re-deliver. **Do not patch the parquet ourselves** — that breaks the integrity contract.

---

## Stage 2 — Causality guard (5 min)

| Step | Script | What it asserts |
|---|---|---|
| 2.1 | `02_causality_guard.py` (was `pressure_tests/phase1_quick_wins.py`) | 9 checks: pre-event filter, pre_trade_price shift, rolling closed='left', wallet cumsum, scaler refit, dedup, answer1=Yes, top-4 markets resolved NO, HF coverage |
| 2.2 | `pontus/scripts/_check_causal_joined.py` | Pontus's 51 invariant checks on the joined feature set |

Both must PASS or the pipeline halts. Re-run after every feature change.

---

## Stage 3 — EDA — owned by Pontus

Not in our scope. We consume his outputs (likely under `pontus/outputs/eda/`) and reference in the report's Methodology. We do NOT produce EDA scripts.

What we expect from him:
- Univariate distributions on the 76-feature joined set
- Correlation heatmap covering both feature families
- Per-market outcome rates (train + test)
- Wallet-feature distributions
- (Optional) PCA scree + top-2 PC scatter

If Pontus's EDA omits PCA, we still have PCA-as-modelling-baseline in `03_sweep.py` (PCA+LogReg is one of the sweep models). Enough to tick the L05 dimensionality-reduction box.

---

## Stage 4 — Supervised model training (4-6 hr)

A single command runs all primary supervised models + IsoForest:

```bash
.venv/bin/python alex/v4_final_ml_pipeline/scripts/03_sweep.py
```

Models in the sweep:
| Slot | Model | Notes |
|---|---|---|
| 1 | LogReg L2 | Linear anchor |
| 2 | LogReg L1 | Elbow / feature selection |
| 3 | Decision Tree | Single-tree interpretability |
| 4 | Random Forest | Bagged trees — current headline |
| 5 | HistGradientBoosting | Boosting (sklearn) |
| 5b | **LightGBM** | NEW — alternative boosting library, often +0.5pp vs HGBM. Skipped silently if `lightgbm` not installed. |
| 6 | PCA(20) → LogReg | Dimensionality-reduction pipeline |
| 7 | **MLP (TF/Keras)** | Feed-forward NN, 64 → 32 → 1 with BN + SELU + dropout 0.3 |
| 8 | Isolation Forest | UNSUPERVISED anomaly detection |

Per supervised model: 5-fold GroupKFold CV on train, fit isotonic on OOF, final fit on full train, score on test. Save metrics + per-market AUC + feature importance.

Outputs:
- `.scratch/preds/preds_<model>.npz` — raw + calibrated + OOF (uniform schema across all models)
- `outputs/sweep_idea1/<model>/{metrics.json, per_market_test.json, feature_importance.json}`
- `outputs/sweep_idea1/comparison_table.md` — cross-model comparison
- MLP-specific: `outputs/sweep_idea1/mlp_keras/training_history.json`

---

## Stage 5 — Unsupervised deep-dive (30 sec)

```bash
.venv/bin/python alex/v4_final_ml_pipeline/scripts/04_iso_forest.py
```

Standalone IsoForest analysis — complements the IsoForest model in `03_sweep.py` with a focused look at anomaly-score correlation with `bet_correct`, top-k anomaly precision, and per-market anomaly distributions.

Outputs `outputs/sweep_idea1/iso_forest/{metrics.json, test_anomaly_scores.parquet}`.

(Optional T2: TF/Keras autoencoder for reconstruction-error anomaly. Not implemented — defer unless v4 IsoForest still null.)

---

## Stage 6 — Hyperparameter tuning (2-3 hr)

```bash
.venv/bin/python alex/v4_final_ml_pipeline/scripts/05_optuna_tuning.py --n_trials 50
```

Drop the partial 70-feature study (`study_v3.5_partial.db`) into archive. Fresh 50-trial TPE per model on v4 features:
- Random Forest
- HistGradientBoosting
- LightGBM (if installed)

SQLite-backed (resumable), parallel subprocesses, 5-fold GroupKFold(market_id) CV inside each trial. Outputs:
- `outputs/rigor/optuna/<model>/{study.db, study_history.csv, best_params.json, comparison.json, preds_test_tuned.npz}`

(Optional T2: tune MLP separately via a `05b_` extension. Defer unless MLP is in the top-2 of Stage 4.)

---

## Stage 7 — Decomposition + rigor + complexity + interpretability (3 hr)

| # | Script | What | Time |
|---|---|---|---|
| 7.1 | `06_phase2_falsification.py` | B1a (consensus decomposition), B1b (naive baseline AUC), A1 (SELL semantics). Outputs `.scratch/pressure_tests/phase2_v4_results.json` | 30 sec |
| 7.2 | `07_rigor_additions.py` (skeleton) | Bootstrap CI (1000 resamples) on test AUC, DeLong test for pairwise model AUC, permutation importance per model, learning curves (AUC vs train fraction). Outputs `outputs/rigor/` | 1.5 hr |
| 7.3 | `08_complexity_benchmark.py` (skeleton) | Per-model fit time, predict time, parameter count. CSV + bar chart. Required by `exam.md:47`. Outputs `outputs/rigor/complexity_benchmark.{csv,png}` | 30 min |
| 7.4 | `09_shap_top_picks.py` (skeleton) | SHAP values for top-1% picks (best model). Per-pick + global summary plot. Outputs `outputs/rigor/shap/` | 1 hr |

**Critical question:** does B1b still falsify on v4? If naive baseline still matches the augmented model on top-1% precision, the negative finding survives. If not, the asymmetric-info narrative recovers. See README's decision tree.

---

## Stage 8 — Economic backtest (2 min)

| # | Script | Note |
|---|---|---|
| 8.1 | `10_backtest.py` | Flat-stake economic backtest. ~30 sec from cached preds |
| 8.2 | `11_realistic_backtest.py` | Capital-aware (bankroll + concentration + gas + slippage + copycat sharing). ~30 sec |
| 8.3 | `12_sensitivity_sweep.py` | cost_floor × copycat-N grid. ~30 sec |

All three load from `.scratch/preds/preds_*.npz` — just re-run after Stage 4 finishes. Outputs overwrite `outputs/backtest/{*, realistic/, sensitivity/}`.

---

## Wall-time budget

| Stage | Time |
|---|---|
| 0 Pre-flight | 5 min |
| 1 Schema validation | 5 min |
| 2 Causality guard | 5 min |
| 3 EDA | **Pontus** |
| 4 Supervised (`03_sweep.py` — RF + HGBM + LightGBM + MLP + 5 others + IsoForest) | 4-6 hr |
| 5 Unsupervised deep-dive | 30 sec |
| 6 Optuna re-tune | 2-3 hr |
| 7 Decomposition + rigor + complexity + SHAP | 3 hr |
| 8 Backtest | 2 min |
| **Total P0+P1 (ours)** | **~10-12 hr** |

T2 stretch (Stacking + AE + MLP-tune + strict-cohort eval) adds ~5 hr.

---

## Parallelism plan

```
After Stage 1 + 2 PASS, Stage 4 (03_sweep.py) internally runs all models
sequentially in one process. The MLP step is the single longest item;
on M4 Pro, total sweep wall time is ~30 min sklearn + ~30 min MLP.

Stage 6 (05_optuna_tuning.py) spawns 2-3 parallel subprocesses:
    RF tuning (~2 hr)
    HGBM tuning (~30 min)
    LightGBM tuning (~30 min, similar to HGBM)

Stages 7.1, 7.2, 7.3, 7.4 are independent → all 4 can run in parallel
after Stage 4 completes.

Stage 8.1, 8.2, 8.3 are independent → can run in parallel.
```

---

## File flow diagram

```
Raw HF dataset
    │
    ▼
[upstream — alex/scripts/build_cohorts.py — already ran]
    │
    ├──► markets_subset.parquet
    ├──► train.parquet
    └──► test.parquet
            │
            ▼
[upstream — alex/scripts/06b_engineer_features.py — already ran]
    │
    ├──► train_features.parquet  (v3.5, 70 features)
    └──► test_features.parquet
            │
            ▼
[Pontus joins his Layer-6 wallet features per-row, ON HIS SIDE]
    │
    ▼
   ┌── train_features_v4.parquet ◄── delivered by Pontus
   └── test_features_v4.parquet  ◄── delivered by Pontus
            │
            ▼
[STAGE 1: 01_validate_schema.py — must PASS]
            │
            ▼
[STAGE 2: 02_causality_guard.py + pontus/scripts/_check_causal_joined.py — must PASS]
            │
            ▼
[STAGE 3: EDA — Pontus's deliverable]
            │
            ▼
[STAGE 4: 03_sweep.py — runs all supervised + IsoForest sequentially]
    │
    └──► .scratch/backtest/preds_<model>.npz × N models
            │
            ├──► [STAGE 5: 04_iso_forest.py]  →  outputs/sweep_idea1/iso_forest/
            │
            ├──► [STAGE 6: 05_optuna_tuning.py]  →  outputs/rigor/optuna/
            │
            ├──► [STAGE 7.1: 06_phase2_falsification.py]  →  pressure_tests/phase2_v4.json
            │
            ├──► [STAGE 7.2: 07_rigor_additions.py]  →  outputs/rigor/{auc_ci,deLong,perm_importance,learning_curve.png}
            │
            ├──► [STAGE 7.3: 08_complexity_benchmark.py]  →  outputs/rigor/complexity_benchmark.{csv,png}
            │
            ├──► [STAGE 7.4: 09_shap_top_picks.py]  →  outputs/rigor/shap/
            │
            └──► [STAGE 8: 10_backtest.py + 11_realistic_backtest.py + 12_sensitivity_sweep.py]
                       │
                       └──► outputs/backtest/{*, realistic/, sensitivity/}
```

---

## Decision points (where pipeline branches based on results)

| Decision | Trigger | Action |
|---|---|---|
| **Build v5 (winrate-causal)?** | Stage 7.1 B1b on v4 still FAILS (naive matches model) AND we have ~3 hr to spare | Write a new `compute_winrate_causal` step (upstream of this pipeline), regenerate v5 features, re-run stages 1-8 on v5 |
| **Skip stacking?** | Stage 4 augmented MLP doesn't beat best tree by > 1pp | Skip stacking — won't add value |
| **Tune MLP separately?** | MLP is in top-2 of stage 4 | Run an `05b_optuna_mlp.py` extension |
| **Skip autoencoder?** | Time-constrained AND IsoForest still null on v4 | Skip — adding another null finding doesn't help |
| **Run strict-cohort?** | Need joint co-report material with Pontus | Implement `13_strict_cohort_eval.py` or similar |
| **Swap LightGBM in?** | LightGBM beats HistGBM by > 0.5 pp on test AUC | Use LightGBM as the primary boosting model in the headline |
| **Replace RF headline with MLP?** | MLP beats RF on test AUC AND CIs don't overlap (DeLong p < 0.05) | Promote MLP to headline — but disclose tree-MLP gap is small in CIs |

---

## "Done" gates

The pipeline is complete when:

1. ✅ Schema validation + causality guard PASS on v4
2. ✅ Stage 4: predictions for all primary models (RF, HGBM, LightGBM, MLP) saved as `.scratch/backtest/preds_<model>.npz`
3. ✅ Stage 6: Optuna study completed for at least RF + HGBM (50 trials each)
4. ✅ Stage 7.1: B1a / B1b / A1 verdicts in `phase2_v4_results.json`
5. ✅ Stage 7.2: bootstrap CI + DeLong + permutation importance + learning curves all written
6. ✅ Stage 7.3: complexity_benchmark.csv has a row per model
7. ✅ Stage 7.4: SHAP top-picks plots written
8. ✅ Stage 8: realistic backtest + sensitivity sweep regenerated against v4 preds
9. ✅ New design decisions logged in `../notes/design-decisions.md` (D-038 onward)

After that we have all the numerical content for the report's Methodology + Results + Discussion sections. Drafting is downstream.

---

## Out of scope

- **Feature engineering** — happens upstream (`alex/scripts/06b_engineer_features.py` for v3.5; Pontus's join for v4)
- Report drafting (Ethical Consideration, Conceptual Framework, LLM Usage Disclosure, Abstract, References)
- New wallet feature engineering on our side beyond Pontus's v2
- Cohort changes (v4 uses the same 75-market split as v3.5)
- Re-running v3.5 numbers (locked in PR #13, become the comparison baseline)
- EDA (Pontus's deliverable)
