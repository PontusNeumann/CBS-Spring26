# v4 final ML pipeline

The locked, end-to-end ML/DL pipeline for Alex's idea1 contribution to the report. Runs once the v4 wallet-augmented parquets land from Pontus, produces all numerical content for the Methodology + Results + Discussion sections, and stops there. Report-writing is downstream of this pipeline.

**Read this README first.** For per-stage details (scripts, inputs/outputs, wall times), see [`pipeline.md`](pipeline.md).

---

## ⚠️ Feature engineering happens BEFORE this pipeline

This pipeline does **not** engineer features from raw trades. It starts from already-built feature parquets:

```
Raw HF dataset
    │
    ▼
[upstream — already done, NOT in this pipeline]
    build_cohorts.py            (alex/scripts/) — produces train.parquet, test.parquet
    06b_engineer_features.py    (alex/scripts/) — produces v3.5 70-feature parquets
    │
    ▼
[Pontus joins his Layer-6 wallet identity features per-row, ON HIS SIDE]
    │
    ▼
data/train_features_v4.parquet      ← THIS PIPELINE'S INPUT
data/test_features_v4.parquet       ← THIS PIPELINE'S INPUT
```

The 12 numbered scripts in `scripts/` consume the v4 parquets and produce models, evaluation, decomposition, rigor, interpretability, and backtests. Feature engineering scripts live in `alex/scripts/` (historical) and `pontus/scripts/` (his side); they're invoked **before** entering this pipeline.

---

## What this pipeline produces

When complete, `outputs/` contains everything needed to lock the report:

- **Sweep metrics** — AUC, Brier, ECE, top-1% precision per model (RF, HGBM, LightGBM, MLP, LogReg, etc.)
- **Tuned model results** — Optuna best-trial AUC + best params per model
- **Decomposition verdicts** — B1a (consensus alignment) and B1b (naive baseline) on v4 features
- **Rigor** — bootstrap CI on AUC, DeLong test, permutation importance, learning curves, complexity benchmark
- **Interpretability** — SHAP values for top-1% picks
- **Backtests** — flat-stake economic + capital-aware realistic + cost-floor × copycats sensitivity grid

---

## Why v4 (lineage from v3.5)

| Version | Features | What changed | Status |
|---|---|---|---|
| v1, v2, v3 | 16 → 65 | Iterative leakage fixes | Archived |
| v3.5 | 70 | Added 10 within-HF wallet aggregates. **AUC 0.899 RF, top-1% 100%, +23.7% N=1 ROI.** B1b naive baseline matches top-1%. | Locked in PR #13. Becomes the v4 baseline. |
| **v4** | **76** | v3.5 + Pontus's 6 Layer-6 wallet identity features (causal-bisected, joined per-row) | **Working target — this pipeline runs on v4.** |
| v5 (stretch) | 77 | v4 + `wallet_prior_win_rate_causal` (we'd write this) | Only build if v4 B1b still falsifies AND we have ~3 hr to spare |

The v3.5 results are the baseline that v4 is compared against. Don't re-run v3.5 — those numbers are locked.

---

## Stages at a glance

| # | Stage | Script(s) | Owner | Time |
|---|---|---|---|---|
| 0 | Pre-flight | — | Us | 5 min |
| 1 | Schema validation | `01_validate_schema.py` | Us | 5 min |
| 2 | Causality guard | `02_causality_guard.py` + Pontus's `_check_causal_joined.py` | Us | 5 min |
| 3 | EDA | (Pontus's deliverable) | **Pontus** | — |
| 4 | Supervised training (RF, HGBM, LightGBM, MLP, LogReg, DT, ExtraTrees, NB, PCA-LogReg, IsoForest) | `03_sweep.py` | Us | 4-6 hr |
| 5 | Unsupervised deep-dive (focused IsoForest analysis) | `04_iso_forest.py` | Us | 30 sec |
| 6 | Hyperparameter tuning (Optuna TPE, fresh study on 76 features) | `05_optuna_tuning.py` | Us | 2-3 hr |
| 7 | Decomposition + rigor + complexity + SHAP | `06_phase2_falsification.py`, `07_rigor_additions.py`, `08_complexity_benchmark.py`, `09_shap_top_picks.py` | Us | 3 hr |
| 8 | Economic backtests | `10_backtest.py`, `11_realistic_backtest.py`, `12_sensitivity_sweep.py` | Us | 2 min |

**Critical-path total: ~10-12 hours.** Optional stretch additions add ~5 hr.

Note: `03_sweep.py` already includes a TF/Keras MLP as one of its 8 models, so there's no separate MLP-training script. Stage 4 is a single command that produces all supervised + IsoForest predictions.

---

## Prerequisites

Must be in place before invoking:

1. **Pontus delivers `data/train_features_v4.parquet` and `data/test_features_v4.parquet`.**
   - Same row counts as v3.5 (1,114,003 train / 257,177 test)
   - Same row order as v3.5 (validated in Stage 1)
   - Same 70 v3.5 features + exactly 6 new wallet columns (no `n_tokentx`, no static `wallet_funded_by_cex`, no `wallet_prior_win_rate`)
2. **Pontus's reproducer script `pontus/scripts/_check_causal_joined.py` is on `main`.** We invoke it in Stage 2.
3. **`.venv` at `ML/report/.venv/` has all deps:** sklearn, pandas, optuna, lightgbm, tensorflow, shap.
4. **Phase 1 pressure tests pass on v3.5 already.** Stage 2 re-runs them on v4 to confirm wallet integration didn't introduce a regression.

If any prerequisite is missing, halt and resolve before starting.

---

## How to invoke

Stages 0-2 are interactive (validation gates). Stages 4-8 can be automated.

```bash
cd ML/report

# Stage 0: Pre-flight
git pull origin main
# Then download Pontus's v4 parquets (release tag TBD); place at:
# data/train_features_v4.parquet
# data/test_features_v4.parquet

# Stage 1: Schema validation (5 min)
.venv/bin/python alex/v4_final_ml_pipeline/scripts/01_validate_schema.py

# Stage 2: Causality guard (~5 min)
.venv/bin/python alex/v4_final_ml_pipeline/scripts/02_causality_guard.py
.venv/bin/python pontus/scripts/_check_causal_joined.py

# Stage 4: Supervised — runs all primary models (RF, HGBM, LightGBM, MLP, etc.)
.venv/bin/python alex/v4_final_ml_pipeline/scripts/03_sweep.py

# Stage 5: Unsupervised
.venv/bin/python alex/v4_final_ml_pipeline/scripts/04_iso_forest.py

# Stage 6: Optuna — backgrounded, parallel subprocesses, ~2-3 hr
.venv/bin/python alex/v4_final_ml_pipeline/scripts/05_optuna_tuning.py --n_trials 50

# Stage 7: Decomposition + rigor — parallel after Stage 4
.venv/bin/python alex/v4_final_ml_pipeline/scripts/06_phase2_falsification.py
.venv/bin/python alex/v4_final_ml_pipeline/scripts/07_rigor_additions.py
.venv/bin/python alex/v4_final_ml_pipeline/scripts/08_complexity_benchmark.py
.venv/bin/python alex/v4_final_ml_pipeline/scripts/09_shap_top_picks.py

# Stage 8: Backtests
.venv/bin/python alex/v4_final_ml_pipeline/scripts/10_backtest.py
.venv/bin/python alex/v4_final_ml_pipeline/scripts/11_realistic_backtest.py
.venv/bin/python alex/v4_final_ml_pipeline/scripts/12_sensitivity_sweep.py
```

(A future improvement: a master `run_v4_pipeline.sh` orchestrating this end-to-end with proper exit-code propagation.)

---

## Decision tree — what to do based on intermediate results

After Stage 2 (causality guard):
- All checks PASS → continue to Stage 4
- Any check FAILS → halt, ask Pontus to re-deliver. Do not patch the parquet ourselves.

After Stage 4 (supervised):
- MLP beats best tree on test AUC → MLP is a valid headline alternative; consider tuning it
- MLP doesn't beat best tree → expected; report MLP as additional model, headline stays with RF/HGBM
- LightGBM beats HGBM by > 0.5 pp → swap LightGBM in as the boosting representative

After Stage 7.1 (B1b on v4):
- B1b STILL FAILS (naive baseline matches augmented model on top-1%) → the asymmetric-info finding stays falsified even with wallet identity features. Consider building v5 (`wallet_prior_win_rate_causal`) if time allows; otherwise lock the negative finding.
- B1b PASSES on v4 (model AUC > naive AUC by > 0.05) → asymmetric-info narrative partially recovers. Update D-001 / D-034 accordingly. **Major decision point.**

After Stage 7.2/7.3 (rigor):
- AUC bootstrap CIs overlap heavily across top models → caveat "RF is best" claim with CI overlap. DeLong test is the formal claim.
- Learning curves still rising → caveat that more data could help.
- Complexity table shows MLP is much slower for marginal AUC gain → defensible reason to recommend tree models for deployment.

After Stage 8 (backtests):
- v4 ROI > v3.5 ROI by > 5pp → wallet features add deployment-realistic value
- v4 ROI ≤ v3.5 ROI → wallet features improve classification but not strategy economics. Report both.

---

## Where outputs land

```
data/
  train_features_v4.parquet          # Pontus delivers
  test_features_v4.parquet           # Pontus delivers
  feature_cols.json                  # Updated to 76 cols by Stage 1

.scratch/
  backtest/preds_<model>.npz         # raw, calibrated, OOF — Stage 4
  pressure_tests/phase{1,2}_v4.json  # Stage 2 + 7.1 results

outputs/
  sweep_idea1/<model>/               # Stage 4 metrics + per-market + importance
  rigor/
    optuna/<model>/                  # Stage 6 study + best params + comparison
    auc_ci.json                      # Stage 7.2 bootstrap
    deLong_pairwise.json             # Stage 7.2 hypothesis tests
    perm_importance_<model>.json     # Stage 7.2
    learning_curve.png               # Stage 7.2
    complexity_benchmark.{csv,png}   # Stage 7.3
    shap/                            # Stage 7.4
  backtest/                          # Stage 8 economic
    realistic/                       # Stage 8 capital-aware
    sensitivity/                     # Stage 8 cost × copycats
  pressure_tests/phase2_v4_results.json  # Stage 7.1
```

---

## What's "done"

The pipeline is complete when all of the following are true:

1. Schema validation + causality guard PASS on v4
2. Stage 4: predictions for all primary models (RF, HGBM, LightGBM, MLP) saved as npz
3. Stage 6: Optuna study completed for at least RF + HGBM (50 trials each)
4. Stage 7.1: B1a / B1b / A1 verdicts written
5. Stage 7.2: bootstrap CI + DeLong + permutation importance + learning curves all written
6. Stage 7.3: complexity_benchmark.csv has a row per model
7. Stage 7.4: SHAP top-picks plots written
8. Stage 8: realistic backtest + sensitivity sweep regenerated against v4 preds
9. New design decisions logged in `../notes/design-decisions.md` (D-038 onward)

After that we have the complete numerical content for the report. Drafting is downstream.

---

## Out of scope for this pipeline

- **Feature engineering** — happens upstream (`alex/scripts/06b_engineer_features.py` for v3.5; Pontus's join for v4)
- **Report writing** (Ethical Consideration, Conceptual Framework, LLM Usage Disclosure, Abstract, References) — handled separately
- **Cohort changes** — v4 uses the same 75-market split as v3.5
- **New wallet feature engineering on our side** — Pontus owns Layer-6
- **EDA** — Pontus owns this entirely
- **Re-running v3.5 numbers** — locked in PR #13, become the comparison baseline
- **Strict-cohort training** (Pontus's 4-train / 1-val / 7-test) — only optional robustness check

---

## Maintenance contract

When the pipeline changes, update both files:

| Change | Update |
|---|---|
| Add a new stage | This README's stage table + `pipeline.md` |
| Change a stage's wall time materially | This README + `pipeline.md` |
| Add a new script | `pipeline.md` script tables + `../scripts/README.md` |
| Add a new output directory | `pipeline.md` "Where outputs land" + `../outputs/README.md` |
| Decision rule changes | This README's decision tree |

When the pipeline runs, append a new entry to `../notes/design-decisions.md` summarising the v4 outcome (D-038 onwards).
