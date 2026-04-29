# Outputs index

What lives in each subdirectory and which script produced it. ~5.6 MB total tracked. Plots, metrics JSON, CSVs.

For *what the numbers mean*, see [`../notes/alex-approach.md`](../notes/alex-approach.md). This file is navigational only.

**Note on output paths:** the v3.5 baseline scripts (`scripts/`) and the v4 pipeline scripts (`v4_final_ml_pipeline/scripts/`) write to the **same output paths** by design — re-running the v4 pipeline overwrites the v3.5 baseline outputs. If you need to preserve the v3.5 numbers, move them under `outputs/_v3.5_archive/` before running v4.

---

## `backtest/` — economic backtest (`10_backtest.py`)

Flat $100/trade, no capital constraint. The "raw economic" view.

| File | What it is |
|---|---|
| `summary.json` | Per-model × per-strategy table: n_trades, hit_rate, total_pnl, mean_pnl_per_trade, sharpe, mean_edge, median_cost. Headline reference. |
| `pnl_curves.png` | Cumulative PnL by strategy for the best model (Random Forest) |
| `edge_distribution.png` | Edge histogram + hit rate by edge bucket |
| `calibration.png` | Calibration diagram (predicted vs empirical hit rate) for the best model |
| `per_market_pnl.csv` | Per-market PnL under general +EV rule, sorted by total PnL |
| `home_run_picks.parquet` | The 4 trades that passed the home-run filter (edge>0.20 + late + cheap). Hit rate: 0% — confirms no asymmetric-info pattern. |

### `backtest/realistic/` — capital-aware backtest (11_realistic_backtest.py)

| File | What it is |
|---|---|
| `summary.json` | All 9 sensitivity-grid scenarios per model × strategy ($1K/$10K/$100K bankroll × 1%/5%/10% bet). Final capital, ROI, max drawdown, n_executed |
| `sensitivity.csv` | Same as summary.json but flat tabular for easy slicing |
| `capital_curves.png` | Equity curves for $10K bankroll, 5% bet, log-y |

### `backtest/sensitivity/` — Phase 3 sensitivity (12_sensitivity_sweep.py)

| File | What it is |
|---|---|
| `sweep.csv` | 90 cells: 3 (model, strategy) × 5 cost_floors × 6 copycat-N. ROI, n_executed, final_capital, max_drawdown |
| `summary.json` | Best/worst/headline cell per (model, strategy) |
| `heatmap_hist_gbm_general_ev.png` | ROI heatmap, cost_floor (rows) × copycats (cols). The headline plot. |
| `heatmap_hist_gbm_top5pct_edge.png` | Same for HistGBM top5pct_edge strategy |
| `heatmap_random_forest_general_ev.png` | Same for RF general_ev. Non-monotonic in N (peaks at N=5). |

---

## `sweep_idea1/` — supervised sweep (`03_sweep.py` in v4, was `07_sweep.py` in v3.5)

One subdirectory per model. Standard contents: `metrics.json` (AUC, Brier, ECE), `per_market_test.json` (per-market AUC), `feature_importance.json` (where applicable).

Models: `decision_tree`, `extra_trees`, `gaussian_nb`, `hist_gbm`, `iso_forest`, `logreg_l1`, `logreg_l2`, `pca_logreg`, `random_forest`, **`mlp_keras`** (TF/Keras MLP — written by the sweep, not a separate script). v4 also includes **`lightgbm`**.

**Best on v3.5: random_forest (test AUC 0.899).**

`iso_forest/test_anomaly_scores.parquet` is the unsupervised arm output from the in-sweep IsoForest. Anomaly score corr with `bet_correct` is -0.007 — null result.

---

## `rigor/` — Stage 6 + Stage 7 outputs (v4 only)

Produced by v4 pipeline scripts `05_optuna_tuning.py`, `07_rigor_additions.py`, `08_complexity_benchmark.py`, `09_shap_top_picks.py`. Empty until v4 runs.

| Subdirectory / file | What it is | Producer |
|---|---|---|
| `optuna/<model>/study.db` | SQLite study (resumable) | `05_optuna_tuning.py` |
| `optuna/<model>/best_params.json` | Best hyperparameters per model | `05_optuna_tuning.py` |
| `optuna/<model>/comparison.json` | Tuned vs default test AUC delta | `05_optuna_tuning.py` |
| `auc_ci.json` | Bootstrap 95% CI on test AUC per model | `07_rigor_additions.py` |
| `deLong_pairwise.json` | Pairwise hypothesis tests for AUC differences | `07_rigor_additions.py` |
| `perm_importance_<model>.json` | Permutation importance (modern feature attribution) | `07_rigor_additions.py` |
| `learning_curve.png` | AUC vs train fraction per model | `07_rigor_additions.py` |
| `complexity_benchmark.{csv,png}` | Fit time + predict time + parameter count per model. Required by `exam.md:47` | `08_complexity_benchmark.py` |
| `shap/values.parquet` | SHAP values per top-1% pick | `09_shap_top_picks.py` |
| `shap/summary.png`, `shap/top_picks.png` | Global + per-pick SHAP plots | `09_shap_top_picks.py` |
| `shap/feature_ranking.json` | Features sorted by mean(|SHAP|) on top-1% | `09_shap_top_picks.py` |

---

## `leakage_diagnostic/` — RF ablation tests (`08_leakage_diagnostic.py` — v3.5 only, not in v4)

| File | What it is |
|---|---|
| `check1_single_feature/metrics.json` | RF with one feature dropped at a time. Confirms no single-feature leak |
| `check2_drop_kyle/metrics.json` | RF without Kyle-lambda family |
| `check3_drop_suspect_family/metrics.json` | RF without 24h-window features |
| `summary.md` | Verdict: bimodal per-market AUC is structural, not leakage |

---

## `findings_report/` — static HTML dashboard

Standalone summary page (open `index.html`) with key plots: cohort structure, per-market AUC, top-k precision, cutoff sweep, realistic ROI. Generated as a one-shot artifact for sharing.

---

## `baselines/` — pre-v3.5 baselines (06_baseline_idea1.py — deprecated)

| Subdirectory | What it is | Status |
|---|---|---|
| `idea1_v1/` | LogReg baseline with `log_trade_value_usd` (had a leak) | Historical / audit trail |
| `idea1_v2/` | LogReg baseline after dropping the leaky feature | Historical / audit trail |
| `logreg/`, `sweep/` | Earlier experiments | Pre-pivot reference |

Don't cite these as current results — see `sweep_idea1/` for the v3.5 numbers.

---

## `pressure_tests/` — Phase 1 + Phase 2 verdicts

| File | What it is | Producer |
|---|---|---|
| `phase1_results.json` | 9 leakage / integrity check verdicts (PASS / NOTE) | v3.5 `pressure_tests/phase1_quick_wins.py` or v4 `02_causality_guard.py` |
| `phase2_v4_results.json` | B1a (consensus decomposition), B1b (naive baseline AUC), A1 (SELL semantics) verdicts on v4 | v4 `06_phase2_falsification.py` |
| `plots/b1a_decomposition.png` | Consensus alignment of top-1% picks per model | v4 `06_phase2_falsification.py` |

---

## `cohort_inventory/` — cohort discovery (`build_cohorts.py` — upstream of both pipelines)

| File | What it is |
|---|---|
| `markets_inventory.parquet` | All HF markets matching the canonical regex, with metadata |
| `markets_inventory.csv` | Same as above, CSV form for spot-checking |
| `keyword_matches.csv` | Question-text matches per keyword pattern (used to build the canonical filter) |

---

## `data_sanity/` — one-off data checks

| File | What it is |
|---|---|
| `report.md` | Free-form sanity-check report (summary stats, anomalies) |
| `feature_correlations.png` | Pairwise feature correlation heatmap |
| `target_rate_per_market.png` | `bet_correct` rate per market (range, distribution) |

---

## `investigations/` — deep-dives on specific concerns

| Subdirectory | Investigation | Conclusion |
|---|---|---|
| `naive_market/` | Does naive market price beat the model? Per-subgroup analysis. | Simpson's-paradox artifact — naive 0.63 ROC was misleading |
| `residual_edge/` | Residual edge analysis after market-implied prob. Plan §5.2 RQ1b. | Signal exists but modest |

---

## When to update this file

- Add a row when a new output directory appears
- Move a directory to "Historical / pre-pivot" when it's superseded
- Update file lists when scripts add or rename outputs
