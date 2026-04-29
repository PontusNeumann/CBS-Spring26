# Scripts index — v3.5 baseline (HISTORICAL)

> ⚠️ **These scripts are the v3.5 baseline that produced [PR #13](https://github.com/PontusNeumann/CBS-Spring26/pull/13).**
> They're kept here as the audit trail for the locked baseline (AUC 0.899, top-1% 100%, +23.7% N=1 ROI).
> **For active v4 work, use [`../v4_final_ml_pipeline/scripts/`](../v4_final_ml_pipeline/scripts/) instead** — that's the cleaned + renumbered version with shared utilities in `_common.py` and 4 new rigor/interpretability scripts.

The `scripts/` ↔ `v4_final_ml_pipeline/scripts/` mapping:

| v3.5 baseline (here) | v4 pipeline (active) | Change |
|---|---|---|
| `06b_engineer_features.py` | _(upstream — same script)_ | Feature engineering happens BEFORE the v4 pipeline. Pontus joins his wallet features to its output. |
| `07_sweep.py` | `03_sweep.py` | LightGBM added |
| `08_leakage_diagnostic.py` | _(not in v4)_ | One-off RF ablation; verdict locked |
| `09_iso_forest.py` | `04_iso_forest.py` | Unchanged |
| `10_backtest.py` | `10_backtest.py` | Same name — uses `_common.py` |
| `11_realistic_backtest.py` | `11_realistic_backtest.py` | Same name — uses `_common.py` |
| `12_sensitivity_sweep.py` | `12_sensitivity_sweep.py` | Same name — replaced importlib hack with `_common` import |
| `13_optuna_tuning.py` | `05_optuna_tuning.py` | Renumbered |
| `pressure_tests/phase1_quick_wins.py` | `02_causality_guard.py` | Promoted from sub-folder |
| `pressure_tests/phase2_falsification.py` | `06_phase2_falsification.py` | Promoted from sub-folder |
| `pressure_tests/fix_cost_and_rerun.py` | _(dropped — historical only)_ | One-off cost-bug diagnostic |
| `_backtest_worker.py`, `_optuna_worker.py` | Same names | Unchanged |

For the v4 pipeline DAG, prerequisites, and decision tree, see [`../v4_final_ml_pipeline/README.md`](../v4_final_ml_pipeline/README.md).

For *why* a script exists or what its results mean, see [`../notes/alex-approach.md`](../notes/alex-approach.md). The rest of this file is a navigational reference for the historical scripts.

---

## v3.5 baseline DAG (historical)

```
build_cohorts.py
       │  produces train.parquet, test.parquet, markets_subset.parquet
       ▼
06b_engineer_features.py
       │  produces train_features.parquet, test_features.parquet, feature_cols.json
       ▼
       ├──► 07_sweep.py ───────► outputs/sweep_idea1/<model>/metrics.json
       │
       ├──► 08_leakage_diagnostic.py ──► outputs/leakage_diagnostic/
       │
       ├──► 09_iso_forest.py ──────────► outputs/sweep_idea1/iso_forest/
       │
       ├──► 10_backtest.py ────────────► outputs/backtest/
       │      ▲ spawns _backtest_worker.py × 3 (logreg, RF, hist_gbm)
       │      ▲ caches preds in .scratch/backtest/preds_*.npz
       │
       ├──► 11_realistic_backtest.py ──► outputs/backtest/realistic/
       │      ▲ reads cached preds from 10_backtest's run
       │
       └──► 12_sensitivity_sweep.py ───► outputs/backtest/sensitivity/
              ▲ reads cached preds, sweeps cost_floor × copycat-N

pressure_tests/
  ├─ phase1_quick_wins.py        # 9 leakage / integrity checks
  ├─ phase2_falsification.py     # B1a/B1b consensus tests + A1 SELL semantics
  └─ fix_cost_and_rerun.py       # Cost-bug discovery + impact diagnostic (historical)
```

---

## Active scripts (v3.5 baseline)

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| [`build_cohorts.py`](build_cohorts.py) | Filter HF dataset by canonical Iran-strike / Iran-ceasefire question regex; apply pre-event timestamp filter; drop low-volume markets | HF `SII-WANGZJ/Polymarket_data` | `data/markets_subset.parquet`, `data/train.parquet`, `data/test.parquet` |
| [`06b_engineer_features.py`](06b_engineer_features.py) | v3.5 70-feature engineer. Trade-level, time-to-deadline, rolling stats (closed='left'), wallet position, market context, Kyle lambda, microstructure | `data/train.parquet`, `data/test.parquet` | `data/train_features.parquet`, `data/test_features.parquet`, `data/feature_cols.json` |
| [`07_sweep.py`](07_sweep.py) | 8-model supervised sweep + IsoForest. 5-fold GroupKFold CV on `market_id`. **Includes a TF/Keras MLP as model #7** | `data/*_features.parquet` | `outputs/sweep_idea1/<model>/{metrics.json, per_market_test.json, feature_importance.json}` |
| [`08_leakage_diagnostic.py`](08_leakage_diagnostic.py) | RF ablation tests: drop kyle_lambda, drop 24h-window features, drop suspect family. Confirmed bimodal per-market AUC is structural, not leakage | `data/*_features.parquet` | `outputs/leakage_diagnostic/{check1,check2,check3}/metrics.json` + `summary.md` |
| [`09_iso_forest.py`](09_iso_forest.py) | Standalone IsoForest with anomaly score scored against `bet_correct`. **Null result** (corr -0.007) | `data/*_features.parquet` | `outputs/sweep_idea1/iso_forest/test_anomaly_scores.parquet` |
| [`10_backtest.py`](10_backtest.py) | Flat-stake economic backtest. Spawns 3 workers in parallel, computes corrected `pre_yes_price`, runs strategy masks, generates plots. Uses cached preds if present (set `RETRAIN=1` to force) | `data/*_features.parquet`, `.scratch/backtest/preds_*.npz` | `outputs/backtest/{summary.json, *.png, per_market_pnl.csv, home_run_picks.parquet}` |
| [`11_realistic_backtest.py`](11_realistic_backtest.py) | Capital-aware execution: bankroll, per-trade caps, concentration limit, gas, slippage, copycat liquidity sharing. 9-cell sensitivity grid | `data/*_features.parquet`, `.scratch/backtest/preds_*.npz` | `outputs/backtest/realistic/{summary.json, sensitivity.csv, capital_curves.png}` |
| [`12_sensitivity_sweep.py`](12_sensitivity_sweep.py) | Phase 3 sensitivity. Sweeps 5 cost_floors × 6 copycat-N values × 3 (model, strategy) targets | `data/*_features.parquet`, `.scratch/backtest/preds_*.npz` | `outputs/backtest/sensitivity/{sweep.csv, summary.json, heatmap_*.png}` |
| [`13_optuna_tuning.py`](13_optuna_tuning.py) | Optuna TPE tuning for RF + HGBM, 50 trials each, 5-fold GroupKFold. SQLite-backed, resumable | `data/*_features.parquet` | `outputs/rigor/optuna/<model>/{study.db, study_history.csv, best_params.json}` |
| [`_backtest_worker.py`](_backtest_worker.py) | Worker for `10_backtest.py`. Trains one model with 5-fold CV, isotonic on OOF, final fit. Deterministic with `random_state=42` | `data/*_features.parquet` | `.scratch/backtest/preds_<model>.npz` + `.importance.json` |
| [`_optuna_worker.py`](_optuna_worker.py) | Worker for `13_optuna_tuning.py`. One model per process | `data/*_features.parquet` | `outputs/rigor/optuna/<model>/study.db` |
| [`dashboard.py`](dashboard.py) | Live HTTP dashboard on port 9876. Reads `outputs/`, renders sweep results, per-market AUC, calibration plots in real time | `outputs/` (read-only) | HTTP only |

---

## Pressure tests

| Script | Purpose |
|---|---|
| [`pressure_tests/phase1_quick_wins.py`](pressure_tests/phase1_quick_wins.py) | 9 leakage / data-integrity checks (C1, D1-D4, A3, C2, F1, F2). All PASS or NOTE on v3.5. |
| [`pressure_tests/phase2_falsification.py`](pressure_tests/phase2_falsification.py) | B1a (consensus-vs-contrarian decomposition of top-1% picks) + B1b (naive consensus baseline) + A1 (SELL semantics). Uses corrected `pre_yes_price`. **B1b FAILED on v3.5** — naive baseline matches model on top-1%. |
| [`pressure_tests/fix_cost_and_rerun.py`](pressure_tests/fix_cost_and_rerun.py) | One-off diagnostic that proved the per-token-price bug (D-029) + measured impact (45× → +14% under N=10 realism). Historical reference; the fix is now in `10_backtest.py` and `11_realistic_backtest.py`. Not in v4 pipeline. |

---

## Deprecated / superseded

| Script | Status | Why kept |
|---|---|---|
| [`06_baseline_idea1.py`](06_baseline_idea1.py) | Deprecated (v1 / v2 LogReg baseline) | Audit trail — used to produce `outputs/baselines/idea1_v1/` and `idea1_v2/`. Has the `log_trade_value_usd` leak that was caught and fixed in v3 |
| [`01_data_sanity.py`](01_data_sanity.py), [`02_baseline_logreg.py`](02_baseline_logreg.py), [`03_baselines_sweep.py`](03_baselines_sweep.py), [`04_naive_market_investigation.py`](04_naive_market_investigation.py), [`05_residual_edge.py`](05_residual_edge.py) | Pre-v3 exploratory scripts | Audit trail for the early baseline + investigation phase |

---

## Conventions (apply to v3.5 and v4 pipelines alike)

- **Random seed:** `RANDOM_SEED = 42` everywhere. Workers + sweep + backtest are all deterministic.
- **CV:** 5-fold `GroupKFold` on `market_id`. No within-market splits anywhere.
- **Calibration:** Isotonic regression fit on OOF predictions, then applied to test.
- **Cost floor:** 0.05 (realism, see `_common.COST_FLOOR_DEFAULT`). 0.001 (raw economic math) for `10_backtest.py`.
- **Pre-event filter:** All trades must have `timestamp < event_time`. Verified in pressure-test C1.
- **Wallet aggregates (ours):** Only within-HF (`cumcount` + `shift(1)` on `maker` / `taker` columns). Pontus's Layer-6 on-chain enrichment comes joined to v4 parquets — we don't reproduce it.

---

## When to update this file

This README is for the historical v3.5 baseline. Don't add new active scripts here — add them to `../v4_final_ml_pipeline/scripts/` and update `../v4_final_ml_pipeline/pipeline.md`.

Update this file only when:
- A historical script is moved to "Deprecated / superseded" (don't delete from disk — keep audit trail)
- The mapping table at the top changes because the v4 pipeline restructures
