# v5 ML pipeline — realistic backtest expansion (RF + MLP + XGBoost focus)

Built on top of v4 (Pontus's pre-consolidation structure). Kept separate from
`v4_final_ml_pipeline/` so it does not collide with the recent main-branch
refactor that moved data under `data/archive/alex/` and introduced
`load_modeling_dataset()` in `_common.py`.

## What's new vs v4

1. **`13_naive_baseline_backtest.py`** — runs the realistic backtest engine on
   the naive consensus predictor (Phase 2 B1b, AUC 0.844). Same constraints,
   same strategy menu, drop-in benchmark for "what does a free heuristic earn?"
2. **`14_overview_chart.py`** — single shareable PNG that combines all 5 models
   and naive baseline into one ROI heatmap. Headline figure for the report.
3. **`11_realistic_backtest.py`** — three changes:
   - Live HTML progress viewer at `outputs/v5/backtest/realistic/progress.html`,
     auto-refreshes every 5s, shows headline pivot + last 30 cells.
   - `mlp_sklearn` added to the model list (5 models total).
   - Sensitivity grid now sweeps `liquidity_scaler ∈ {1.0, 0.10}` so every
     scenario is reported under both no-copycats (default) and 10× copycat
     stress.
   - Strategy menu expanded to 10: + `phat_gt_0.95`, `phat_gt_0.99`,
     `general_ev_cheap` (`cost < 0.30`), `general_ev_late`
     (`time_to_deadline < 1d`).
4. **`_backtest_worker.py`** — added factories for `mlp_sklearn`,
   `decision_tree`, `logreg_l1`, `pca_logreg` (PCA k=16), and `xgboost`
   (`tree_method=hist`, n_estimators=400, max_depth=8, learning_rate=0.05,
   min_child_weight=200, untuned defaults). v4 only had logreg_l2 / RF /
   hist_gbm / lightgbm.
5. **`_common.py`** — `LIQUIDITY_SCALER_DEFAULT` flipped from `0.10` to `1.0`
   (no copycats is the more honest realistic default). 0.10 retained as a
   stress test inside `fill_share_grid`.

All path constants (`ROOT`, `DATA`, `EXPECTED_N_FEATURES = 64`) match v4 as it
existed at commit `5a0beab`. Outputs are namespaced under `outputs/v5/` so
nothing collides with v4's `outputs/backtest/` (or its archived counterparts on
main).

## Headline numbers — $10K, 5% bet, no copycats, 31-day cross-regime test

| model       | best strategy        | 31d ROI | per-market test AUC |
|-------------|----------------------|--------:|--------------------:|
| **MLP**     | `phat_gt_0.9`        | **+20%** | [0.06, 1.00] |
| naive       | `phat_gt_0.9`        | +17% | — |
| RF          | `phat_gt_0.9`        | +14% | [0.03, 1.00]¹ |
| **XGBoost** | `phat_gt_0.9`        | **+13%** | **[0.21, 1.00]** ← tightest range |
| hist_gbm    | `top5pct_edge`       | +27% (single 601-trade cell, fragile) | [0.00, 1.00]¹ |
| lightgbm    | `top1pct_phat`       | +4% | [0.02, 1.00]¹ |
| logreg_l2   | `top1pct_phat`       | -2% | — |

¹ uses Apr 27 stale 80-feature cached preds — inflated raw AUC. XGBoost and MLP
are on the cleaned 64-feature schema, fair comparison; the others should be
retrained for an apples-to-apples baseline.

**Why XGBoost is interesting**: the only model with no per-market AUC near
random (worst market 0.21 vs others 0.00-0.06). Tightest cross-market stability
of any tree model — the most direct evidence we have that some signal genuinely
transfers across the 10 ceasefire markets, not just averages over a few
favourable cohorts.

See [`outputs/v5/backtest/overview.png`](../outputs/v5/backtest/overview.png) for
the full 7-model × 10-strategy grid.

## Run order

```bash
cd ML/report/alex
# 1. (skip if cached) train predictions for all 6 models
for m in logreg_l2 random_forest hist_gbm lightgbm xgboost mlp_sklearn; do
  python v5_final_ml_pipeline/scripts/_backtest_worker.py \
    --model $m --out .scratch/backtest/preds_$m.npz
done

# 2. raw economic backtest
python v5_final_ml_pipeline/scripts/10_backtest.py

# 3. capital-aware (live HTML viewer at outputs/v5/backtest/realistic/progress.html)
python v5_final_ml_pipeline/scripts/11_realistic_backtest.py

# 4. naive consensus baseline
python v5_final_ml_pipeline/scripts/13_naive_baseline_backtest.py

# 5. cost_floor x copycats sensitivity
python v5_final_ml_pipeline/scripts/12_sensitivity_sweep.py

# 6. one-shot overview chart
python v5_final_ml_pipeline/scripts/14_overview_chart.py
```

## Data dependency

v5 still consumes the v4 schema parquets:
- `data/train_features_v4.parquet`
- `data/test_features_v4.parquet`
- `data/feature_cols.json` (64 cleaned features)
- `data/test.parquet`, `data/markets_subset.parquet`

It does **not** read Pontus's new `data/consolidated_modeling_data.parquet`. To
run v5 against the 73-feature consolidated dataset, the worker + 11/13 would
need to be ported to `_common.load_modeling_dataset()`. That port is
intentionally not done in this PR — keep v5 frozen against the schema it was
validated on.
