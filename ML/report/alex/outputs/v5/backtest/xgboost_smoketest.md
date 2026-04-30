# XGBoost smoke test

## AUC + top-1% precision (cleaned 64-feat schema, test cohort)

| model | raw AUC | cal AUC | top-1% precision | per-market AUC |
|---|---:|---:|---:|---|
| hist_gbm | 0.8928 | 0.8869 | 0.999 | [0.00, 1.00] |
| lightgbm | 0.8386 | 0.8106 | 1.000 | [0.02, 1.00] |
| random_forest | 0.8987 | 0.8877 | 1.000 | [0.03, 1.00] |
| mlp_sklearn | 0.8021 | 0.8020 | 1.000 | [0.06, 1.00] |
| xgboost | 0.7773 | 0.7773 | 1.000 | [0.21, 1.00] |

## Realistic-backtest ROI ($10K, 5% bet, no copycats, 31-day cross-regime test)

| strategy | hist_gbm | lightgbm | random_forest | mlp_sklearn | xgboost |
|--- | ---: | ---: | ---: | ---: | ---:|
| `phat_gt_0.99` | +0.0% | +0.0% | +2.2% | +11.9% | +4.3% |
| `phat_gt_0.95` | +0.0% | +0.0% | +10.6% | +17.7% | +11.4% |
| `phat_gt_0.9` | +0.0% | +0.0% | +14.0% | +19.6% | +13.0% |
| `top1pct_phat` | +4.0% | +4.4% | +4.4% | +6.1% | +1.3% |
| `top1pct_edge` | +2.0% | -17.0% | -3.6% | -14.6% | -28.5% |
| `top5pct_edge` | +27.3% | -14.9% | -15.9% | -23.4% | -52.4% |
| `general_ev` | -76.9% | -74.6% | -75.8% | -48.6% | -57.8% |
| `general_ev_late` | -85.8% | -85.8% | -83.7% | -7.4% | -23.4% |
| `general_ev_cheap` | -99.5% | -87.7% | -87.7% | -76.4% | -68.5% |
| `home_run` | +0.0% | +0.0% | -0.0% | -0.6% | -1.8% |

## Caveats

- XGBoost trained 2026-04-30 on the 64-feature cleaned schema (D-042 cohort-flip features removed).
- The hist_gbm / lightgbm / random_forest cached preds in `.scratch/backtest/` are dated Apr 27 — **before** the schema cleanup. Their raw AUCs are inflated by cohort-flip features. XGBoost is being compared against a stronger-than-real baseline.
- mlp_sklearn cached preds are from Apr 29 (post-cleanup, fair comparison).
- No hyperparameter tuning. Defaults: n_estimators=400, max_depth=8, learning_rate=0.05, min_child_weight=200, tree_method=hist.
