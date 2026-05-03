# v5 tuning summary

## Best configs
### random_forest
- Best OOF AUC: **0.81679**
- Tuned test AUC: **0.77509**
- Default test AUC: 0.89872
- **Δ test AUC: -0.12363**
- Best params: `{'n_estimators': 300, 'max_depth': 15, 'min_samples_leaf': 188, 'max_features': 0.3}`

### mlp_sklearn
- Best OOF AUC: **0.88757**
- Tuned test AUC: **0.80473**
- Default test AUC: 0.80205
- **Δ test AUC: +0.00267**
- Best params: `{'arch_idx': 5, 'activation': 'relu', 'alpha': 0.0035974197956574233, 'learning_rate_init': 0.003085473344630512, 'batch_size': 2048}`

## Realistic-backtest deltas ($10K, 5%, no copycats, headline scenario)

| strategy | RF baseline | RF tuned | Δ | MLP baseline | MLP tuned | Δ |
|---|---:|---:|---:|---:|---:|---:|
| `phat_gt_0.99` | +2.2% | +0.4% | -1.8pp | +11.9% | +7.0% | -4.9pp |
| `phat_gt_0.95` | +10.6% | +6.6% | -4.0pp | +17.7% | +6.8% | -10.9pp |
| `phat_gt_0.9` | +14.0% | +12.7% | -1.3pp | +19.6% | +6.6% | -13.0pp |
| `top1pct_phat` | +4.4% | +1.7% | -2.7pp | +6.1% | +1.8% | -4.3pp |
| `top1pct_edge` | -3.6% | -33.5% | -29.9pp | -14.6% | -11.8% | +2.8pp |
| `top5pct_edge` | -15.9% | -53.7% | -37.8pp | -23.4% | +19.6% | +43.0pp |
| `general_ev` | -75.8% | -75.6% | +0.2pp | -48.6% | -3.0% | +45.6pp |
| `general_ev_late` | -83.7% | -85.6% | -1.9pp | -7.4% | +2.9% | +10.3pp |
| `general_ev_cheap` | -87.7% | -83.1% | +4.6pp | -76.4% | -16.6% | +59.8pp |
| `home_run` | -0.0% | -3.1% | -3.1pp | -0.6% | -0.1% | +0.5pp |
