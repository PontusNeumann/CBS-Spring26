# Residual-edge analysis (RQ1b per plan §5.2)

_Does the trained model's `p_hat` carry predictive information BEYOND what `market_implied_prob` already encodes? Evaluated via calibration, edge distribution, and residual ROC._

## Model: LogisticRegression(class_weight='balanced') on leakage-free features (32 features after all P0 pruning — see `data-pipeline-issues.md`).


## train — edge = p_hat_logreg − market_implied_prob

- n valid: 202,082 / 202,082
- edge mean: -0.0158
- edge std:  0.3591
- |edge| mean: 0.3303
- edge percentiles (1/5/25/50/75/95/99): -0.564  -0.508  -0.355  -0.043  +0.345  +0.484  +0.547

### Predictive tests (does edge predict bet_correct?)
- ROC of raw edge: 0.5541
- ROC of |edge|:   0.4844
- Partial correlation (edge ⊥ market_implied_prob) with bet_correct: +0.1012
- ROC of edge RESIDUAL (after linearly subtracting market_implied_prob): 0.5513

### Per-subgroup edge direction and predictive value

| Subgroup | n | mean edge | |edge| mean | ROC (edge → bc) | bc_rate |
|---|---:|---:|---:|---:|---:|
| `BUY_idx0` | 35,857 | +0.2585 | 0.3093 | 0.1371 | 0.705 |
| `BUY_idx1` | 41,234 | -0.2731 | 0.3345 | 0.0605 | 0.401 |
| `SELL_idx0` | 57,175 | +0.2862 | 0.3367 | 0.8966 | 0.365 |
| `SELL_idx1` | 67,816 | -0.2591 | 0.3335 | 0.9416 | 0.619 |

## val — edge = p_hat_logreg − market_implied_prob

- n valid: 13,154 / 13,154
- edge mean: -0.0088
- edge std:  0.4017
- |edge| mean: 0.3673
- edge percentiles (1/5/25/50/75/95/99): -0.532  -0.497  -0.401  -0.113  +0.400  +0.532  +0.889

### Predictive tests (does edge predict bet_correct?)
- ROC of raw edge: 0.6149
- ROC of |edge|:   0.4825
- Partial correlation (edge ⊥ market_implied_prob) with bet_correct: +0.1051
- ROC of edge RESIDUAL (after linearly subtracting market_implied_prob): 0.5914

### Per-subgroup edge direction and predictive value

| Subgroup | n | mean edge | |edge| mean | ROC (edge → bc) | bc_rate |
|---|---:|---:|---:|---:|---:|
| `BUY_idx0` | 2,296 | +0.3934 | 0.3957 | nan | 0.000 |
| `BUY_idx1` | 2,985 | -0.3308 | 0.3324 | nan | 1.000 |
| `SELL_idx0` | 3,723 | +0.3873 | 0.3897 | nan | 1.000 |
| `SELL_idx1` | 4,150 | -0.3552 | 0.3566 | nan | 0.000 |

## test — edge = p_hat_logreg − market_implied_prob

- n valid: 13,414 / 13,414
- edge mean: -0.0888
- edge std:  0.4022
- |edge| mean: 0.3999
- edge percentiles (1/5/25/50/75/95/99): -0.545  -0.526  -0.450  -0.312  +0.354  +0.484  +0.532

### Predictive tests (does edge predict bet_correct?)
- ROC of raw edge: 0.3813
- ROC of |edge|:   0.4069
- Partial correlation (edge ⊥ market_implied_prob) with bet_correct: +0.0595
- ROC of edge RESIDUAL (after linearly subtracting market_implied_prob): 0.5339

### Per-subgroup edge direction and predictive value

| Subgroup | n | mean edge | |edge| mean | ROC (edge → bc) | bc_rate |
|---|---:|---:|---:|---:|---:|
| `BUY_idx0` | 3,829 | +0.3831 | 0.3833 | nan | 0.000 |
| `BUY_idx1` | 5,113 | -0.4017 | 0.4018 | nan | 1.000 |
| `SELL_idx0` | 1,812 | +0.3417 | 0.3417 | nan | 1.000 |
| `SELL_idx1` | 2,660 | -0.4599 | 0.4599 | nan | 0.000 |

## Plots

- `calibration_{train,val,test}.png` — left panel: LogReg p_hat; right panel: market_implied_prob. Per-subgroup calibration curves.
- `edge_{train,val,test}.png` — edge distribution + edge vs realised bet_correct rate (binned).