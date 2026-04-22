# Naive-market investigation

_Script: `alex/scripts/04_naive_market_investigation.py`. Purpose: determine whether the naive-market baseline's test ROC 0.63 is a real signal or a Simpson's-paradox artifact of the subgroup structure._

## Top-level naive-market ROC (recap)

| Split | n | bc_rate | p_hat mean | ROC |
|---|---:|---:|---:|---:|
| train | 202,082 | 0.518 | 0.516 | 0.4547 |
| val | 13,154 | 0.510 | 0.542 | 0.4300 |
| test | 13,414 | 0.516 | 0.575 | 0.6315 |

## train — side × outcomeIndex subgroup breakdown

| Subgroup | Interpretation | n | bc_rate | p_hat mean | p_hat std | ROC (naive) |
|---|---|---:|---:|---:|---:|---:|
| `BUY_idx0` | BUY YES | 35,857 | 0.705 | 0.243 | 0.213 | 0.9419 |
| `BUY_idx1` | BUY NO | 41,234 | 0.401 | 0.774 | 0.230 | 0.9581 |
| `SELL_idx0` | SELL YES | 57,175 | 0.365 | 0.220 | 0.217 | 0.0551 |
| `SELL_idx1` | SELL NO | 67,816 | 0.619 | 0.754 | 0.237 | 0.0479 |

## val — side × outcomeIndex subgroup breakdown

| Subgroup | Interpretation | n | bc_rate | p_hat mean | p_hat std | ROC (naive) |
|---|---|---:|---:|---:|---:|---:|
| `BUY_idx0` | BUY YES | 2,296 | 0.000 | 0.122 | 0.127 | — |
| `BUY_idx1` | BUY NO | 2,985 | 1.000 | 0.908 | 0.112 | — |
| `SELL_idx0` | SELL YES | 3,723 | 1.000 | 0.128 | 0.129 | — |
| `SELL_idx1` | SELL NO | 4,150 | 0.000 | 0.883 | 0.122 | — |

## test — side × outcomeIndex subgroup breakdown

| Subgroup | Interpretation | n | bc_rate | p_hat mean | p_hat std | ROC (naive) |
|---|---|---:|---:|---:|---:|---:|
| `BUY_idx0` | BUY YES | 3,829 | 0.000 | 0.116 | 0.085 | — |
| `BUY_idx1` | BUY NO | 5,113 | 1.000 | 0.900 | 0.080 | — |
| `SELL_idx0` | SELL YES | 1,812 | 1.000 | 0.117 | 0.091 | — |
| `SELL_idx1` | SELL NO | 2,660 | 0.000 | 0.921 | 0.080 | — |

## test — per-market naive ROC

| Market | n | bc_rate | p_hat mean | ROC (naive) |
|---|---:|---:|---:|---:|
| Trump announces US x Iran ceasefire end by April 15, 2026? | 4,291 | 0.518 | 0.633 | 0.5994 |
| Trump announces US x Iran ceasefire end by April 18, 2026? | 3,732 | 0.483 | 0.567 | 0.5711 |
| Trump announces US x Iran ceasefire end by April 12, 2026? | 2,372 | 0.559 | 0.578 | 0.6960 |
| Will the US x Iran ceasefire be extended by April 14, 2026? | 1,198 | 0.508 | 0.466 | 0.6524 |
| Trump announces US x Iran ceasefire end by April 10, 2026? | 1,011 | 0.498 | 0.504 | 0.7245 |
| Will the US x Iran ceasefire be extended by April 18, 2026? | 692 | 0.590 | 0.543 | 0.6964 |
| Trump announces US x Iran ceasefire end by April 8, 2026? | 118 | 0.492 | 0.499 | 0.8618 |

## test — within (market × subgroup) bet_correct variance check

If bc_std ≈ 0 for nearly every cell, aggregate ROC can only come from between-cell ranking. That confirms Simpson's-paradox artifact.

Cells audited: 28. Cells with `bc_std = 0` (constant bet_correct within cell): **28**.

| Market | Side | idx | n | bc_mean | bc_std | p_mean | Within-cell ROC |
|---|---|---:|---:|---:|---:|---:|---:|
| Trump announces US x Iran ceasefire end by April 1 | BUY | 1 | 1,737 | 1.000 | 0.000 | 0.891 | — |
| Trump announces US x Iran ceasefire end by April 1 | BUY | 1 | 1,241 | 1.000 | 0.000 | 0.872 | — |
| Trump announces US x Iran ceasefire end by April 1 | BUY | 0 | 1,075 | 0.000 | 0.000 | 0.149 | — |
| Trump announces US x Iran ceasefire end by April 1 | SELL | 1 | 1,068 | 0.000 | 0.000 | 0.909 | — |
| Trump announces US x Iran ceasefire end by April 1 | BUY | 1 | 1,028 | 1.000 | 0.000 | 0.927 | — |
| Trump announces US x Iran ceasefire end by April 1 | BUY | 0 | 1,002 | 0.000 | 0.000 | 0.136 | — |
| Trump announces US x Iran ceasefire end by April 1 | SELL | 1 | 856 | 0.000 | 0.000 | 0.923 | — |
| Trump announces US x Iran ceasefire end by April 1 | BUY | 0 | 683 | 0.000 | 0.000 | 0.083 | — |
| Trump announces US x Iran ceasefire end by April 1 | SELL | 0 | 560 | 1.000 | 0.000 | 0.151 | — |
| Trump announces US x Iran ceasefire end by April 1 | SELL | 0 | 484 | 1.000 | 0.000 | 0.130 | — |
| Will the US x Iran ceasefire be extended by April  | BUY | 0 | 426 | 0.000 | 0.000 | 0.092 | — |
| Trump announces US x Iran ceasefire end by April 1 | BUY | 0 | 390 | 0.000 | 0.000 | 0.051 | — |
| Trump announces US x Iran ceasefire end by April 1 | BUY | 1 | 390 | 1.000 | 0.000 | 0.953 | — |
| Will the US x Iran ceasefire be extended by April  | BUY | 1 | 369 | 1.000 | 0.000 | 0.920 | — |
| Trump announces US x Iran ceasefire end by April 1 | SELL | 1 | 363 | 0.000 | 0.000 | 0.930 | — |
| Trump announces US x Iran ceasefire end by April 1 | SELL | 0 | 298 | 1.000 | 0.000 | 0.080 | — |
| Will the US x Iran ceasefire be extended by April  | BUY | 1 | 298 | 1.000 | 0.000 | 0.874 | — |
| Will the US x Iran ceasefire be extended by April  | SELL | 0 | 239 | 1.000 | 0.000 | 0.099 | — |
| Will the US x Iran ceasefire be extended by April  | BUY | 0 | 202 | 0.000 | 0.000 | 0.136 | — |
| Will the US x Iran ceasefire be extended by April  | SELL | 1 | 164 | 0.000 | 0.000 | 0.950 | — |
| Trump announces US x Iran ceasefire end by April 1 | SELL | 1 | 118 | 0.000 | 0.000 | 0.962 | — |
| Trump announces US x Iran ceasefire end by April 1 | SELL | 0 | 113 | 1.000 | 0.000 | 0.037 | — |
| Will the US x Iran ceasefire be extended by April  | SELL | 0 | 110 | 1.000 | 0.000 | 0.123 | — |
| Will the US x Iran ceasefire be extended by April  | SELL | 1 | 82 | 0.000 | 0.000 | 0.902 | — |
| Trump announces US x Iran ceasefire end by April 8 | BUY | 0 | 51 | 0.000 | 0.000 | 0.050 | — |

## Plots

- `p_by_subgroup_test.png` — p_hat distribution by (side, outcomeIndex) × winner/loser, test.
- `p_by_subgroup_train.png` — same for train.
- `calibration_test.png` — does p_hat predict bet_correct monotonically?
- `calibration_train.png` — same for train.