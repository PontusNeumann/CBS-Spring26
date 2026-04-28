# Leakage diagnostic summary


| check | n_features | OOF AUC | fold AUCs | test AUC | per-market AUC range |
|---|---:|---:|---|---:|---|
| check1_single_feature | 1 | 0.498 | [0.517, 0.499, 0.505, 0.482, 0.488] | 0.497 | [0.50, 0.50] |
| check2_drop_kyle | 64 | 0.848 | [0.766, 0.497, 0.991, 0.998, 0.991] | 0.876 | [0.06, 1.00] |
| check3_drop_suspect_family | 54 | 0.848 | [0.761, 0.501, 0.993, 0.999, 0.989] | 0.871 | [0.06, 1.00] |

## Reference (v3 sweep, full feature set)
- LogReg L2: OOF 0.623, test 0.615, per-market [0.42, 0.74] — clean
- Random Forest (full set): OOF 0.867, test 0.884, per-market [0.03, 1.00] — leaky

## Suspect feature family (Check 3 drops these)
- `kyle_lambda_market_static`
- `log_recent_volume_24h`
- `log_trade_count_24h`
- `market_price_vol_last_24h`
- `order_flow_imbalance_24h`
- `pre_trade_price_change_24h`
- `recent_price_high_1h`
- `recent_price_low_1h`
- `recent_price_mean_24h`
- `recent_price_range_1h`
- `sister_price_dispersion_5min`