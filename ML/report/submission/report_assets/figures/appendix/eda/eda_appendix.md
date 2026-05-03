# Appendix A — Wallet-joined cohort EDA (sidecar)

Sidecar EDA captions for review before merging into the main report. Numbering picks up after the existing Appendix A.9.

### Figure A.10. (✓)
`02_wallet_coverage.png`

Wallet enrichment coverage by split. After two PolygonScan extension passes, every taker in Alex's idea1 cohort is matched to a row in the wallet enrichment table, giving 100.0 percent train and 100.0 percent test coverage on the 1,371,180-trade cohort. The Layer 6 NaN rate falls to 0 percent after the retry passes.

### Figure A.11. (✓)
`08_train_test_shift.png`

Train-to-test mean shift for the top 15 numeric features, expressed as the absolute standardised mean difference (Cohen d convention: 0.2 small, 0.5 medium, 0.8 large). Time-to-deadline indicators show large shift (d above 0.9), reflecting the regime change between the Iran-strike and Iran-ceasefire countdowns. Wallet-history features show medium shift, motivating the per-market GroupKFold protocol.

### Figure A.12. (✓)
`09_late_flow.png`

Hit rate and trade-count share by time-to-deadline bucket for train (Iran strike) and test (Iran ceasefire) cohorts. Test base rate falls below 0.40 inside the final hour and below 0.34 in the last five minutes, replicating the late-concentrated informed-flow signature documented in Mitts and Ofir 2026. The corresponding train-cohort buckets stay near the 0.50 base rate, isolating the phenomenon to the ceasefire regime.

### Figure A.13. (✓)
`10_wallet_strata.png`

Bet-correct base rate stratified by three Layer 6 wallet features. Left: deciles of wallet age in days at the time of the trade. Centre: causal CEX-funding flag, defined as first USDC inbound from a known CEX hot wallet observed before the trade timestamp. Right: deciles of polygon nonce at trade time, a proxy for prior on-chain experience. The split structure motivates inclusion of the Layer 6 enrichment in the strict-branch feature set.

### Figure A.14. (✓)
`11_per_market_bimodality.png`

Per-market bet_correct base rate, split by train and test cohorts. The single-event resolution of each market produces a bimodal distribution: markets resolving with the consensus side cluster near 1, markets resolving against consensus cluster near 0. The shape is the structural reason single-feature ROC across markets is highly variable and motivates the GroupKFold(market_id) evaluation protocol.

### Figure A.15. (✓)
`12_feature_stability.png`

Single-feature ROC-AUC heatmap across markets for the top eight features by absolute Pearson correlation with the target. Several features (log_payoff_if_correct, contrarian_score, risk_reward_ratio_pre) achieve median single-feature AUC near 0.31 with p95 above 0.67, indicating a single transferable rule that inverts between YES- and NO-resolved markets. This is the structural finding behind the per-market bimodality and reinforces the use of group-aware cross validation.

### Figure A.16. (✓)
`13_mutual_information.png`

Top 20 features by mutual information with bet_correct, computed on a 150,000-row stratified sample using scikit-learn's mutual_info_classif. Mutual information captures non-linear dependence that the Pearson correlation in Figure A.5 cannot detect. The leading features are short-window microstructure variables (realized volatility, jump component, order-flow imbalance) rather than wallet-identity features, consistent with the literature emphasis on flow signals.

### Figure A.17. (✓)
`14_feature_taxonomy.png`

Numeric feature counts by engineering layer. Of the 79 numeric features in the wallet-joined cohort, the market-state rolling layer contributes the largest share, followed by price and volatility microstructure, the HF-internal taker aggregates, and the on-chain Layer 6 wallet features. The taxonomy frames how the report's permutation-importance and ablation analyses are organised.

### Figure A.18. (✓)
`15_tail_diagnostics.png`

Excess kurtosis ranking for the 15 most fat-tailed numeric features, with the |kurt|=3 fat-tail threshold marked. The accompanying CSV (15_tail_diagnostics.csv) reports the 1st, 5th, 95th, and 99th percentiles plus tail-conditional means for each feature, supporting the winsorisation choice during pre-processing.

### Figure A.19. (✓)
`16_temporal_drift.png`

Daily bet_correct base rate during the strike-countdown (train) and ceasefire-countdown (test) windows, with a seven-day rolling mean overlay. Daily bins are filtered to days with at least 50 trades. The rolling means stay within a tight band around 0.50, confirming there is no temporal drift in the target within either cohort that would invalidate the static train and test split.

