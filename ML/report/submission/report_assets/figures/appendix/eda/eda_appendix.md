# Appendix A — Wallet-joined cohort EDA (sidecar)

Sidecar EDA captions for review before merging into the main report. Numbering picks up after the existing Appendix A.9.

### Figure A.10. (✓)
`08_train_test_shift.png`

Train-to-test mean shift for the top 15 numeric features, expressed as the absolute standardised mean difference (Cohen d convention: 0.2 small, 0.5 medium, 0.8 large). Time-to-deadline indicators show large shift (d above 0.9), reflecting the regime change between the Iran-strike and Iran-ceasefire countdowns. Wallet-history features show medium shift, motivating the per-market GroupKFold protocol.

### Figure A.11. (✓)
`09_late_flow.png`

Hit rate and trade-count share by time-to-deadline bucket for train (Iran strike) and test (Iran ceasefire) cohorts. Test base rate falls below 0.40 inside the final hour and below 0.34 in the last five minutes, replicating the late-concentrated informed-flow signature documented in Mitts and Ofir 2026. The corresponding train-cohort buckets stay near the 0.50 base rate, isolating the phenomenon to the ceasefire regime.

### Figure A.12. (✓)
`10_wallet_strata.png`

Bet-correct base rate stratified by three Layer 6 wallet features. Left: deciles of wallet age in days at the time of the trade. Centre: causal CEX-funding flag, defined as first USDC inbound from a known CEX hot wallet observed before the trade timestamp. Right: deciles of polygon nonce at trade time, a proxy for prior on-chain experience. The split structure motivates inclusion of the Layer 6 enrichment in the strict-branch feature set.

### Figure A.13. (✓)
`11_per_market_bimodality.png`

Per-market hit rate decomposed by trade side. The x-axis plots the share of YES-side trades that ended bet_correct in a market, and the y-axis plots the same share for NO-side trades. Marker size is proportional to log10 trade count per market. The 73 markets cleanly partition into two clusters: 49 YES-resolved markets in the lower-right (high YES-side hit rate, low NO-side) and 24 NO-resolved markets in the upper-left (mirror image). The clusters sit near (0.7, 0.3) rather than (1.0, 0.0) because bet_correct also reflects BUY vs SELL direction within each side, so YES SELL trades behave like NO bets. The two-cluster structure is the reason single-feature ROC inverts across markets and motivates the GroupKFold(market_id) evaluation protocol.

### Figure A.14. (✓)
`12_feature_stability.png`

Single-feature ROC-AUC heatmap across markets for the top eight features by absolute Pearson correlation with the target. ROC-AUC ranges from 0 to 1 and measures, for one feature alone, the probability that a randomly chosen positive trade (bet_correct=1) has a higher feature value than a randomly chosen negative trade (bet_correct=0). 0.5 is no signal, values above 0.5 mean the feature ranks positives higher, and values below 0.5 mean the relationship inverts. Per-market AUCs straddling 0.5 therefore indicate a feature whose direction flips between YES- and NO-resolved markets. Several features (log_payoff_if_correct, contrarian_score, risk_reward_ratio_pre) achieve median single-feature AUC near 0.31 with p95 above 0.67, the structural finding behind the per-market resolution split in Figure A.13 and the reason group-aware cross validation is required.

### Figure A.15. (✓)
`13_mutual_information.png`

Top 25 features by mutual information with bet_correct, computed on a 150,000-row stratified sample using scikit-learn's mutual_info_classif and colored by feature group. Mutual information captures non-linear dependence that the Pearson correlation in Figure A.5 cannot detect. The x-axis is zoomed to the relevant range so the rank gap between adjacent features is legible. The leading features are short-window market-state and price microstructure variables (realized volatility, jump component, order-flow imbalance, recent volume) rather than wallet-identity features, consistent with the literature emphasis on flow signals.

### Figure A.16. (✓)
`14_feature_taxonomy.png`

Numeric feature counts by engineering layer. Of the 79 numeric features in the wallet-joined cohort, the market-state rolling layer contributes the largest share, followed by price and volatility microstructure, the HF-internal taker aggregates, and the on-chain Layer 6 wallet features. The taxonomy frames how the report's permutation-importance and ablation analyses are organised.

### Figure A.17. (✓)
`19_event_timing.png`

Trade volume per day across the joined cohort, colored by split and annotated with the two events that anchor the cohort design. The train cohort runs from late December through the Iran strike on 28 February, ending in a single intraday spike of roughly ninety-three thousand trades on the day of the event. The test cohort runs from 1 March through 7 April and is distributed across the run-up to the ceasefire announcement, with no comparable spike. The calendar separation between the two regimes is the structural reason the cohort is split by event rather than randomly.

### Figure A.18. (✓)
`16_temporal_drift.png`

Daily bet_correct base rate during the strike-countdown (train) and ceasefire-countdown (test) windows, with a seven-day rolling mean overlay. Daily bins are filtered to days with at least 50 trades. The rolling means stay within a tight band around 0.50, confirming there is no temporal drift in the target within either cohort that would invalidate the static train and test split.

### Table A.1.

| Split | n markets | Min trades / market | Median trades / market | Max trades / market |
|---|---|---|---|---|
| Train | 63 | 1,274 | 9,753 | 119,842 |
| Test | 10 | 5,127 | 14,692 | 90,730 |

Cohort sizing per split, summarising the wide trade-count spread across the 73 markets that motivates the GroupKFold(market_id) protocol used in the modeling section. Source: 07_cohort_sizing_table.csv.

