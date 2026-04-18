# Project Plan

**Course:** Machine Learning and Deep Learning, CBS, Spring 2026
**Group:** Alejandro Laurlund Gato (161989), Alexander Myrup (160363), Pontus Neumann (185912), Linus Stamov Yu (160714)
**Working title:** *Informed Trading on Polymarket: Predicting Bet Correctness from Behavioural Features in Settled Iran Prediction Markets*
**Document purpose:** Finalise the research question, approach, and method, and map each methodological choice to the course syllabus. Supersedes earlier options discussed in the handover and in chat.

---

## 1. Summary

The project studies informed trading on Polymarket using resolved Iran geopolitical markets. Two framings were considered: (i) flag individual wallets as insiders, then evaluate copy-trade returns; (ii) model the predictability of bet correctness from behavioural trade features. Option (ii) is chosen. It has an observable target, aligns tightly with the course syllabus (MLP, autoencoders, class imbalance, anomaly detection), and avoids the labelling circularity that arises when proxy-labelling wallets as insiders before training a model to detect them. Explicit wallet flagging is demoted to a secondary, descriptive output derived from the primary model.

## 2. Research Questions

### Scope and purpose

This project studies informed trading on Polymarket by modelling the predictability of bet correctness from behavioural trade features on resolved Iran geopolitical prediction markets. The scope is confined to six to eight settled sub-markets under Polymarket events 114242 and 236884, with bet correctness derived from official resolution. The purpose is twofold: first, to establish whether a multilayer perceptron can outperform the contemporaneous market-implied probability as a predictor of trade correctness; second, to characterise the trades that carry this edge and assess whether they are consistent with documented patterns of informed trading.

### Short form

**RQ1.** Can a multilayer perceptron trained on pre-execution behavioural features predict the correctness of Polymarket trades on settled Iran markets better than the contemporaneous market-implied probability and classical baselines?

**RQ2.** Do the trades that carry this predictive edge exhibit a feature profile, unsupervised-anomaly status, and wallet identities consistent with informed trading?

### Detailed form

Two questions, each tied to a distinct Results subsection and a distinct set of metrics. RQ1 concerns whether a predictive edge exists. RQ2 concerns whether that edge is consistent with informed trading.

**RQ1 — Predictability.**
Does a multilayer perceptron trained on pre-execution behavioural features of Polymarket trades (market state, recent market activity, wallet history, on-chain wallet identity, and news proximity) achieve a higher ROC-AUC for predicting `bet_correct` on a temporally held-out test set of settled Iran markets than (a) the contemporaneous market-implied probability and (b) classical baselines (logistic regression and random forest) using the same features?

*Scope:* six to eight resolved Iran sub-markets under Polymarket events 114242 and 236884; behavioural feature set as listed in Section 4; temporal split by market settlement date; no random split within a market.
*Maps to Results:* "Predictive performance and uplift over the market-implied baseline."
*Success criterion:* ROC-AUC of the MLP is strictly higher than both the market-implied null and the best classical baseline on the test set, with Brier-score-improvement and calibration reported alongside.

**RQ2 — Informed-trading signature.**
Conditional on RQ1, do the trades carrying the predictive edge (top-decile `p_hat` on the test set) concentrate in a subset of trades whose feature profile, unsupervised-anomaly status, and wallet identities are consistent with informed trading?

*Scope:* three converging tests.
1. *Feature profile.* Top-decile trades are compared against the remaining ninety percent on the informed-trading signals highlighted in Mitts and Ofir (2026): within-trader bet size, cross-sectional bet size, pre-event timing, directional concentration, and wallet newness.
2. *Anomaly overlap.* Overlap between top-decile MLP trades and top-decile autoencoder reconstruction-error trades, benchmarked against a random-overlap null.
3. *Documented cases.* Recall of documented insider wallets (Magamyman, the Iran ceasefire cluster, Burdensome-Mix) among top-decile MLP trades and among top-decile autoencoder trades.

*Maps to Results:* "Characterisation of high-edge trades."
*Success criterion:* top-decile trades are statistically distinguishable from the remainder on a majority of the Mitts and Ofir signals, autoencoder overlap exceeds the random-overlap null, and recall of documented cases is strictly above chance.

## 3. Framing Decision

### Why reframe away from explicit wallet labelling

Wallet-level insider labels are unobservable. Any proxy (for example, wallets with abnormal pre-resolution returns) is close to tautological: a model then trained to predict the proxy recovers the proxy definition, not an independent phenomenon. Trade-level bet correctness, by contrast, is directly observable from the market resolution and requires no hand-crafted label function.

The reframe preserves the spirit of the handover. In an efficient market, behavioural features at trade time should carry no predictive signal for correctness beyond the market-implied probability. A systematic predictive edge therefore constitutes evidence of informed trading at the trade level. Wallet-level inferences are then built up from the trade-level scores, rather than assumed in advance.

### What the reframe preserves from the handover
- The 35-feature behavioural feature set (market state, wallet history, on-chain age, news timing).
- Validation against documented cases (Magamyman, Burdensome-Mix, Iran ceasefire cluster).
- Data sources, target markets, and the Polygonscan extraction plan.

### What the reframe adds from the chat discussion
- A temporal train / validation / test split by market settlement date.
- A copy-trade economic validation in Results, using top-decile model scores.
- An autoencoder arm (Lecture 11) as an unsupervised parallel to the MLP.

## 4. Data

| Item | Detail |
|---|---|
| Markets | Resolved sub-markets under Polymarket events 114242 ("US strikes Iran by ...") and 236884 ("Iran x Israel/US conflict ends by ..."). Target: 6 to 8 markets covering both YES and NO outcomes. |
| Unit of analysis | One resolved trade. Expected dataset size: ~30k to 60k trades once Polygonscan extraction completes (handover blocker). |
| Target | `bet_correct` in {0, 1} from market resolution and the side of the trade. |
| Benchmark at trade time | `market_implied_prob` at execution, taken from the CLOB mid-price or the price field of the trade itself. |
| Features | Behavioural, no-lookahead, grouped as: market-state-so-far, recent market activity, wallet history, wallet on-chain identity, news proximity. See handover section 6 for the full list. |
| Split | Temporal by settlement date. Earliest settled markets form the training set, later markets validation, latest markets test. Within a single market, no random split, to eliminate cross-trade leakage. |
| Class balance | Depends on the outcome balance of selected markets. Where imbalance is moderate or worse, apply class weighting first, then SMOTE as a secondary option (Lecture 7). |

## 5. Method, Mapped to Course Lectures

### 5.1 Primary model: MLP for trade-level correctness (Lectures 8, 9)

- Architecture: fully connected feed-forward network, two to four hidden layers, SELU activations, Glorot initialisation.
- Regularisation: dropout in [0.2, 0.4] on dense layers (Lecture 9 rule of thumb), batch normalisation after each hidden layer.
- Loss: binary cross-entropy. Optimiser: Adam with learning-rate scheduling.
- Input: standardised behavioural features plus the `market_implied_prob` at trade time, so the MLP can learn edges above the market baseline rather than rediscovering the baseline.
- Output: predicted probability of bet correctness, `p_hat`.

### 5.2 Unsupervised arm: autoencoder anomaly detection (Lecture 11)

- Undercomplete stacked autoencoder trained on all trade feature vectors, SELU activations, MSE loss.
- Anomaly score: per-trade reconstruction error. High error = behaviourally unusual trade.
- Purpose: a parallel, unsupervised lens on the data that does not use the correctness target. Cross-check whether autoencoder-flagged trades have a higher empirical win rate and greater overlap with documented cases than a random sample.

### 5.3 Baselines (Lectures 4, 5, 6, 7)

- Logistic regression on the same features, with L2 regularisation.
- Random forest, used both as a predictive baseline and for feature-importance ranking.
- Isolation Forest as an unsupervised anomaly baseline, to compare against the autoencoder.
- Naive market baseline: predict correctness equal to `market_implied_prob` and no behavioural information. This is the efficient-market null.

### 5.4 Class imbalance and data issues (Lecture 7)

- Compute per-market outcome balance. If overall `bet_correct` rate falls outside the 35 to 65 percent band, apply sklearn `class_weight="balanced"` first.
- If still degenerate, apply SMOTE to the training fold only, never to validation or test.
- Outlier handling before training: winsorise `trade_value_usd` and `wallet_total_volume_usd` at the 1st and 99th percentiles. No whole-row removal, since extreme trades are part of the phenomenon of interest.

### 5.5 Validation strategy

Three layers, all on held-out markets:

1. **Statistical.** Compare MLP ROC-AUC and calibration against the naive market baseline and against the classical baselines.
2. **Case-based.** Measure whether trades by Magamyman and other documented insider wallets fall in the top decile of `p_hat` and in the top decile of autoencoder reconstruction error.
3. **Economic.** Simulate a copy-trade strategy that enters in the same direction as top-decile `p_hat` trades on held-out markets. Compare cumulative PnL, hit rate, Sharpe ratio, and maximum drawdown to a market-implied baseline, a simple momentum rule, and random entry.

## 6. Evaluation Metrics

| Layer | Metrics |
|---|---|
| Classification | ROC-AUC, PR-AUC, Brier score, calibration curve, F1 at the operating point where precision = 0.7. |
| Uplift over market | ROC-AUC of `p_hat` residualised against `market_implied_prob`. |
| Anomaly arms | Precision at top 1%, top 5%, top 10% against the case-based validation set. |
| Economic | Cumulative PnL, annualised Sharpe, hit rate, maximum drawdown, turnover. |
| Complexity | Training wall time and inference latency per 1k trades, relative to logistic regression (mandatory per the project guidelines). |

## 7. Interpretability and Ethics

Lecture 14 treats XAI at a conceptual level rather than as specific techniques. The report will therefore rely on:

- Feature-importance rankings from the random forest baseline and permutation importance on the MLP validation set.
- Partial-dependence plots for the top three features.
- A brief XAI framing paragraph using the traceability, accuracy, and understanding pillars from Lecture 14.

The ethical consideration section will cover: (i) privacy of pseudonymous on-chain wallets, (ii) the potential for a published detector to be reverse-engineered by sophisticated insiders, (iii) regulatory and enforcement externalities, and (iv) dataset bias toward high-volume English-language geopolitical markets.

## 8. Report Outline Alignment

The docx at `ML/report/ML_final_exam_185912.docx` already follows the new extended guidelines. Three subsection headings inside *Methodology → Data Analytics: Modelling, Methods and Tools* should be updated to reflect this plan when the next round of edits is made:

- H3: *Primary model — MLP for trade-level correctness*
- H3: *Unsupervised arm — autoencoder and Isolation Forest*
- H3: *Baselines and the market-implied benchmark*

Secondary validation (copy-trade backtest, documented-case flag rate) sits under Results, not Methodology.

## 9. Deliverables and Next Steps

| # | Task | Owner | Depends on |
|---|---|---|---|
| 1 | Polygonscan API key and extraction script | open | — |
| 2 | Full trade extraction for 6 to 8 target markets | open | 1 |
| 3 | Wallet enrichment (Data API + Polygonscan) | open | 2 |
| 4 | GDELT news-timing enrichment | open | 2 |
| 5 | Feature engineering pipeline | open | 3, 4 |
| 6 | EDA notebook following repository Design.md conventions | open | 5 |
| 7 | MLP training and baselines | open | 5 |
| 8 | Autoencoder arm | open | 5 |
| 9 | Case-based validation against documented wallets | open | 7, 8 |
| 10 | Copy-trade backtest | open | 7 |
| 11 | Report drafting in the CBS docx template | all | 6 to 10 |

## 10. Open Decisions

- Whether to include a secondary experiment on SEC Form 3/4/5 data as a transfer-learning probe (handover section 3). Recommended: defer to appendix or future work, keep core scope on Polymarket to stay within the 15-page limit.
- Whether to model trade value in USD as a feature or as a sample weight.
- Choice of operating point for the precision / recall tradeoff when presenting flagged trades.
