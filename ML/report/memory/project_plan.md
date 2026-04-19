# Project Plan

**Course:** Machine Learning and Deep Learning, CBS, Spring 2026
**Group:** Alejandro Laurlund Gato (161989), Alexander Myrup (160363), Pontus Neumann (185912), Linus Stamov Yu (160714)
**Title:** *Mispricing on Polymarkets*
**Subtitle:** *Detecting Probability Asymmetries in Iran Geopolitical Markets with Machine Learning*
**Possible working title:** *: *
**Document purpose:** Finalise the research question, approach, and method, and map each methodological choice to the course syllabus. Supersedes earlier options discussed in the handover and in chat.

---

## 1. Summary

The project studies asymmetries between market-implied probabilities and model-predicted probabilities in resolved Iran geopolitical Polymarket events. The hypothesis is that a machine learning model, trained on behavioural and market-state features available at trade time, can generate a probability estimate that systematically diverges from the contemporaneous market-implied probability, and that this gap constitutes a tradable signal in settled markets. The primary output is a trading algorithm that enters positions whenever the predicted probability and the market-implied probability differ beyond a threshold, evaluated by out-of-sample economic performance on temporally held-out markets. The pulled dataset covers all 74 resolved sub-markets across four Iran-related events (114242, 236884, 355299, 357625), comprising 346,898 trades across 73,839 wallets between 2025-12-22 and 2026-04-19.

A secondary, out-of-scope observation is noted for the Discussion. If feature-importance analysis reveals that the signal concentrates on features resembling those associated with informed trading in the literature (abnormal bet size, pre-event timing, wallet newness, directional concentration), the report will reflect on this theoretical parallel. This does not enter the research question, scope, or evaluation. It is flagged as an avenue for future work.

## 2. Research Questions

### Scope and purpose

This project studies the gap between Polymarket's contemporaneous market-implied probabilities and the probabilities predicted by a machine learning model on resolved Iran geopolitical markets. The scope covers all 74 resolved sub-markets under Polymarket events 114242, 236884, 355299, and 357625 (strikes, conflict-end, ceasefire announcement, ceasefire extensions), with trade correctness derived from official resolution. The purpose is to test whether systematic asymmetries between the two probability series exist in settled markets and whether a simple trading rule built on the gap produces positive risk-adjusted returns relative to naive baselines on temporally held-out data.

### Short form

**RQ1.** Can a machine learning model trained on pre-execution market-state and behavioural features produce a probability estimate that systematically differs from the contemporaneous market-implied probability in settled Iran prediction markets, such that a trading rule acting on the gap generates positive risk-adjusted returns out of sample?

### Detailed form

A single research question, decomposed into a predictive sub-question and an economic sub-question. Both tie to the same Results section.

**RQ1a — Probability gap.**
Does a multilayer perceptron trained on pre-execution features (market state, recent market activity, wallet history, on-chain wallet identity, news proximity) produce a probability estimate `p_hat` whose residual against the contemporaneous market-implied probability `market_implied_prob` predicts trade correctness on a temporally held-out test set?

*Scope:* all 74 resolved Iran sub-markets under Polymarket events 114242, 236884, 355299, and 357625; feature set as listed in Section 4; temporal split by market settlement date; no random split within a market.
*Success criterion:* ROC-AUC of `p_hat` residualised against `market_implied_prob` is strictly above 0.5 on the test set, with Brier-score improvement over the market-implied null and an improved calibration curve.

**RQ1b — Trading rule.**
Does a threshold-based trading rule that enters positions when `|p_hat − market_implied_prob|` exceeds a chosen cutoff generate positive cumulative PnL and Sharpe ratio out of sample, relative to (a) a naive market-implied baseline, (b) a momentum rule, and (c) random entry?

*Scope:* rule parameters (gap threshold, position sizing, holding period) are tuned on the validation markets only and then frozen for the test markets.
*Success criterion:* positive cumulative PnL on the test markets, Sharpe strictly above the three baselines, and a hit rate materially above `market_implied_prob` on entered trades.

## 3. Framing Decision

### Why the trading-signal framing

The efficient-market null for Polymarket is that the market-implied price already incorporates all public information, so no pre-execution feature set should add predictive power. A systematic gap between `p_hat` and `market_implied_prob` that predicts settlement is therefore direct evidence of mispricing that can, in principle, be monetised. Framing the project around this gap makes the target observable (settlement is public), makes the evaluation economic and quantitative (PnL, Sharpe, drawdown), and avoids the labelling circularity that arises when one tries to pre-label wallets as informed.

### What the framing preserves
- The 35-feature behavioural and market-state feature set (market state, wallet history, on-chain age, news timing).
- Data sources, target markets, and the Polygonscan extraction plan.
- Temporal train / validation / test split by market settlement date.

### What the framing changes
- The headline deliverable is the trading rule and its out-of-sample economic performance, not a wallet-level insider detector.
- Unsupervised anomaly detection (autoencoder, Isolation Forest) is retained from Lecture 11 as a secondary lens on trades where the gap is largest, to check whether flagged trades share structure.
- Any resemblance between the features driving the signal and documented informed-trading traits is handled in Discussion, not in the research question.

## 4. Data

| Item | Detail |
|---|---|
| Markets | All resolved sub-markets under Polymarket events 114242 ("US strikes Iran by ..."), 236884 ("Iran x Israel/US conflict ends by ..."), 355299 ("Trump announces US x Iran ceasefire end by ...") and 357625 ("US x Iran ceasefire extended by ..."). 74 resolved markets covering both YES and NO outcomes. |
| Unit of analysis | One resolved trade. Realised dataset: 346,898 trades, 73,839 unique wallets, spanning 2025-12-22 to 2026-04-19. |
| Data sources | Polymarket Gamma API (event and market metadata), CLOB API (token-level mid-price history, with trade-execution price as fallback once CLOB history is purged after resolution), Data API (paginated trade history with side-split fallback at the offset cap). No Polygonscan or external enrichment used in this iteration. |
| Target | `bet_correct` in {0, 1} from market resolution and the side of the trade. |
| Benchmark at trade time | `market_implied_prob` at execution, taken from the CLOB mid-price where available, otherwise the price field of the trade itself. |
| Features | Behavioural and market-state, strictly no-lookahead, grouped as: market-state-so-far (`market_trade_count_so_far`, `market_volume_so_far_usd`, `market_price_vol_last_1h`), wallet history (`wallet_prior_trades`, `wallet_prior_volume_usd`, `wallet_prior_win_rate`, `wallet_first_minus_trade_sec`), and trade-level market state (`trade_value_usd`, `settlement_minus_trade_sec`). Wallet on-chain identity (Polygonscan) and news proximity (GDELT) are deferred to future work. |
| Pre-modelling filter | Drop post-resolution close-out trades (`settlement_minus_trade_sec <= 0`, ~16.5% of rows), which execute at the locked winning-token price and carry no predictive signal. |
| Split | Temporal by settlement date. Earliest settled markets form the training set, later markets validation, latest markets test. Within a single market, no random split, to eliminate cross-trade leakage. |
| Class balance | Current `bet_correct` rate is 0.518, inside the 35 to 65 percent band, so no resampling is required up front. If imbalance develops after filtering or per split fold, apply `class_weight="balanced"` first, then SMOTE on the training fold only as a secondary option (Lecture 7). |
| Known limitation | Polymarket Data API caps pagination offset at ~3000; side-split lifts this to ~7000 trades per market. 21 of 74 markets are truncated at this ceiling, creating a recency bias on long-lived markets. Documented as a scope limitation in the report. |

## 5. Method, Mapped to Course Lectures

### 5.1 Primary model: MLP for probability estimation (Lectures 8, 9)

- Architecture: fully connected feed-forward network, two to four hidden layers, SELU activations, Glorot initialisation.
- Regularisation: dropout in [0.2, 0.4] on dense layers (Lecture 9 rule of thumb), batch normalisation after each hidden layer.
- Loss: binary cross-entropy. Optimiser: Adam with learning-rate scheduling.
- Input: standardised behavioural and market-state features. `market_implied_prob` is withheld from the feature set so that `p_hat` is an independent probability estimate, directly comparable to the market.
- Output: predicted probability of settlement in favour of the trade, `p_hat`. The gap `p_hat − market_implied_prob` is the trading signal.

### 5.2 Trading rule

- Entry: take the side of the trade whenever `|p_hat − market_implied_prob|` exceeds a threshold `tau`, tuned on the validation markets.
- Sizing: flat stake initially; a Kelly-scaled variant as a robustness check.
- Holding period: hold to settlement. An early-exit variant at a fixed time-to-event is a robustness check.
- Frozen for test: `tau`, sizing rule, and holding period are frozen after validation tuning and not retouched on the test markets.

### 5.3 Unsupervised arm: autoencoder anomaly detection (Lecture 11)

- Undercomplete stacked autoencoder trained on all trade feature vectors, SELU activations, MSE loss.
- Anomaly score: per-trade reconstruction error.
- Purpose: a parallel, unsupervised lens on the data that does not use the correctness target. Cross-check whether trades with the largest `|p_hat − market_implied_prob|` gap also carry high reconstruction error.

### 5.4 Baselines (Lectures 4, 5, 6, 7)

- Logistic regression on the same features, producing a baseline `p_hat` for the same trading rule.
- Random forest, producing a baseline `p_hat` and feeding the feature-importance ranking used in Discussion.
- Isolation Forest as an unsupervised anomaly baseline against the autoencoder.
- Naive market baseline: trading rule with `p_hat = market_implied_prob`, which by construction produces zero gap and zero signal. This is the efficient-market null.

### 5.5 Class imbalance and data issues (Lecture 7)

- Compute per-market outcome balance. If the overall `bet_correct` rate falls outside the 35 to 65 percent band, apply sklearn `class_weight="balanced"` first.
- If still degenerate, apply SMOTE to the training fold only, never to validation or test.
- Outlier handling before training: winsorise `trade_value_usd` and `wallet_total_volume_usd` at the 1st and 99th percentiles. No whole-row removal, since extreme trades are part of the phenomenon of interest.

### 5.6 Validation strategy

Two layers, both on held-out markets:

1. **Statistical.** ROC-AUC and calibration of `p_hat` and of the residualised gap against `bet_correct` on validation and test markets. Brier-score improvement over the market-implied null.
2. **Economic.** Cumulative PnL, Sharpe ratio, hit rate, and maximum drawdown of the trading rule, on the test markets only, against the three baselines.

## 6. Evaluation Metrics

| Layer | Metrics |
|---|---|
| Probability quality | ROC-AUC, PR-AUC, Brier score, calibration curve of `p_hat`. |
| Gap quality | ROC-AUC of the residual `p_hat − market_implied_prob` for predicting `bet_correct`. |
| Trading rule | Cumulative PnL, annualised Sharpe, hit rate, maximum drawdown, turnover, trade count. |
| Unsupervised arm | Overlap between top-decile gap trades and top-decile reconstruction-error trades, benchmarked against a random-overlap null. |
| Complexity | Training wall time and inference latency per 1k trades, relative to logistic regression (mandatory per the project guidelines). |

## 7. Interpretability and Ethics

Lecture 14 treats XAI at a conceptual level rather than as specific techniques. The report will therefore rely on:

- Feature-importance rankings from the random forest baseline and permutation importance on the MLP validation set.
- Partial-dependence plots for the top three features driving the gap.
- A brief XAI framing paragraph using the traceability, accuracy, and understanding pillars from Lecture 14.

The ethical consideration section will cover: (i) privacy of pseudonymous on-chain wallets, (ii) the potential for a published trading signal to be arbitraged away or reverse-engineered, (iii) regulatory externalities of prediction-market mispricing, and (iv) dataset bias toward high-volume English-language geopolitical markets.

## 8. Out of Scope — Reserved for Discussion and Future Work

The following observation is explicitly *outside* the research question and the evaluation. It is reserved for the Discussion and theory sections of the report and may be extended in future work.

- **Feature-importance parallel to informed-trading traits.** If the features driving the `p_hat − market_implied_prob` gap resemble those documented in the informed-trading literature (within-trader bet size, cross-sectional bet size, pre-event timing, directional concentration, wallet newness; see Mitts and Ofir 2026), the Discussion will flag the parallel and note that the signal may be partially picking up informed flow. No labelling, evaluation, or success criterion in the main study depends on this interpretation.
- **Documented-case validation** (Magamyman, Burdensome-Mix) is similarly deferred to Discussion as illustrative anecdotes, not as evaluation targets. The Iran ceasefire cluster, by contrast, is now part of the main evaluation scope via events 355299 and 357625.

## 9. Report Outline Alignment

The docx at `ML/report/ML_final_exam_paper.docx` follows the new extended guidelines and has been synced with this plan (cover, motivation, dataset, data extraction, features, and limitations updated 19 April). Subsection headings inside *Methodology → Data Analytics: Modelling, Methods and Tools*:

- H3: *Primary model — MLP for probability estimation*
- H3: *Trading rule on the probability gap*
- H3: *Unsupervised arm — autoencoder and Isolation Forest*
- H3: *Baselines and the market-implied benchmark*

The economic evaluation of the trading rule sits under Results. The informed-trading parallel, if observed, sits under Discussion.

## 10. Deliverables and Next Steps

| # | Task | Status | Depends on |
|---|---|---|---|
| 1 | Polymarket Gamma + CLOB + Data API fetcher with side-split and trade-price fallback | done (19 Apr) | — |
| 2 | Full trade extraction across four target events | done (19 Apr) | 1 |
| 3 | True resolution-timestamp derivation (`resolution_ts`) and `settlement_minus_trade_sec` | done (19 Apr) | 2 |
| 4 | Running market and wallet features in `trades_enriched.csv` | done (19 Apr) | 2 |
| 5 | Polygonscan on-chain wallet enrichment | deferred (no API key, free-data-only scope) | 2 |
| 6 | GDELT news-timing enrichment | deferred (free API, awaits feature-definition design) | 2 |
| 7 | Offset-cap workaround via Polymarket Goldsky subgraph | deferred (documented as limitation) | — |
| 8 | Post-resolution filter applied (`settlement_minus_trade_sec > 0`) | open | 3 |
| 9 | Temporal train / validation / test split by settlement date | open | 8 |
| 10 | EDA notebook following repository Design.md conventions | open | 8 |
| 11 | MLP training and baselines (logistic regression, random forest, naive market) | open | 9 |
| 12 | Trading-rule tuning on validation markets | open | 11 |
| 13 | Out-of-sample trading-rule evaluation on test markets | open | 12 |
| 14 | Autoencoder arm and overlap check | open | 9 |
| 15 | Feature-importance analysis for Discussion | open | 11 |
| 16 | Report drafting in the CBS docx template | all | 10 to 15 |

## 11. Open Decisions

- Choice of gap threshold `tau` and position-sizing rule (flat vs Kelly-scaled) on the validation markets.
- Whether to include a time-to-event holding-period variant as a robustness check or as the main specification.
- Whether to model trade value in USD as a feature or as a sample weight.
- Exact settlement-date boundaries for the temporal train / validation / test split across the four event clusters.
- Treatment of `wallet_prior_win_rate` NaN on first-trade rows (~21%): impute to global mean, use the raw NaN with `wallet_prior_trades == 0` as a categorical indicator, or exclude first-trade rows.
