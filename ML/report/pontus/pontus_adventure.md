# Pontus's Adventure — Modelling Plan

**Course:** Machine Learning and Deep Learning, CBS, Spring 2026
**Group:** Alejandro Laurlund Gato (161989), Alexander Myrup (160363), Pontus Neumann (185912), Linus Stamov Yu (160714)
**Title:** *Mispricing on Polymarkets*
**Subtitle:** *Detecting Probability Asymmetries in Iran Geopolitical Markets with Machine Learning*
**Document purpose:** Pontus's modelling direction. Builds on `foundation.md` (shared RQ, data, features, missing-data policy, EDA, evaluation). This file starts as a copy of the modelling sections from the pre-split project plan; Pontus edits it to reflect his chosen model family, hyper-parameter choices, and trading-rule tuning. Section numbers preserve the original `project_plan.md` numbering so cross-references to the foundation and to `alex_adventure.md` stay aligned.

**Companion documents:**
- `foundation.md` — shared project foundation. Read first.
- `alex_adventure.md` — the parallel modelling direction (for cross-reference and eventual comparison).

---

## 5. Method (adventure portions)

### 5.1 Primary model: MLP for probability estimation (Lectures 8, 9)

- Architecture: fully connected feed-forward network, two to four hidden layers, SELU activations, Glorot initialisation.
- Regularisation: dropout in [0.2, 0.4] on dense layers (Lecture 9 rule of thumb), batch normalisation after each hidden layer.
- Loss: binary cross-entropy. Optimiser: Adam with learning-rate scheduling.
- Input: standardised behavioural and market-state features. `market_implied_prob` is withheld from the feature set so that `p_hat` is an independent probability estimate, directly comparable to the market.
- Output: predicted probability of settlement in favour of the trade, `p_hat`. The gap `p_hat − market_implied_prob` is the trading signal.

### 5.2 Trading rule — two strategies evaluated side by side

For a candidate trade at market price `P_market`, the per-token edge is `edge = P_model - P_market` (with sign flipped for SELL-side trades).

| Strategy | Gate | Sizing | Primary metric |
|---|---|---|---|
| **General +EV** | `edge > 0.02` | flat $100 per trigger | total PnL, Sharpe |
| **Home-run** (primary for geopolitical markets) | `edge > 0.20` AND `time_to_settlement < 6h` AND `price < 0.30` | larger per trigger | precision@k, PnL concentration |

The two-strategy design reflects the shape of the underlying phenomenon. The Columbia paper (Mitts and Ofir 2026) documents that informed flow in Iran markets is **bursty and late-concentrated**, not diffuse. A general +EV rule catches the long tail; a home-run rule concentrates capital on the pattern the documented cases fit (short time-to-deadline, low implied probability, large edge). The home-run rule is the primary trading evaluation; the general rule is the robustness check.

**Cutoff-date sweep.** Run the streaming backtest with `N in {14, 7, 3, 1}` days before each deadline and plot PnL vs N. Expected shape: the home-run curve rises sharply as N shrinks, confirming that informed flow concentrates near the deadline.

**Calibration.** After training, isotonic regression on the held-out calibration slice. Report Brier score and ECE. Calibration matters because the edge math only works if `P_model = 0.8` actually means right 80 percent of the time.

**Frozen for test.** Gate thresholds, sizing, and any calibrator parameters are tuned on the validation slice and frozen before touching the test slice.

### 5.3 Unsupervised arm: autoencoder anomaly detection (Lecture 11)

- Undercomplete stacked autoencoder trained on all trade feature vectors, SELU activations, MSE loss.
- Anomaly score: per-trade reconstruction error.
- Purpose: a parallel, unsupervised lens on the data that does not use the correctness target. Cross-check whether trades with the largest `|p_hat − market_implied_prob|` gap also carry high reconstruction error.

### 5.4 Baselines (Lectures 4, 5, 6, 7)

- Logistic regression on the same features, producing a baseline `p_hat` for the same trading rule.
- Random forest, producing a baseline `p_hat` and feeding the feature-importance ranking used in Discussion.
- Isolation Forest as an unsupervised anomaly baseline against the autoencoder.
- Naive market baseline: trading rule with `p_hat = market_implied_prob`, which by construction produces zero gap and zero signal. This is the efficient-market null.

### 5.7 Validation strategy

Three layers, all on held-out data:

1. **Statistical.** ROC-AUC and calibration of `p_hat` and of the residualised gap against `bet_correct` on the validation and test slices. Brier-score improvement over the market-implied null. (Foundation §6 lists the required metrics.)
2. **Economic.** Cumulative PnL, Sharpe ratio, hit rate, maximum drawdown, and precision@k of each trading rule on the test slice only, against the baselines in §5.4. Streaming event-replay protocol — at each event, the decision uses only state strictly before the event timestamp.
3. **Named-case sanity check.** Magamyman (the Columbia paper's primary documented Iran-strike insider, ~$553K entering at 17 percent implied probability 71 minutes before news) serves as a named validation anchor. Pull the wallet address from the paper appendix or the `pselamy/polymarket-insider-tracker` GitHub repo and check: does the model assign high `p_hat` to his documented trades, do the home-run triggers fire on them, and which feature values does the model find most salient? This is an illustrative anecdote in the Discussion, not a Results target — labelling off a single wallet would introduce selection bias if used for model selection.

## 10. Deliverables and Next Steps (adventure-scope)

| # | Task | Status | Depends on |
|---|---|---|---|
| 13 | MLP training and baselines (logistic regression, random forest, isolation forest, naive market, autoencoder) | open | foundation row 6 |
| 14 | Isotonic calibration on validation slice | open | 13 |
| 15 | Streaming event-replay backtest — general +EV and home-run, cutoff-date sweep | open | 14 |
| 16 | Out-of-sample trading-rule evaluation on the test slice | open | 15 |
| 17 | Feature-importance and permutation importance for Discussion | open | 13 |
| 18 | Magamyman sanity check | open | 13 |

Outputs land under `outputs/modelling/pontus/<model>/` (`metrics.json`, `feature_list.json`, loss curve, calibration plot, PnL series). These feed foundation row 19 (report drafting) and the §12 convergence comparison in this file.

## 11. Open Decisions (adventure-scope)

- Exact edge thresholds for the general +EV rule (currently 0.02) and the home-run rule (currently 0.20) — may shift after validation tuning.
- Position-sizing rule for the home-run strategy: flat larger stake versus Kelly-scaled.
- Whether to model trade value in USD as a feature (already included via `trade_value_usd` and `log_size`) or additionally as a sample weight during training.

## 12. Convergence and Comparison

Both adventures train on the same cohorts (foundation §4) and report the same metric set (foundation §6), so the final report can table adventure-A vs adventure-B results side by side.

Specifically, the final comparison produces:

- A single results table with rows = {probability quality, gap quality, trading rule, unsupervised arm, complexity} and columns = {Alex, Pontus, best-of-both}.
- A calibration-curve overlay for the two `p_hat` series on the test slice.
- A PnL-curve overlay for the home-run strategy under each adventure.
- A short narrative in the report's Results section describing what each direction found, where they agreed, and where they diverged.

Divergences worth noting in the report — ranked by importance for the reader:

1. Model family (MLP vs whatever the other adventure picks).
2. Trading-rule parameterisation (gate thresholds, sizing rule).
3. XAI technique (permutation importance vs SHAP vs tree feature importance).
4. Unsupervised arm choice (autoencoder vs Isolation Forest vs something else).

Comparison lives in a shared notebook (to be created) that loads both adventures' `outputs/modelling/<adventure>/**/metrics.json` and builds the table + overlays. The notebook is referenced from foundation row 19.
