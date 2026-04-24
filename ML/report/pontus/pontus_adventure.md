# Pontus's Adventure — Modelling Plan

**Course:** Machine Learning and Deep Learning, CBS, Spring 2026
**Group:** Alejandro Laurlund Gato (161989), Alexander Myrup (160363), Pontus Neumann (185912), Linus Stamov Yu (160714)
**Title:** *Mispricing on Polymarkets*
**Subtitle:** *Detecting Probability Asymmetries in Iran Geopolitical Markets with Machine Learning*
**Document purpose:** Pontus's modelling direction. Builds on `foundation.md` (shared RQ, data, features, missing-data policy, EDA, evaluation). This file starts as a copy of the modelling sections from the pre-split project plan; Pontus edits it to reflect his chosen model family, hyper-parameter choices, and trading-rule tuning. Section numbers preserve the original `project_plan.md` numbering so cross-references to the foundation and to `alex_adventure.md` stay aligned.

**Companion documents:**
- `foundation.md` — shared project foundation. Read first. §4 lists the final feature set; §11 enumerates the leakage audit resolutions (P0-1, P0-2, P0-8, P0-9, P0-11, P0-12, Leak A/B/C) and the physical-drop finalisation.
- `alex_adventure.md` — the parallel modelling direction (for cross-reference and eventual comparison).
- `data-pipeline-issues.md` — canonical issues log.
- `alex/notes/session-learnings-2026-04-22.md` — Alex's diagnostic session bullets (P0-11 direction determinism, naive-market Simpson's paradox, residual-edge +0.06 partial correlation, tree memorisation via Layer 6).

---

## 0. Hard constraints inherited from the exam brief

- **TensorFlow / Keras only for all neural-network code.** The CBS MLDP exam
  rubric requires it. PyTorch is not permitted for the MLP or the autoencoder.
  sklearn baselines (LogReg, RF, GBM, Isolation Forest) are fine.
- The shared `scripts/12_train_mlp.py` is deprecated — it was PyTorch and read
  the old `split` column that no longer exists. Treat it as historical.

## 0.1 Data state this adventure builds on

- Consolidated dataset at `data/03_consolidated_dataset.csv` is the **finalised,
  leakage-free CSV**: 1,209,787 rows × **54 columns** (pre-drop snapshot preserved
  at `data/03_consolidated_dataset.pre_dropped_variables.csv`). After excluding
  IDs / labels / filter / benchmark columns, **~36 model features** remain.
- Feature groups retained (foundation §4): trade-local (`log_size`,
  `trade_value_usd`), market context normalised (`market_price_vol_last_1h`),
  time (`pct_time_elapsed` from `deadline_ts`), wallet-global
  (`wallet_prior_trades`, `wallet_prior_volume_usd`, `wallet_first_minus_trade_sec`,
  `wallet_prior_win_rate_causal`), wallet-in-market behavioural (bursting
  counts, `wallet_spread_ratio`, `wallet_is_whale_in_market`,
  `wallet_prior_trades_in_market`, `wallet_median_gap_in_market`), interactions
  (`size_vs_wallet_avg`, `size_vs_market_cumvol_pct`, `size_vs_market_avg`),
  Layer 6 on-chain identity (9 features, ends with `wallet_funded_by_cex_scoped`),
  Layer 7 cross-market entropy (`wallet_market_category_entropy`), six binary
  missingness indicators.
- Dropped and never reintroduced as model features: `side`, `outcomeIndex`
  (P0-11); `wallet_position_size_before_trade`, `trade_size_vs_position_pct`,
  `wallet_cumvol_same_side_last_10min`, `wallet_directional_purity_in_market`,
  `wallet_has_both_sides_in_market`, `market_buy_share_running` (P0-12);
  `is_position_exit`, `is_position_flip` (P0-2); `wallet_is_whale_in_market`
  is RETAINED but the definition is now causal expanding p95 (P0-1 fixed upstream);
  `wallet_prior_win_rate` (naive) is replaced by `wallet_prior_win_rate_causal`
  (P0-9).
- Cohort parquets at `data/experiments/{train,val,test}.parquet` are
  pre-filtered via §4 settlement filter and carry the 54-column schema
  (202,082 / 13,154 / 13,414 rows). Regenerate with
  `python scripts/14_build_experiment_splits.py`.
- `_check_causal.py` at 34 / 34 green as of 2026-04-22; run after any rebuild.

---

## 5. Method (adventure portions)

### 5.1 Primary model: MLP with market embedding (Lectures 8, 9)

- **Framework.** `tf.keras` Functional API with a two-input model. sklearn pipeline for the preprocessing stages (imputer, winsoriser, scaler). Random seed fixed at numpy, TensorFlow, and sklearn levels.
- **Architecture.** The primary model is a fully connected feed-forward network with a market-embedding input layer. `condition_id` is mapped to an integer index (0 … n_train_markets − 1) and passed through an `Embedding(n_train_markets, 4)` whose output is gated by a 0 / 1 mask input. The gated embedding is concatenated with the 36-feature vector and fed through three hidden layers of sizes [256, 128, 64] with SELU activation, LeCun-normal initialisation, batch normalisation, and dropout of 0.3. The output is a sigmoid `p_hat`. During training the mask is set to 1, so the embedding absorbs each training market's log-odds intercept; at inference on val and test the mask is 0, so the prediction is driven entirely by the feature trunk. Rationale: the per-market base-rate information that makes the finalised feature set 37 × more market-identifiable than chance (foundation §11, `pontus/notes/split-strategy.md`) is absorbed into the embedding at train time and stripped at test time. The test-set `p_hat` therefore has no per-market-intercept channel to exploit.
- **Optimiser.** Adam at learning rate 1e-3 with `ReduceLROnPlateau` on validation BCE; early stopping with patience 8 on validation BCE, not ROC (§5.7).
- **Input.** 36 standardised features from `data/experiments/train.parquet` (minus `NON_FEATURE_COLS` per §0.1). `market_implied_prob` is withheld so that `p_hat` is independent of the market's own belief and the gap `p_hat − market_implied_prob` is clean signal. Missingness: median impute on train fold only, then standardise; the six indicator columns are kept as explicit features so the model can read "was this NaN."
- **Winsorisation.** `trade_value_usd` and `wallet_prior_volume_usd` winsorised at the 1st / 99th percentile on the training fold only (§5.5 in foundation).
- **Output.** Predicted probability of settlement in favour of the trade, `p_hat`, computed with `mask = 0` on val and test to produce the market-identity-free prediction. The gap `p_hat − market_implied_prob` is the trading signal (§5.2). `p_hat` is calibrated via isotonic regression on the validation fold.
- **Class balance.** `bet_correct` rate is inside 35-65 % → `class_weight='balanced'` only. SMOTE not applied (rate is 0.52 in training).
- **Group sensitivity.** Because Layer 6 on-chain features can act as near-unique wallet identifiers (trees score 0.86-0.96 train / 0.52 test on raw features per Alex's sweep), evaluate with `GroupKFold(proxyWallet)` wherever within-training CV is reported. The current test cohort has 0 % wallet overlap with train, so the headline test ROC is already a pure novel-wallet generalisation number.
- **Baselines.** The embedding MLP is compared against two simpler families trained on the same 36 features: (a) logistic regression with L2 regularisation, (b) random forest (n_estimators = 400, min_samples_leaf = 200, `class_weight = "balanced"`). A 5-fold GroupKFold(proxyWallet) OOF stacked ensemble (LogReg + calibrated RF + base-MLP → LogReg meta) was implemented in `pontus/scripts/22_v2_pipeline.py` for methodological completeness but did not beat the embedding MLP on test ROC.
- **Negative result worth reporting — feature residualisation.** A per-market expanding-mean residualisation of the features was tested as an alternative way to neutralise the market-identity channel (`pontus/scripts/24_v2_residualised.py`). The residualised test ROC dropped from 0.582 to 0.543, and the 74-class market-identifiability audit on residualised features went UP from 37 × to 73 × random. The residual's higher-order structure (distribution shape, local scale, small-sample tail) fingerprints markets even more strongly than the raw feature; mean-centering alone is insufficient to remove multivariate distributional market-identity. The embedding architecture is the correct fix; residualisation is not.

### 5.2 Trading rule — two strategies evaluated side by side

For a candidate trade, the per-token edge is `edge = p_hat − market_implied_prob` (sign flipped for SELL-side trades via the side column recovered from the pre-drop snapshot or the cohort parquet — note that `side` and `outcomeIndex` are not model features but they are needed at trading-rule time to resolve edge sign, and both live in `data/experiments/*.parquet` alongside the features).

Time-to-deadline for the home-run gate is computed on the fly at trading-rule time as `(deadline_ts − timestamp)` in seconds. `deadline_ts` is in the CSV; `time_to_settlement_s` is not a model feature (P0-8), but that restriction applies to the model, not to the trading rule.

| Strategy | Gate | Sizing | Primary metric |
|---|---|---|---|
| **General +EV** | `edge > 0.02` | flat $100 per trigger | total PnL, Sharpe |
| **Home-run** (primary for geopolitical markets) | `edge > 0.20` AND `time_to_deadline < 6h` AND `market_implied_prob < 0.30` | larger per trigger | precision@k, PnL concentration |

The two-strategy design reflects the shape of the underlying phenomenon. The Columbia paper (Mitts and Ofir 2026) documents that informed flow in Iran markets is **bursty and late-concentrated**, not diffuse. A general +EV rule catches the long tail; a home-run rule concentrates capital on the pattern the documented cases fit (short time-to-deadline, low implied probability, large edge). The home-run rule is the primary trading evaluation; the general rule is the robustness check.

**Benchmark caveat (P0-4).** `market_implied_prob` falls back to the trade's own execution price on all 67 HF-path markets because CLOB history was not fetched for them (see foundation §11 known limitation). The residual-edge diagnostic in §5.7 is less affected because it re-projects `p_hat` against the same benchmark, but the absolute PnL on HF-path trades reads a price that is slightly biased by the trade itself. Mitigation: report the headline PnL restricted to `source == "api"` (the 7 ceasefire markets, ~14.6k trades, CLOB mid is real) alongside the full-cohort number.

**Cutoff-date sweep.** Run the streaming backtest with `N ∈ {14, 7, 3, 1}` days before each deadline and plot PnL vs N. Expected shape: the home-run curve rises sharply as N shrinks, confirming that informed flow concentrates near the deadline.

**Calibration.** After training, isotonic regression on the validation fold. Report Brier score and 15-bin ECE. Calibration matters because the edge math only works if `p_hat = 0.8` actually means right 80 percent of the time.

**Frozen for test.** Gate thresholds, sizing, and any calibrator parameters are tuned on the validation fold and frozen before touching the test fold.

### 5.3 Unsupervised arm: autoencoder anomaly detection (Lecture 11)

- **Framework.** `tf.keras` Model API (required by §0).
- **Architecture.** Undercomplete stacked autoencoder, symmetric encoder/decoder, SELU activations, MSE loss. Starting bottleneck 8 → 16 → 36 for a 36-feature input. Trained on all trade feature vectors from the training fold only.
- **Anomaly score.** Per-trade reconstruction error.
- **Purpose.** A parallel, unsupervised lens on the data that does not use the correctness target. Cross-check whether trades with the largest `|p_hat − market_implied_prob|` gap also carry high reconstruction error. An overlap significantly above the random-overlap null is the secondary evidence that the gap signal is anomalous (not just model variance).

### 5.4 Baselines (Lectures 4, 5, 6, 7)

All baselines use `sklearn`. All consume the same 36 features, same train / val / test fold assignment, same winsorisation and imputation.

- **Logistic regression (L2).** Linear baseline, produces a baseline `p_hat` for the same trading rule. Coefficients are the feature-importance ranking in §5.7.
- **Random forest.** Tree baseline. Must use heavy regularisation — Alex observed `RandomForestClassifier()` defaults reaching 0.96 train vs 0.52 test via Layer-6-driven wallet memorisation. Starting point: `n_estimators=400, min_samples_leaf=200, max_features="sqrt"`; tune via `GroupKFold(proxyWallet)`.
- **Isolation Forest.** Unsupervised anomaly baseline against the autoencoder. Overlap-at-top-decile comparison against a random-overlap null is the Lecture-11 metric.
- **Naive market.** Trading rule with `p_hat = market_implied_prob`. By construction this produces zero gap and zero signal. **Important:** the aggregate ROC of the naive-market baseline is not 0.5 in this dataset — Alex showed it reaches 0.63 on test via a Simpson's paradox through the `(side × outcomeIndex)` distribution, even though the efficient-market null implies calibration, not ROC 0.5. The honest null comparison is therefore in terms of (a) Brier / ECE against the model, and (b) **residual edge ROC** (`p_hat` residualised against `market_implied_prob`), where the naive baseline goes to 0.5 by construction. Report both; see §5.7.

### 5.7 Validation strategy

Four layers, all on held-out data:

1. **Statistical (probability-level).** ROC-AUC, PR-AUC, Brier, and 15-bin ECE of `p_hat` on val and test folds. Brier improvement over the market-implied null is the efficient-market-null test (calibration, not ROC).
2. **Residual-edge (gap-level; RQ1b test).** Compute the residual of `edge = p_hat − market_implied_prob` after linearly projecting out `market_implied_prob` on the training fold, and report the residual's ROC-AUC and partial correlation with `bet_correct` on val and test. This is the honest gap-signal test that aggregate ROC comparisons miss (the naive-market baseline scores 0.63 test ROC via Simpson's paradox — see §5.4 and Alex's investigation at `alex/outputs/investigations/naive_market/`). The target to beat is the residual-edge partial correlation of ~+0.06 (~7σ) reported in `alex/outputs/investigations/residual_edge/`.
3. **Economic (trading-rule level).** Cumulative PnL, Sharpe ratio, hit rate, maximum drawdown, precision@k of each trading rule on the test fold only, against the baselines in §5.4. Streaming event-replay protocol — at each event, the decision uses only state strictly before the event timestamp. PnL reported both on the full test cohort AND restricted to `source == "api"` (unbiased CLOB benchmark, ~14.6k trades — see §5.2 P0-4 caveat).
4. **Robustness (stability and wallet-generalisation).** (a) Within-training 5-fold `GroupKFold(proxyWallet)` ROC to verify the model is not memorising wallets via Layer 6; (b) test-fold ROC split by "wallet seen in training" vs "novel wallet"; (c) bootstrap 95 % confidence intervals on test metrics at the `proxyWallet` level (not at the trade level) to account for within-wallet correlation in bursts.
5. **Named-case sanity check.** Magamyman (the Columbia paper's primary documented Iran-strike insider, ~$553K entering at 17 percent implied probability 71 minutes before news) serves as a named validation anchor. Pull the wallet address from the paper appendix or the `pselamy/polymarket-insider-tracker` GitHub repo and check: does the model assign high `p_hat` to his documented trades, do the home-run triggers fire on them, and which feature values does the model find most salient? This is an illustrative anecdote in the Discussion, not a Results target — labelling off a single wallet would introduce selection bias if used for model selection.

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

- **MLP hidden-layer sizes.** Start at [256, 128, 64]; shrink if val BCE
  plateaus early.
- **Edge thresholds.** General +EV at 0.02 and home-run at 0.20 are placeholders;
  tune on the validation fold before freezing for test.
- **Position-sizing rule for the home-run strategy.** Flat larger stake versus
  Kelly-scaled. Default: flat. Kelly comparison as robustness.
- **Trade value weighting.** Model trade value in USD only through
  `trade_value_usd` and `log_size` features, OR additionally as a sample weight
  during training. Default: features only.
- **XAI technique for the Discussion.** Permutation importance on the test fold
  is the simplest (low compute, no retraining); SHAP on a 10k-trade sample is
  the richer alternative. Alex's workspace may pick a tree-importance angle —
  aim for a different technique here so the §12 comparison is informative.
- **Unsupervised arm emphasis.** Whether the autoencoder drives a secondary
  Results claim or stays a Discussion-only cross-check. Depends on whether the
  gap-anomaly overlap materially beats the random-overlap null.
- **Differentiation from Alex.** Tentative split: Pontus = deeper MLP + SHAP +
  autoencoder; Alex = shallower MLP + tree importance + Isolation Forest. Revisit
  before first convergence write-up.

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
