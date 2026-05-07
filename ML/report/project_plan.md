# Project Plan — Mispricing on Polymarkets

**Course:** Machine Learning and Deep Learning, CBS, Spring 2026 (KAN-CDSCO2004U)
**Group:** Alejandro Laurlund Gato (161989), Alexander Myrup (160363), Pontus Neumann (185912), Linus Stamov Yu (160714)
**Title:** *Mispricing on Polymarkets*
**Subtitle:** *Detecting Probability Asymmetries in Iran Geopolitical Markets with Machine Learning*

**Document purpose.** Single source of truth for the project after the 2026-05-03 clean-up. Replaces and supersedes `foundation.md`, the prior `project_plan.md`, and `data-pipeline-issues.md`. Reflects the finalized pipeline that is now hand-in-ready in `submission/`.

**Companion documents.**
- `paper_guidelines.md` — grading rules, formatting, LLM-disclosure template, submission checklist.
- `submission/README.md` — how to run the pipeline end-to-end.
- `submission/data/MISSING_DATA.md` — per-feature missing-value handling.
- `archive/` — preserved historical material (Alex pipeline, working scripts, EDA outputs, prior plans, leakage audit, .docx snapshots).

---

## 1. Summary

The project studies whether a machine-learning model trained on pre-trade behavioral and market-state features can produce a probability of trade success (`bet_correct`) that systematically diverges from the contemporaneous market-implied probability on resolved Iran geopolitical Polymarket events, and whether that gap supports a profitable trading rule out of sample.

The dataset covers all 74 resolved sub-markets across four Iran-related event clusters (Polymarket events 114242, 236884, 355299, 357625), 1,371,180 trades by 109,080 wallets between 2025-12-22 and 2026-04-19. Train and test are market-cohort-disjoint and temporally separated: 63 strike-ladder markets pre-2026-02-28 for training, 10 ceasefire-ladder markets post-strike for test.

The submission compares seven supervised models (logistic regression L1/L2, decision tree, random forest, histogram gradient boosting, LightGBM, PCA→LogReg, sklearn MLP) under 5-fold GroupKFold cross-validation, with Isolation Forest retained as an unsupervised diagnostic. Each model's raw probabilities are isotonic-calibrated against out-of-fold predictions, then evaluated on a realistic capital-aware backtest with gas, slippage, concentration limits, and a naive market-favorite baseline as the falsification control. A hyperparameter sweep with Optuna (TPE, MedianPruner) runs in parallel for Random Forest, HistGBM, and MLP.

A secondary, out-of-scope observation is reserved for the Discussion: if feature-importance analysis shows the signal concentrates on features that resemble those associated with informed trading in the literature (abnormal bet size, pre-event timing, wallet newness, directional concentration), the report flags the parallel and notes it as future work.

## 2. Research Questions

### Scope and purpose

This project studies the gap between Polymarket's contemporaneous market-implied probabilities and the probabilities predicted by a machine-learning model on resolved Iran geopolitical markets. The scope covers all 74 resolved sub-markets under Polymarket events 114242 (US-strikes), 236884 (Iran x Israel/US conflict ends), 355299 (US x Iran ceasefire announcement), and 357625 (US x Iran ceasefire extension). Trade correctness is derived from official resolution.

### Short form

**RQ1.** Can a machine-learning model trained on pre-execution market-state and behavioral features produce a probability estimate that systematically differs from the contemporaneous market-implied probability in settled Iran prediction markets, such that a trading rule acting on the gap generates positive risk-adjusted returns out of sample?

### Detailed form

**RQ1a — Probability gap.** Does a supervised model trained on pre-execution features (market state, recent market activity, wallet history, on-chain wallet identity) produce a probability estimate `p_hat` whose residual against the contemporaneous market-implied probability `pre_yes_price_corrected` predicts `bet_correct` on a temporally held-out test set?
*Success criterion:* ROC-AUC of `p_hat` on the test cohort is materially above 0.5, with calibrated Brier-score improvement over the market-implied null and an improved reliability diagram.

**RQ1b — Trading rule.** Does a portfolio of capital-aware betting strategies acting on `p_hat` (high-confidence thresholds, top-K by edge, EV-positive filters) generate positive cumulative ROI out of sample, relative to a naive market-favorite baseline that requires no model?
*Success criterion:* positive ROI on the test cohort under realistic frictions (gas, slippage, concentration limits) on at least one strategy, and a strict ROI improvement over the naive baseline on the same strategy.

## 3. Framing Decision

The efficient-market null for Polymarket is that the market-implied price already incorporates all public information, so no pre-execution feature set should add predictive power. A systematic gap between `p_hat` and `pre_yes_price_corrected` that predicts settlement is therefore direct evidence of mispricing, and acting on the gap monetizes it.

This framing makes the target observable (settlement is public), keeps the evaluation economic and quantitative (ROI, drawdown, hit rate), and avoids the labeling circularity of trying to pre-label wallets as informed. Any resemblance between the features driving the signal and informed-trading traits is reserved for the Discussion, not the research question.

## 4. Data

| Item | Detail |
|---|---|
| Markets | All 74 resolved sub-markets under Polymarket events 114242, 236884, 355299, 357625. 19 YES / 55 NO by end-of-market token prices. |
| Unit of analysis | One resolved trade. Final dataset: 1,371,180 trades (1,114,003 train + 257,177 test). |
| Sources | Hybrid pipeline: (a) market metadata via Polymarket Gamma API, (b) trade history for 67 markets via the HuggingFace mirror `SII-WANGZJ/Polymarket_data` (38.7 GB, streamed via duckdb httpfs), (c) trade history for 7 ceasefire markets via the Polymarket Data API, (d) CLOB mid-price for ceasefire markets, (e) on-chain wallet enrichment via Etherscan V2 (108,621 of 109,080 wallets enriched, 99.58% coverage). |
| Target | `bet_correct ∈ {0, 1}` from market resolution and the side of the trade. Pos rate 50.31% train, 50.37% test (no resampling needed). |
| Benchmark | `pre_yes_price_corrected` — pre-trade YES-token price reconstructed per-market and bundled in `submission/data/backtest_context.parquet` (fixes the per-token vs YES-normalized price bug that previously inflated ROI by ~400×). |
| Feature inventory | 87 columns total: 5 meta/id (`split`, `market_id`, `ts_dt`, `timestamp`, `bet_correct` — the last is the target) + 70 core features + 12 wallet features. The submission's `01_data_prep.py` produces a modeling feature list of **80 features** after dropping the 5 meta/id columns (which include the target) and the forbidden lifetime flags filtered at load time per `FORBIDDEN_LEAKY_COLS` (§6.2). The 4 low-signal CEX features (`wallet_funded_by_cex_scoped`, `wallet_cex_usdc_cumulative_at_t`, `wallet_log_cex_usdc_cum`, `wallet_n_cex_deposits_at_t`) are kept in the feature list for inventory completeness; the 2026-04-29 audit recommended dropping them on signal grounds (mutual information ≤ 0.0014). See **§6 Open decisions**. |
| Feature groups | (1) trade-local — `log_size`, `side_buy`, `outcome_yes`, payoff/distance/count features; (2) time/cyclical — `pct_time_elapsed`, time-to-deadline, hour-of-day sin/cos; (3) pre-trade price + price-change features at 5min/1h/24h windows; (4) microstructure — Kyle's λ, realized vol, jump component, signed OI autocorrelation; (5) order flow — imbalance at 5min/1h/24h, YES volume share, taker side skew; (6) behavior derivatives — contrarian/consensus scores, risk-reward, long-shot indicator; (7) per-taker history — log priors, cumvol, unique markets, directional purity, position size, yes-share; (8) on-chain wallet identity — Polygon age, nonce, inbound count, USDC funding history. |
| Pre-modeling filter | Already applied upstream: post-resolution close-out trades (`settlement_minus_trade_sec ≤ 0`, ~16.5% of raw rows) dropped before consolidation. |
| Split | Market-cohort-disjoint and temporally separated. **Train:** 63 markets, 1,114,003 trades, all `timestamp` < 2026-02-28 06:34:59 UTC (Operation Epic Fury launch). **Test:** 10 markets, 257,177 trades, all `timestamp` ≥ 2026-02-28 14:02:57 UTC. Zero market overlap. K-fold inside train uses `GroupKFold(n_splits=5)` on `market_id`. The cohort-disjoint split is the leak-defense mechanism for the within-market direction-determinism channel previously surfaced (see §5 Leakage policy). |
| Class balance | 0.503 positive in both splits; no resampling, no SMOTE, no `class_weight` adjustments at the data level (some sklearn estimators set `class_weight="balanced"` internally as a safety net but it has no effect on this balanced dataset). |
| Single source of truth | `submission/data/consolidated_modeling_data.parquet` (303 MB, 1.37M × 87) for modeling, plus `submission/data/backtest_context.parquet` (20 MB) for backtest-only price, liquidity, and taker diagnostics. The `wallet_enrichment.parquet` table (355 MB) is already joined into the consolidated parquet; it lives in `archive/` for traceback only. |

## 5. Method

The full pipeline lives in `submission/scripts/`. Each script is a teacher-friendly merge of the working files previously scattered across `archive/alex/v5_final_ml_pipeline/scripts/`. Source-of-truth merge tables are at the top of every script.

### 5.1 Pipeline overview (six numbered scripts + EDA)

| Script | Stage | What it produces |
|---|---|---|
| `01_data_prep.py` | Load + leakage checks | `feature_cols.json` (80 features), `leakage_report.json` with modeling and backtest-context checks |
| `02_features.py` | Taxonomy + scaler + Isolation Forest diagnostic | `feature_taxonomy.json`, `scaler.joblib`, `iso_forest_scores.parquet` |
| `03_train_models.py` | Train 7-8 models with 5-fold GroupKFold + complexity benchmark | `metrics.json`, `preds_oof.npy`, `preds_test.npz` per model; `comparison.csv`, `complexity.csv` |
| `04_calibration.py` | Isotonic recalibration + reliability diagrams + bootstrap CI + permutation importance | `preds_test_cal.npz` per model; `calibration_summary.csv`, `auc_bootstrap_ci.csv`, `permutation_importance_<best>.csv`, reliability PNGs |
| `05_backtest.py` | Capital-aware sim across 540 cells + naive baseline + diagnostics + headline figure | `sensitivity.csv`, `falsification.json`, `diagnostics_residual_edge.csv`, `diagnostics_consensus.csv`, `diagnostics_sell_semantics.json`, `overview.png` |
| `06_tuning_optuna.py` | Optuna TPE for RF / HistGBM / MLP | `best_params.json`, `study_history.csv`, `comparison_vs_default.json`, `preds_test_tuned.npz` per tuned model |
| `report_tools/eda.py` | EDA (work in progress) | 19 figures + tables written directly to `submission/report_assets/figures/appendix/eda/` |

### 5.2 Models (curriculum mapping in parentheses)

- **Logistic regression L2** (Lecture 4) — linear anchor.
- **Logistic regression L1** (Lecture 4) — sparse / feature selection.
- **Decision Tree** (Lecture 5) — interpretable single tree.
- **Random Forest** (Lecture 5) — bagged ensemble.
- **Histogram Gradient Boosting** (Lecture 5) — fast boosting.
- **LightGBM** — alternative gradient-boosting library; included if installed.
- **PCA(K = elbow) → Logistic Regression** (Lectures 4 + 5) — dimensionality reduction. K is chosen by the geometric-elbow method on the cumulative variance curve, not a magic 0.95 threshold.
- **MLP (sklearn)** (Lectures 8, 9) — `(64, 32)` ReLU, Adam, early stopping. The Keras MLP previously planned was swapped for sklearn after `model.fit` deadlocked on the local stack; the architecture and regularization strength are equivalent for this 80-feature input.
- **Isolation Forest** (Lecture 7) — UNSUPERVISED diagnostic. The score is written by `02_features.py` for outlier-detection coverage and optional discussion, but it is not joined into the supervised feature matrix or headline model table.

### 5.3 Cross-validation and calibration

5-fold `GroupKFold(market_id)` so no market spans train and val of any fold. The same group constraint is also the leak-defense rationale for the train/test split (no market in both sides), making fold-internal validation directly representative of test-time generalization.

Per model the script saves:
- `preds_oof.npy` — out-of-fold predictions on train rows.
- `preds_test.npz` — raw test predictions from the full-train refit.

`04_calibration.py` then fits `IsotonicRegression(out_of_bounds="clip")` on (oof, y_train) and applies it to raw test predictions, producing `preds_test_cal.npz`. Reliability diagrams (raw vs calibrated vs perfect-calibration line) are saved per model and combined into one figure for the report.

### 5.4 Backtest

`05_backtest.py` first attaches `submission/data/backtest_context.parquet`, then simulates ten betting strategies across a parameter grid. The sidecar is required because the modeling parquet intentionally omits raw fields that are not features, including true `usd_amount`, raw taker-side fields, and corrected `pre_yes_price_corrected`.

- **Confidence thresholds:** `phat_gt_0.99`, `phat_gt_0.95`, `phat_gt_0.9`.
- **Top-K by score:** `top1pct_phat`, `top1pct_edge`, `top5pct_edge`.
- **Expected-value rules:** `general_ev` (edge > 2 cents), `general_ev_late` (+ near-deadline), `general_ev_cheap` (+ low cost).
- **Home run:** high edge, low cost, near deadline (asymmetric-information hypothesis).

Each (model × strategy) cell is run across the grid `initial_capital ∈ {1k, 10k, 100k}` × `max_bet_pct ∈ {1%, 5%, 10%}` × `liquidity_scaler ∈ {1.0, 0.10}` (no copycats vs. 10× copycats sharing fill), giving 18 scenarios per cell. The execution loop applies a 5% cost floor (caps payoff at 19×), $0.50 gas per trade, 5% slippage above the 25%-of-trade-volume threshold, a 20% concentration cap per market, and chronological capital release on resolution.

A **naive consensus baseline** (`p_hat` = market-implied probability) runs the same strategies and is compared in `falsification.json`. Each ML model must beat this free heuristic for the strategy to count as model-driven alpha.

The same script also writes three compact diagnostics for report robustness: `diagnostics_residual_edge.csv` tests whether `p_hat - market_prob` retains signal after controlling for market-implied probability; `diagnostics_consensus.csv` decomposes top picks into consensus-following versus contrarian selections; `diagnostics_sell_semantics.json` checks whether SELL rows look like closing trades rather than fresh directional bets.

The headline figure `overview.png` is the heatmap of ROI per (strategy, model) at the headline scenario ($10K, 5% bet, no copycats).

### 5.5 Hyperparameter tuning

`06_tuning_optuna.py` runs Optuna TPE with `MedianPruner` for Random Forest, HistGBM, and the sklearn MLP. RF and HistGBM use 5-fold GroupKFold; MLP uses a single `GroupShuffleSplit` (80/20) holdout because per-fold MLP fit time made full KFold infeasible overnight. Per model the script saves `best_params.json`, `study_history.csv`, `comparison_vs_default.json`, and `preds_test_tuned.npz`.

The 2026-04-30 tuning run produced (test AUC, raw):
- **Random Forest:** tuned 0.7751 vs default 0.8987 → **−0.124** (the default's AUC of 0.899 is suspicious; the tuned run is the more honest number).
- **MLP (sklearn):** tuned 0.8047 vs default 0.8021 → **+0.0027** (effectively a wash).
- **Backtest:** tuned RF is worse than default on 8 of 10 strategies; tuned MLP is worse on high-confidence strategies but better on broad-net strategies (`general_ev_cheap` +59.8pp, `general_ev` +45.6pp), consistent with the missing isotonic refit on tuned predictions shifting probability mass.
- Per the team agreement, the **default** model is the headline; the tuned overview is presented as a side-by-side artifact in `submission/report_assets/figures/main/overview_tuned.png`.

### 5.6 Complexity benchmark

Required by the guidelines. `03_train_models.py:benchmark_complexity()` measures wall-clock fit time, predict time per 1,000 rows (median over 3 runs), and a parameter-count proxy per model (coefficient size for linear, total leaf count for forests, parameter count for MLP). Output: `outputs/metrics/complexity.csv`.

## 6. Leakage and safety policy

Consolidated from the 2026-04-29 leakage audit (full historical log preserved in `archive/data-pipeline-issues.md`). The submission's `01_data_prep.py` enforces the policy in code; this section documents the rationale.

### 6.1 Cohort-disjoint split as the headline defense

The train/test split is by market cohort, not by trade timestamp inside a market. With 63 train markets and 10 test markets and zero overlap, no classifier can memorize per-market resolution from train rows and apply it to test rows of the same market. This protection lets the modeling feature set retain `side_buy`, `outcome_yes`, `taker_directional_purity_in_market`, `taker_position_size_before_trade`, and `market_buy_share_running` — features that would otherwise re-open the within-market direction-determinism leak (the previous P0-11 / P0-12 channels). Fold-internal validation uses `GroupKFold(market_id)` so the same protection applies during CV.

### 6.2 Hard-excluded forbidden columns

`01_data_prep.py:FORBIDDEN_LEAKY_COLS` excludes four columns from every modeling feature list:

| Column | Why excluded |
|---|---|
| `kyle_lambda_market_static` | Definitional leak — fit on each market's first half and broadcast to all rows; trades in the first half see post-trade information from the same market. |
| `wallet_funded_by_cex` | Lifetime CEX-funding flag; uses post-trade events. The point-in-time variant `wallet_funded_by_cex_scoped` is causally clean and retained. |
| `n_tokentx` | Lifetime transaction total; peeks at post-trade activity. |
| `wallet_prior_win_rate` | Naive cumulative mean over priors regardless of resolution time. The causal variant `wallet_prior_win_rate_causal` (only priors with `resolution_ts < t`) is retained. Empirical leak component on train: Pearson r = 0.367 (leaky) vs 0.236 (causal), a +0.131 leak-driven component on the strongest single linear correlate. |

### 6.3 Causality checks (`01_data_prep.py`)

- **S1** — train + test row counts match the released contract (1,114,003 + 257,177).
- **S2** — class balance inside the 35-65% band (no resampling needed).
- **C1** — no train trade is timestamped after the strike event; no test trade is timestamped after the ceasefire announcement.
- **F3** — none of the four forbidden columns is allowed into the feature list (they may exist in the parquet for traceback completeness, but get filtered out at load time).
- **D1** — `pre_trade_price` matches `price.shift(1)` per market on > 99.5% of rows (verifies the upstream one-bar shift is correct).

All five checks pass on the bundled dataset.

### 6.4 Per-token vs YES-normalized price (D-029)

The HF `price` field is per-token (each token has its own price), not YES-normalized. The naive `pre_trade_price = price.shift(1)` would give the previous trade's per-token price, which could be either side of the market. `compute_pre_yes_price_corrected()` in `05_backtest.py` tracks last-seen token1 and token2 prices separately, shifts to exclude the current trade, and reconstructs YES probability as `last_t1` (else `1 - last_t2`, else 0.5). Discovered 2026-04-28 as the cause of a ~400× ROI inflation in the previous backtest.

### 6.5 Open decisions

- **Feature count: 80 vs 76.** The submission keeps 80 features; the 2026-04-29 audit recommended dropping 4 additional low-signal CEX features (mutual information ≤ 0.0014, marginal hit-rate diff ≤ 0.6 pp), which would yield a 76-feature inventory. Either is defensible. Drop them if the report wants a single tighter feature inventory.
- **`market_implied_prob` source asymmetry.** 100% of HF rows fall back to the trade's execution price (no CLOB history pulled for events 114242 and 236884); only 3% of API rows match `price` (the rest use CLOB mid). Not a feature leak (the column is excluded from features), but the trading-rule benchmark is the trade's own execution price on HF rows. To cite as a methodology limitation; or fetch CLOB history for the 67 HF markets (~1 h network) for a rigorous fix.

## 7. Missing-data policy

The modeling parquet contains zero NaN cells across all 1,371,180 rows × 87 columns. Structural missingness — features that would otherwise be mathematically undefined given the prior history available at row time — is resolved by substitution with semantically meaningful constants at the feature-engineering stage rather than retained as NaN. The substitution rules:

- **Cumulative `log_*` counts and volumes** take value 0 on a wallet's first-ever trade (`np.log1p(0) = 0` is the correct mathematical value).
- **Per-(market, taker) features** (`taker_directional_purity_in_market`, `taker_position_size_before_trade`, `log_taker_prior_trades_in_market`) take value 0 on the first trade in a market via an explicit first-row reset before `cumsum().shift(1)`.
- **Cross-market wallet share** (`taker_yes_share_global`) takes 0.5 in the absence of prior trades — a neutral prior with no directional information.
- **Wallet on-chain features** (Polygon age, nonce, inbound count, USDC funding) take value 0 on rows whose wallet failed Etherscan V2 enrichment (459 of 109,080 wallets, ~0.31% of trades).

Two binary indicator columns mark the most informative missingness species so the model can route substituted-zero cases differently from observed-zero cases: `wallet_enriched` (1 if Etherscan enrichment succeeded) and `taker_first_trade_in_market` (1 on a wallet's first trade in this market). Per-feature substitution rules and the change log live in `submission/data/MISSING_DATA.md`.

The convention is the constant strategy in `sklearn.impute.SimpleImputer` taught in the course (Lecture 2 preprocessing) and documented in Géron (2022, §2). The framework remains compatible with Rubin (1976) and Little & Rubin (2019), with the "no information yet" cases treated as a structural sub-type beyond the classic MCAR/MAR/MNAR taxonomy.

## 8. Evaluation Metrics

| Layer | Metrics | Where |
|---|---|---|
| Probability quality | ROC-AUC, Brier score, Expected Calibration Error (ECE) — raw and calibrated | `outputs/models/<m>/metrics.json`, `outputs/metrics/calibration_summary.csv` |
| Confidence intervals | 1,000-resample bootstrap 95% CI on test AUC | `outputs/metrics/auc_bootstrap_ci.csv` |
| Reliability | Per-model and combined reliability diagrams | `outputs/metrics/reliability_*.png` |
| Feature attribution | sklearn permutation importance (top-15) for the best model | `outputs/metrics/permutation_importance_<best>.csv` |
| Trading rule | Per cell: number of signals, number executed, final capital, ROI, max drawdown | `outputs/backtest/sensitivity.csv` |
| Falsification | Best ML model vs. naive market-favorite baseline per strategy | `outputs/backtest/falsification.json` |
| Residual signal | Residual-edge AUC and partial correlation after controlling for market probability | `outputs/backtest/diagnostics_residual_edge.csv` |
| Claim decomposition | Top-pick consensus/contrarian shares and SELL-semantics check | `outputs/backtest/diagnostics_consensus.csv`, `outputs/backtest/diagnostics_sell_semantics.json` |
| Headline | ROI heatmap (strategy × model) at $10K bankroll, 5% max bet, no copycats | `outputs/backtest/overview.png` |
| Complexity | Fit time, predict time per 1,000 rows, parameter count proxy | `outputs/metrics/complexity.csv` |

## 9. Interpretability and Ethics

Lecture 14 treats XAI conceptually rather than as specific techniques. The report uses:
- Permutation-importance ranking on the best calibrated model (`04_calibration.py`).
- The reliability diagrams from §8 as the calibration / "does 0.8 mean 80%" check.
- A traceability + accuracy + understanding framing paragraph from Lecture 14.

The ethics section anchors on a concrete policy tension. In November 2025 Polymarket CEO Shayne Coplan described insider edge on *60 Minutes* as "a good thing" and "an inevitability." On 23 March 2026 Polymarket announced explicit rules prohibiting (1) trading on stolen confidential information, (2) trading on illegal tips, and (3) trading by anyone in a position of authority over the event outcome. The section covers:

- **Privacy.** Pseudonymous on-chain wallets are persistent; pattern-linking is de-anonymising. Features are aggregated and no individual wallet list is published. The Magamyman case in Discussion uses information already public in the Columbia paper (Mitts and Ofir 2026).
- **Dual use.** A trained model could help regulators detect manipulation or help platforms surveil users.
- **Label validity.** "Informed trading" vs "skill" vs "luck" is genuinely ambiguous. `bet_correct` is a probabilistic signal, not a legal determination.
- **Enforcement gap.** Despite public documentation of Magamyman, Burdensome-Mix, and the Biden-pardons wallets, no publicly disclosed wallet has been banned and no profits clawed back as of the cut-off date.
- **Dataset bias.** Selected markets are high-volume English-language geopolitical contracts on a specific event cluster; findings do not necessarily generalise.
- **Platform surveillance context.** Polymarket's own ML detection stack ("Vergence", Palantir + TWG AI) launched 2026-03-10 but is scoped to sports markets. Geopolitical markets — the domain of this project — have no publicly disclosed ML surveillance, which is one reason the work is worth doing.
- **LLM usage disclosure.** Required by the exam brief. The Contribution and LLM Usage Disclosure section of the report documents which parts of the pipeline were co-authored with Claude. Template in `paper_guidelines.md`.

## 10. Out of Scope — Reserved for Discussion and Future Work

- **Feature-importance parallel to informed-trading traits.** If features driving the gap resemble those documented in the informed-trading literature (within-trader bet size, pre-event timing, directional concentration, wallet newness; see Mitts and Ofir 2026), Discussion flags the parallel and notes the signal may be partially picking up informed flow.
- **Documented-case anecdotes** (Burdensome-Mix on Maduro, Biden-pardons wallets) are illustrative, not evaluation targets.
- **Cross-market sibling-price injection.** A second-order arbitrage feature layer using prices of related sub-markets (e.g. "by Feb 14" price as a feature for the "by Feb 28" market) is deferred.
- **Cross-event-family pooling** with non-Iran clusters (Maduro, Biden pardons, Taylor Swift) is deferred.
- **GDELT news-timing enrichment** is deferred.
- **Orderfilled event-level analysis.** The HF mirror also exposes `orderfilled_part1-4.parquet` (raw on-chain OrderFilled logs, ~125 GB combined). The derived `trades.parquet` carries the aggregated per-trade records the project needs; raw orderfilled is reserved for future maker/taker attribution at order level.
- **Magamyman sanity check** — pull Magamyman's wallet from the Columbia appendix or `pselamy/polymarket-insider-tracker`, check whether the best model assigns high `p_hat` to his documented trades. Illustrative anecdote in Discussion only, not a Results target.
- **CLOB history for HF markets** — would resolve the `market_implied_prob` source asymmetry noted in §6.5.

## 11. Repository Layout

```
ML/report/
├── KAN-CDSCO2004U_*.docx, .pdf       <- the actual hand-in document
├── paper_guidelines.md                <- grading rules + LLM disclosure template
├── project_plan.md                    <- this file (single source of truth)
├── Design.md                          <- script naming + code-style conventions
├── .env.example                       <- template for Etherscan V2 keys
│
├── submission/                        <- the code+data hand-in bundle
│   ├── README.md                      run instructions + script-merge map
│   ├── requirements.txt               pinned dependencies
│   ├── data/                          single source of truth (303 MB)
│   │   ├── consolidated_modeling_data.parquet
│   │   ├── consolidated_modeling_data.info.json
│   │   ├── README.md, MISSING_DATA.md, release-manifest-2026-04-29.md
│   ├── scripts/
│   │   ├── config.py                  central paths + seeds + N_FOLDS
│   │   ├── 01_data_prep.py            load + leakage checks + feature list
│   │   ├── 02_features.py             taxonomy + scaler + Isolation Forest score
│   │   ├── 03_train_models.py         7-8 models + GroupKFold + complexity benchmark
│   │   ├── 04_calibration.py          isotonic + reliability + bootstrap CI + permutation importance
│   │   ├── 05_backtest.py             realistic backtest + naive baseline + headline figure
│   │   └── 06_tuning_optuna.py        Optuna TPE for RF / HistGBM / MLP
│   ├── outputs/                       regenerated when scripts re-run (gitignored upstream)
│   │   └── data/, models/, metrics/, backtest/, tuning/
│   └── report_assets/                 frozen artifacts cited in the .docx
│       ├── figures/main/              overview_baseline.png, overview_tuned.png, capital_curves.png
│       ├── figures/appendix/eda/      19 EDA figures + tables + eda_appendix.docx
│       └── tables/                    tuning_summary.md, backtest_sensitivity.csv, tuned_*.json
│
├── report_tools/                      <- live docx automation (NOT graded)
│   ├── README.md                      what each script does + run order
│   ├── eda.py                         regenerate EDA into submission/report_assets/
│   ├── build_eda_appendix.py          pack EDA figures into eda_appendix.docx
│   ├── 28_finalise_report.py          finalise pass over the .docx body
│   ├── 29_pin_appendix_tables.py      pin appendix tables
│   ├── 30_us_spelling_and_table_fix.py US spelling + table-cell fixes
│   ├── 31_fix_page_total.py           page-total field fix
│   ├── 36_remove_stray_logo.py        remove a stray logo from one body page
│   └── backup/                        .pre_*.docx rollback snapshots
│
└── archive/                           <- preserved historical material
    ├── data-pipeline-issues.md        full P0/P1/P2 log + 2026-04-29 audit (folded into §6)
    ├── foundation.md                  pre-clean-up shared foundation (folded into §1-3, §9)
    ├── alex/                          full v5 pipeline + tuning + EDA scripts
    ├── scripts/                       docx-postprocessing scripts (.py)
    ├── outputs/                       v2/v3/v4 backtests, leakage diagnostics, EDA dumps
    ├── data_old/                      original data folder including wallet_enrichment.parquet (355 MB)
    ├── backup/, assets/, notes/, handovers/, guidelines/
    └── ML_final_exam_paper.pre*.docx  13 historical .docx snapshots
```

## 12. Reproduction workflow

From `ML/report/`:

```bash
pip install -r submission/requirements.txt
cd submission/scripts/
python 01_data_prep.py            # ~30 s   leakage + feature list
python 02_features.py             # ~5 min  IsoForest + scaler
python 03_train_models.py         # ~60 min CV + complexity benchmark
python 04_calibration.py          # ~10 min calibration + bootstrap CI + permutation importance
python 05_backtest.py             # ~5 min  capital-aware sim
python 06_tuning_optuna.py        # ~120 min Optuna TPE for RF + HistGBM (MLP optional)
```

Scripts can be run independently after `01_data_prep.py` produces `feature_cols.json`. `06_tuning_optuna.py` is optional and the report uses the side-by-side baseline-vs-tuned figure already in `submission/report_assets/figures/main/`.

## 13. Status as of 2026-05-07

- **Data pipeline:** done. Single consolidated modeling parquet shipped, with `backtest_context.parquet` added for corrected YES prices, real trade liquidity, and taker-side diagnostics. Row counts, target balance, forbidden-column exclusion, and context alignment are verified by `01_data_prep.py`.
- **Modeling pipeline:** done. Six numbered scripts in `submission/scripts/`, all teacher-friendly and verified by syntax + import checks. `01_data_prep.py` runtime-verified end-to-end against the bundled data. **Alex is currently re-running the full pipeline**; `submission/outputs/{data,models,metrics,backtest,tuning}/` will populate from his run and feed the writing pass.
- **Headline figures:** the 2026-04-30 set is ready in `submission/report_assets/figures/main/` (baseline overview, tuned overview, capital curves). To be refreshed against Alex's new outputs before final integration.
- **EDA:** 19 figures + tables in `submission/report_assets/figures/appendix/eda/`, eda_appendix.docx generated.
- **Tuning:** 30-trial RF and 30-trial MLP runs completed 2026-04-30; baseline kept as headline per team agreement. HistGBM is in the `06_tuning_optuna.py` scope per §5.5 but no HistGBM tuning has produced a headline number; treat the tuning summary as **RF + MLP** only when quoting numbers in the report.
- **Report:** `KAN-CDSCO2004U_161989_160363_185912_160714_Polymarket_Mispricing.docx` is the live deliverable; final integration of `submission/report_assets/` figures and tables is the open item now that Alex's run has unblocked the writing pass.
- **LLM disclosure:** template in `paper_guidelines.md` §7; needs filling in with actual percentages and validation notes per the extended guidelines.

## 14. Open Decisions

- **Feature count: 80 vs 76.** See §6.5.
- **Headline model.** Default RF leads on baseline test AUC (0.899) but the number is suspiciously high; tuned RF drops to 0.775. The MLP baseline (0.802) is the more honest headline if the report wants to cite a single number. Decision: present both, lead with the multi-model comparison table.
- **Market-implied benchmark limitation.** The corrected pre-trade YES price and real trade liquidity are now bundled in `backtest_context.parquet`. The remaining limitation is CLOB-history asymmetry for HF markets, which should be documented rather than rebuilt this close to deadline.
- **Tuned predictions in headline backtest.** Tuned RF and MLP predictions exist (`outputs/v5/rigor/optuna/`) but are not currently isotonic-recalibrated, so threshold-based strategy results are not directly comparable to baseline. Either rerun `04_calibration.py + 05_backtest.py` against tuned predictions, or keep the side-by-side framing already in `submission/report_assets/figures/main/`.

## 15. References

- Mitts, J. & Ofir, A. (2026). *Informed Trading on Polymarket: Iran Geopolitical Markets*. Columbia Business School working paper.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 3rd ed., O'Reilly. ISBN 978-1098125974.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., Springer.
- Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with Missing Data*, 3rd ed., Wiley.
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592.
- Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: A Next-Generation Hyperparameter Optimization Framework. *KDD*.
