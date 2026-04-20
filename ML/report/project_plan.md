# Project Plan

**Course:** Machine Learning and Deep Learning, CBS, Spring 2026
**Group:** Alejandro Laurlund Gato (161989), Alexander Myrup (160363), Pontus Neumann (185912), Linus Stamov Yu (160714)
**Title:** *Mispricing on Polymarkets*
**Subtitle:** *Detecting Probability Asymmetries in Iran Geopolitical Markets with Machine Learning*
**Possible working title:** *: *
**Document purpose:** Finalise the research question, approach, and method, and map each methodological choice to the course syllabus. Supersedes earlier options discussed in the handover and in chat.

**Companion documents (read-only supporting material):**
- `handovers/handover_<date>.md` — per-session handover covering only the most recent work done. Anything that matters long-term is folded back into this plan.
- `docs/archive/mldp-project-overview.md` — Alex's narrower 7-sub-market proposal. Superseded by this plan; kept for history only.

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

*Scope:* all 74 resolved Iran sub-markets under Polymarket events 114242, 236884, 355299, and 357625; feature set as listed in Section 4; trade-timestamp temporal split (see Section 4); no random split within a market.
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
| Data sources | **Hybrid pipeline** combining HuggingFace and Polymarket APIs. (a) Event and market metadata from Polymarket Gamma API (authoritative source of resolution status). (b) Trade history for 67 of 74 resolved markets (events 114242 and 236884) from the HuggingFace mirror `SII-WANGZJ/Polymarket_data` (on-chain CTF Exchange events, MIT licensed, ~38.7 GB), streamed over HTTPS via duckdb's httpfs and filtered server-side to the target condition_ids; full trade history preserved, no offset cap. (c) Trade history for the 7 ceasefire markets under events 355299 and 357625 (created after the HF snapshot cutoff of 2026-03-31, absent from HF) from the Polymarket Data API with side-split pagination. All 7 ceasefire markets have under 5k trades each and fit comfortably under the API's ~7k ceiling, so no data loss. (d) CLOB mid-price history for ceasefire markets only, with trade-execution price as fallback. No Polygonscan or external enrichment used. |
| Target | `bet_correct` in {0, 1} from market resolution and the side of the trade. |
| Benchmark at trade time | `market_implied_prob` at execution, taken from the CLOB mid-price where available, otherwise the price field of the trade itself. |
| Features | Six layers, all strictly no-lookahead (every row's feature at time `t` uses only rows with `timestamp < t`, enforced via `groupby().cumsum().shift(1)` and `rolling(window, on='timestamp', closed='left')`): (1) **trade-local** — `log_size`, `side`, `outcomeIndex`; (2) **market context** — `market_trade_count_so_far`, `market_volume_so_far_usd`, `market_vol_1h_log`, `market_vol_24h_log`, `market_price_vol_last_1h`; (3) **time** — `time_to_settlement_s`, its log variant, `pct_time_elapsed`; (4) **wallet global** — `wallet_prior_trades`, `wallet_prior_volume_usd`, `wallet_prior_win_rate`, `wallet_first_minus_trade_sec`; (4b) **wallet-in-market** in four clusters: bet-slicing (`wallet_trades_in_market_last_1min/10min/60min`, `wallet_is_burst`), directional purity (`wallet_directional_purity_in_market`, `wallet_has_both_sides_in_market`), position-aware (`wallet_position_size_before_trade`, `trade_size_vs_position_pct`, `is_position_exit`, `is_position_flip`, `wallet_is_whale_in_market`), depth (`wallet_prior_trades_in_market`, `wallet_cumvol_same_side_last_10min`); (5) **interactions** — `size_vs_wallet_avg`, `size_x_time_to_settlement`, `size_vs_market_cumvol_pct`. **The market-implied probability (`price` / `market_implied_prob`) is deliberately excluded from the feature set** so `p_hat` is independent of the market's own belief and the gap `p_hat - market_implied_prob` is a clean signal. `market_implied_prob` is retained in the dataset only as the trading-rule benchmark. Wallet on-chain identity (Polygonscan) and news proximity (GDELT) are deferred to future work. |
| Pre-modelling filter | Drop post-resolution close-out trades (`settlement_minus_trade_sec <= 0`, ~16.5% of rows), which execute at the locked winning-token price and carry no predictive signal. |
| Split | **Trade-timestamp temporal.** Each trade is assigned to train / validation / test based on its own `timestamp`, not its market's settlement date. Rationale: events 114242 and 355299 cluster their NO and YES resolutions on different calendar dates; a settlement-date split would put the all-NO early markets in training and the YES markets in test, leaving the model with no in-sample YES outcomes and no chance to generalise to the test set. A trade-timestamp split delivers outcome-mixed rows because pre-deadline trades on eventually-YES markets existed well before those markets settled. Implementation: quantiles over the full trade timestamp range (`<q_train=0.7 → train`, `<q_val=0.85 → validation`, rest test). Each trade keeps its own market's outcome as its label — no mixing. For within-training cross-validation, `GroupKFold` on `proxyWallet` (alternative: `TimeSeriesSplit` on `timestamp`). |
| Class balance | Current `bet_correct` rate is 0.518, inside the 35 to 65 percent band, so no resampling is required up front. If imbalance develops after filtering or per split fold, apply `class_weight="balanced"` first, then SMOTE on the training fold only as a secondary option (Lecture 7). |
| Known limitation | Previously: Polymarket Data API caps pagination offset at ~7000 trades per market (after side-split), truncating 21 of 74 markets. **Mitigated** by routing events 114242 and 236884 through the HuggingFace mirror, which carries the complete on-chain trade history for those 67 markets. Residual limitation: the HF snapshot cutoff of 2026-03-31; any market still trading or created after that date must use the API path (which applies only to the 7 ceasefire markets, each well under the ~7k ceiling). |

### Multi-cluster data strategy

The project has deliberately chosen to re-stream the 38.7 GB HF trades parquet for each additional event cluster rather than download the full file once (~39 GB on disk).

- Each cluster build takes ~60 min network + ~15 min enrichment, producing a cluster-specific `00_hf_trades_cache.parquet` (tens of MB) and `03_trades_features.csv` (hundreds of MB to ~1 GB).
- After N clusters are built, their `03_trades_features.csv` files are concatenated into a single mother frame on disk. The concatenation must recompute two families of columns because they are cluster-local: (a) the running/prior features (`wallet_prior_trades`, `wallet_prior_volume_usd`, `wallet_prior_win_rate`, `market_trade_count_so_far`, `market_volume_so_far_usd`, `market_price_vol_last_1h`, and the `wallet_trades_in_market_last_*min` family), and (b) the `split` column, which is a trade-timestamp quantile inside the cluster and must be reissued globally on the merged frame.
- Rationale for repeated streams over a one-time full download: disk is tight, the cluster list is small (Iran now, potentially Maduro / Biden pardons later per §8), and the per-cluster subset parquets are the only permanent cache we need to keep.

## 5. Method, Mapped to Course Lectures

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

### 5.5 Class imbalance and data issues (Lecture 7)

- Compute per-market outcome balance. If the overall `bet_correct` rate falls outside the 35 to 65 percent band, apply sklearn `class_weight="balanced"` first.
- If still degenerate, apply SMOTE to the training fold only, never to validation or test.
- Outlier handling before training: winsorise `trade_value_usd` and `wallet_total_volume_usd` at the 1st and 99th percentiles. No whole-row removal, since extreme trades are part of the phenomenon of interest.

### 5.6 Validation strategy

Three layers, all on held-out data:

1. **Statistical.** ROC-AUC and calibration of `p_hat` and of the residualised gap against `bet_correct` on the validation and test slices. Brier-score improvement over the market-implied null.
2. **Economic.** Cumulative PnL, Sharpe ratio, hit rate, maximum drawdown, and precision@k of each trading rule on the test slice only, against the baselines in Section 5.4. Streaming event-replay protocol — at each event, the decision uses only state strictly before the event timestamp.
3. **Named-case sanity check.** Magamyman (the Columbia paper's primary documented Iran-strike insider, ~$553K entering at 17 percent implied probability 71 minutes before news) serves as a named validation anchor. Pull the wallet address from the paper appendix or the `pselamy/polymarket-insider-tracker` GitHub repo and check: does the MLP assign high `p_hat` to his documented trades, do the home-run triggers fire on them, and which feature values does the model find most salient? This is an illustrative anecdote in the Discussion, not a Results target — labelling off a single wallet would introduce selection bias if used for model selection.

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

The ethical consideration section is anchored on a concrete policy tension documented in early 2026. In November 2025, Polymarket CEO Shayne Coplan described insider edge on *60 Minutes* as "a good thing" and "an inevitability." Four months later, on 23 March 2026, Polymarket announced explicit rules prohibiting (1) trading on stolen confidential information, (2) trading on illegal tips, and (3) trading by anyone in a position of authority over the event outcome. That pivot frames the section.

Concretely, the section will cover:
- **Privacy.** Pseudonymous on-chain wallets are persistent; pattern-linking is de-anonymising. Features are aggregated and no individual wallet list is published. The Magamyman case in the Discussion uses information already public in the Columbia paper.
- **Dual use.** A trained model could help regulators detect manipulation or help platforms surveil users. The Coplan quote versus the March-2026 rule change is the real policy tension.
- **Label validity.** "Informed trading" versus "skill" versus "luck" is genuinely ambiguous. `bet_correct` is a probabilistic signal, not a legal determination.
- **Enforcement gap.** Despite public documentation of Magamyman, Burdensome-Mix, and the Biden-pardons wallets, no publicly disclosed wallet has been banned and no profits clawed back as of the cut-off date. Detection without consequence is the current equilibrium; the work's policy relevance depends on whether that equilibrium shifts.
- **Dataset bias.** Selected markets are high-volume English-language geopolitical contracts on a specific event cluster; findings do not necessarily generalise.
- **Platform surveillance context.** Polymarket's own ML detection stack ("Vergence", Palantir + TWG AI) launched 10 March 2026 but is scoped to sports markets. Geopolitical markets — the domain of this project — have no publicly disclosed ML surveillance, which is one reason the work is worth doing.
- **LLM usage disclosure.** Required by the exam brief. The Contribution and LLM Usage Disclosure section of the report documents which parts of the pipeline were co-authored with Claude.

## 8. Out of Scope — Reserved for Discussion and Future Work

The following observation is explicitly *outside* the research question and the evaluation. It is reserved for the Discussion and theory sections of the report and may be extended in future work.

- **Feature-importance parallel to informed-trading traits.** If the features driving the `p_hat − market_implied_prob` gap resemble those documented in the informed-trading literature (within-trader bet size, cross-sectional bet size, pre-event timing, directional concentration, wallet newness; see Mitts and Ofir 2026), the Discussion will flag the parallel and note that the signal may be partially picking up informed flow. The Magamyman sanity check in Section 5.6 feeds this discussion.
- **Documented-case anecdotes** (Burdensome-Mix on Maduro, Biden-pardons wallets) are illustrative rather than evaluation targets. The Iran ceasefire cluster, by contrast, is part of the main evaluation scope via events 355299 and 357625.
- **Cross-market sibling-price injection.** A second-order arbitrage feature layer using prices of related sub-markets (e.g. "by Feb 14" price as a feature for the "by Feb 28" market) is deferred.
- **Cross-event-family pooling** with non-Iran clusters (Maduro, Biden pardons, Taylor Swift) is deferred.
- **Orderfilled event-level analysis.** The HF mirror also exposes `orderfilled_part1-4.parquet` (raw on-chain OrderFilled logs, ~125 GB combined). Not used here — the derived `trades.parquet` carries the aggregated per-trade records we need. Reserved for future work that needs maker/taker attribution at order level.

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
| 2 | Full trade extraction across four target events (API-only, v1) | done (19 Apr) | 1 |
| 3 | True resolution-timestamp derivation (`resolution_ts`) and `settlement_minus_trade_sec` | done (19 Apr) | 2 |
| 4 | Running market and wallet features (market-state, wallet-global) | done (19 Apr) | 2 |
| 5 | Expanded six-layer feature set — time, log_size, wallet-in-market bursting, directional purity, position-aware, interactions | done (19 Apr, Alex-adoption pass) | 2 |
| 6 | Trade-timestamp temporal split column in `03_trades_features.csv` | done (19 Apr, Alex-adoption pass) | 5 |
| 7 | Hybrid HF + API build script (`scripts/02_build_dataset.py`), rewritten 20 Apr to use pyarrow + fsspec chunked row-group reads with retry + reopen (duckdb httpfs hit Snappy decompression errors mid-stream on the 38.7 GB file) | done (20 Apr) | 5 |
| 8 | Run the hybrid build end-to-end for the Iran cluster (events 114242, 236884, 355299, 357625). Output: `data/03_trades_features.csv`, 1,209,787 rows × 57 cols, 806 MB; 67 HF markets + 7 API markets; no market truncated. HF stream took ~64 min; enrichment ~13 min | done (20 Apr) | 7 |
| 9 | Post-resolution filter applied (`settlement_minus_trade_sec > 0`) in modelling | open | 3 |
| 10 | Polygonscan on-chain wallet enrichment | deferred (no API key, free-data-only scope) | 2 |
| 11 | GDELT news-timing enrichment | deferred (free API, awaits feature-definition design) | 2 |
| 12 | EDA script following repository Design.md conventions; figures and tables inserted into report Appendix with cross-references from the EDA body | done (20 Apr) | 8 |
| 13 | MLP training and baselines (logistic regression, random forest, isolation forest, naive market, autoencoder) | open | 6 |
| 14 | Isotonic calibration on validation slice | open | 13 |
| 15 | Streaming event-replay backtest — general +EV and home-run, cutoff-date sweep | open | 14 |
| 16 | Out-of-sample trading-rule evaluation on the test slice | open | 15 |
| 17 | Feature-importance and permutation importance for Discussion | open | 13 |
| 18 | Magamyman sanity check | open | 13 |
| 19 | Report drafting in the CBS docx template | all | 12 to 18 |

## 11. Open Decisions

- Exact edge thresholds for the general +EV rule (currently 0.02) and the home-run rule (currently 0.20) — may shift after validation tuning.
- Position-sizing rule for the home-run strategy: flat larger stake versus Kelly-scaled.
- Whether to model trade value in USD as a feature (already included via `trade_value_usd` and `log_size`) or additionally as a sample weight during training.
- Exact quantile boundaries for the trade-timestamp train / validation / test split (default 0.70 / 0.85 / 1.00).
- Treatment of `wallet_prior_win_rate` NaN on first-trade rows (~21%): impute to global mean (0.518), use the raw NaN with `wallet_prior_trades == 0` as a categorical indicator, or exclude first-trade rows.
- Treatment of NaN on the expanded wallet-in-market features. Expected null rates on first occurrence per (wallet, market): `wallet_directional_purity_in_market` and `wallet_spread_ratio` around 48 percent; `size_vs_wallet_avg` around 21 percent; `pct_time_elapsed` around 2 percent (markets missing both `resolution_ts` and `end_date`).
- Burst-detection thresholds: number of trades `K` in a rolling window `N` minutes that triggers `wallet_is_burst`. Default `K=3, N=10min`.
- Whale threshold for `wallet_is_whale_in_market`: default 95th percentile of per-market cumulative wallet volume.

## 12. Repository Layout

Everything lives under `ML/report/`. The folder layout after the 19 April restructure:

```
report/
├── ML_final_exam_paper.docx          # The paper itself — updated with text and results as the project progresses
├── project_plan.md                   # This file. The source of truth. Read first.
│
├── docs/                             # Superseded material kept for history
│   └── archive/
│       └── mldp-project-overview.md  # Alex's narrower 7-sub-market proposal (superseded)
│
├── handovers/                        # Per-session handovers; latest only at top level
│   ├── handover_<latest>.md          # Most recent session, covers what was done since previous handover
│   ├── ML_final_exam_paper_backup.docx
│   └── archive/
│       └── handover_<older>.md
│
├── scripts/                          # Data-pipeline and modelling code
│   ├── 01_polymarket_api.py          # Gamma / CLOB / Data API client; feature enrichment library.
│   ├── 02_build_dataset.py           # Primary entry point. Hybrid HF + API builder.
│   ├── 02_build_dataset_legacy.py    # Legacy API-only builder. Superseded by 02_build_dataset.py but kept for reference.
│   ├── 03_enrich_wallets.py          # Polygonscan / Etherscan batch enrichment. Deferred — not wired in for v1.
│   ├── 03b_enrichment_dashboard.py   # Live HTML dashboard for long-running enrichment runs.
│   ├── 04_eda.py                     # EDA plots and summary tables; reads 03_trades_features.csv, writes to outputs/eda/.
│   ├── 05_docx_fix_text.py           # One-shot: align docx body text with project state.
│   └── 06_docx_insert_eda.py         # One-shot: insert EDA narrative and appendix into the docx.
│
├── data/                             # Local data outputs; .gitignored except for snapshots
│   ├── 00_hf_markets_master.parquet  # HF markets metadata cache (116 MB) — routing table
│   ├── 00_hf_trades_cache.parquet    # HF-path Iran trades cache (~50 MB)
│   ├── 01_markets_meta.csv           # Iran markets metadata (74 resolved)
│   ├── 01_prices.csv                 # Iran intraday price history
│   ├── 02_trades.csv                 # Concatenated Iran trades (HF + API)
│   ├── 03_trades_features.csv        # Mother dataframe (features + labels + split)
│   └── _backup_<date>/               # Pre-refetch snapshots of the CSVs
│
├── outputs/                          # Generated artefacts from scripts
│   └── eda/                          # Figures, skewness table, HTML report, summary.txt
│
├── assets/                           # Static image assets used in the docx cover and figures
│
└── guidelines/                       # CBS course materials and reference exemplars
    ├── Project_guidelines.pdf
    ├── Project_guidelines_new_extended_instructions.pdf
    ├── Face_Mask_Detection_sample_report.pdf
    └── Guidelines-for-the-use-of-Generative-Artificial-Intelligence-GenAI-in-exams-at-CBS.pdf
```

**Reproduction workflow.** From `report/`:
1. `python scripts/02_build_dataset.py` — streams the HF subset, pulls the ceasefire markets via the API, writes `data/03_trades_features.csv` with train/val/test labels.
2. `python scripts/04_eda.py` — writes plots and `report.html` into `outputs/eda/`.
3. Modelling code — to be added under `scripts/` (MLP, baselines, calibration, backtest). Each script reads from `data/`, writes to `outputs/<stage>/`.
