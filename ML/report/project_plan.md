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

The project studies asymmetries between market-implied probabilities and model-predicted probabilities in resolved Iran geopolitical Polymarket events. The hypothesis is that a machine learning model, trained on behavioural and market-state features available at trade time, can generate a probability estimate that systematically diverges from the contemporaneous market-implied probability, and that this gap constitutes a tradable signal in settled markets. The primary output is a trading algorithm that enters positions whenever the predicted probability and the market-implied probability differ beyond a threshold, evaluated by out-of-sample economic performance on temporally held-out markets. The pulled dataset covers all 74 resolved sub-markets across four Iran-related events (114242, 236884, 355299, 357625), comprising 1,209,787 trades across 109,080 wallets between 2025-12-22 and 2026-04-19.

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
| Unit of analysis | One resolved trade. Realised dataset: 1,209,787 trades, 109,080 unique wallets, spanning 2025-12-22 to 2026-04-19. |
| Data sources | **Hybrid pipeline** combining HuggingFace and Polymarket APIs. (a) Event and market metadata from Polymarket Gamma API (authoritative source of resolution status). (b) Trade history for 67 of 74 resolved markets (events 114242 and 236884) from the HuggingFace mirror `SII-WANGZJ/Polymarket_data` (on-chain CTF Exchange events, MIT licensed, ~38.7 GB), streamed over HTTPS via duckdb's httpfs and filtered server-side to the target condition_ids; full trade history preserved, no offset cap. (c) Trade history for the 7 ceasefire markets under events 355299 and 357625 (created after the HF snapshot cutoff of 2026-03-31, absent from HF) from the Polymarket Data API with side-split pagination. All 7 ceasefire markets have under 5k trades each and fit comfortably under the API's ~7k ceiling, so no data loss. (d) CLOB mid-price history for ceasefire markets only, with trade-execution price as fallback. No Polygonscan or external enrichment used. |
| Target | `bet_correct` in {0, 1} from market resolution and the side of the trade. |
| Benchmark at trade time | `market_implied_prob` at execution, taken from the CLOB mid-price where available, otherwise the price field of the trade itself. |
| Features | Seven layers, all strictly no-lookahead (every row's feature at time `t` uses only rows with `timestamp < t`, enforced via `groupby().cumsum().shift(1)` and `rolling(window, on='timestamp', closed='left')`): (1) **trade-local** — `log_size`, `side`, `outcomeIndex`; (2) **market context** — `market_trade_count_so_far`, `market_volume_so_far_usd`, `market_vol_1h_log`, `market_vol_24h_log`, `market_price_vol_last_1h`, `market_buy_share_running`; (3) **time** — `time_to_settlement_s`, its log variant, `pct_time_elapsed`; (4) **wallet global** — `wallet_prior_trades`, `wallet_prior_volume_usd`, `wallet_prior_win_rate`, `wallet_first_minus_trade_sec`; (4b) **wallet-in-market** in four clusters: bet-slicing (`wallet_trades_in_market_last_1min/10min/60min`, `wallet_is_burst`, `wallet_median_gap_in_market`), directional purity (`wallet_directional_purity_in_market`, `wallet_has_both_sides_in_market`), position-aware (`wallet_position_size_before_trade`, `trade_size_vs_position_pct`, `is_position_exit`, `is_position_flip`, `wallet_is_whale_in_market`), depth (`wallet_prior_trades_in_market`, `wallet_cumvol_same_side_last_10min`); (5) **interactions** — `size_vs_wallet_avg`, `size_x_time_to_settlement`, `size_vs_market_cumvol_pct`, `size_vs_market_avg`; (6) **on-chain identity (Layer 6, Bucket 2 #5, 21 Apr)** — `wallet_polygon_age_at_t_days`, `wallet_polygon_nonce_at_t` (+ log), `wallet_n_inbound_at_t` (+ log), `wallet_n_cex_deposits_at_t`, `wallet_cex_usdc_cumulative_at_t` (+ log), `days_from_first_usdc_to_t`, `wallet_funded_by_cex`, `wallet_funded_by_cex_scoped`, `wallet_enriched` — computed via `scripts/11_add_layer6.py` bisect on per-wallet Etherscan V2 timestamp arrays; (7) **cross-market diversity (Bucket 2 #4, 21 Apr)** — `wallet_market_category_entropy`, Shannon entropy (nats) over coarse category distribution of a wallet's prior distinct markets across the full Polymarket universe, derived from the HuggingFace mirror via `scripts/10_wallet_category_entropy.py`. **The market-implied probability (`price` / `market_implied_prob`) is deliberately excluded from the feature set** so `p_hat` is independent of the market's own belief and the gap `p_hat - market_implied_prob` is a clean signal. `market_implied_prob` is retained in the dataset only as the trading-rule benchmark. News proximity (GDELT) remains deferred to future work. |
| Pre-modelling filter | Drop post-resolution close-out trades (`settlement_minus_trade_sec <= 0`, ~16.5% of rows), which execute at the locked winning-token price and carry no predictive signal. |
| Split | **Market-cohort, narrow-window training (22 Apr revision).** Train / validation / test are defined at the market level, not by trade-timestamp quantile. **Train:** 4 strike markets — Feb 25, 26, 27, and Feb 28 — for ~215k trades in a cohesive 4-day peak-informed-flow window (Feb 28 is the documented asymmetric-information market per Mitts & Ofir 2026, USD 143M anomaly, and must be in training, not held out). Train mix is 3 NO markets (Feb 25–27) + 1 YES market (Feb 28); Feb 28 alone is ~61% of training trades, by design. **Val:** 1 older-deadline strike NO market well outside the peak window (candidate: `US strikes Iran by January 23, 2026?`, ~39k trades NO), used for MLP early stopping and isotonic calibration. Choosing a January NO strike as val gives a larger val sample than any single ceasefire market and keeps val in the same event family as training, so early-stopping signal is less noisy — while the temporal distance (~5 weeks before the peak window) makes val a genuine "did we overfit to Feb 25–28 microstructure" check. **Test:** to be finalised. Candidates are a mix of non-strike NO markets (7 ceasefire markets in events 355299 + 357625, 3 conflict-end markets in event 236884 — all resolved NO) and held-out strike YES markets from March (e.g. Mar 15 = ~25k trades YES, Mar 31 = ~76k trades YES) to give YES/NO balance in the test cohort. Target final test set: one NO + one YES market, each ~5–25k trades, picked to balance sample size, event-family diversity, and generalisation difficulty. Rationale for the overall design: (a) training on the 4-day peak-signal window matches the documented informed-flow concentration and allows fast iteration; (b) val is within-family but temporally distant to detect peak-window overfitting; (c) test crosses event families where possible (strikes → ceasefires / conflict-end) to serve as the main RQ1 generalisation test; (d) where test includes a held-out strike YES market, this disambiguates "learned NO-bias" from "learned trade-correctness". Memorisation risk from market-identifying absolute-scale features (e.g. `time_to_settlement_s`, `market_cumvol_log`) is already mitigated by the v3 feature-pruning fix from PR #5. **Expansion path if initial results are weak:** add the week before (Feb 18–24) to training, then the month before, until signal stabilises — itself a finding. See `scripts/14_build_experiment_splits.py`. The earlier trade-timestamp quantile split is superseded by this design. |
| Class balance | Current `bet_correct` rate is 0.518, inside the 35 to 65 percent band, so no resampling is required up front. If imbalance develops after filtering or per split fold, apply `class_weight="balanced"` first, then SMOTE on the training fold only as a secondary option (Lecture 7). |
| Known limitation | Previously: Polymarket Data API caps pagination offset at ~7000 trades per market (after side-split), truncating 21 of 74 markets. **Mitigated** by routing events 114242 and 236884 through the HuggingFace mirror, which carries the complete on-chain trade history for those 67 markets. Residual limitation: the HF snapshot cutoff of 2026-03-31; any market still trading or created after that date must use the API path (which applies only to the 7 ceasefire markets, each well under the ~7k ceiling). |

### Multi-cluster data strategy

The project has deliberately chosen to re-stream the 38.7 GB HF trades parquet for each additional event cluster rather than download the full file once (~39 GB on disk).

- Each cluster build takes ~60 min network + ~15 min enrichment, producing a cluster-specific `00_hf_trades_cache.parquet` (tens of MB) and `03_consolidated_dataset.csv` (hundreds of MB to ~1 GB).
- After N clusters are built, their `03_consolidated_dataset.csv` files are concatenated into a single consolidated dataset on disk. The concatenation must recompute two families of columns because they are cluster-local: (a) the running/prior features (`wallet_prior_trades`, `wallet_prior_volume_usd`, `wallet_prior_win_rate`, `market_trade_count_so_far`, `market_volume_so_far_usd`, `market_price_vol_last_1h`, and the `wallet_trades_in_market_last_*min` family), and (b) the `split` column, which is a trade-timestamp quantile inside the cluster and must be reissued globally on the merged dataset.
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

### 5.6 Missing-data typology and handling

Missingness in the feature frame partitions into two classes.

**Structural missingness** is NaN on expanding / running quantities that are mathematically undefined given the prior history available at row time: expanding win-rate on a wallet's first trade, cross-market category entropy before two distinct markets have been observed, directional purity on the first (wallet, market) trade, `size_vs_wallet_avg` on the first trade. The NaN is the truthful value. Filling with a sentinel (zero in particular) would conflate "not yet defined" with a legitimate realised value — for category entropy specifically, zero is a valid realised value meaning a wallet concentrated in exactly one category, so imputing 0 for undefined entropy would erase the distinction.

**Pipeline missingness** is NaN where the underlying quantity exists but could not be observed: all twelve Layer 6 on-chain columns (nine semantic features plus three log variants) on rows whose wallet failed Etherscan V2 enrichment, and `pct_time_elapsed` on markets missing both `resolution_ts` and `end_date` metadata. For these, zero would be semantically wrong for most features (a real wallet is never zero days old on Polygon; a real market does resolve); NaN plus the enrichment/pipeline flag is the faithful representation.

The dataset preserves NaN in every affected raw feature and carries one binary indicator column per missingness species: `wallet_has_prior_trades`, `wallet_has_prior_trades_in_market`, `wallet_has_cross_market_history`, `market_timing_known` (added by `scripts/11b_add_missingness_flags.py`), and `wallet_enriched` (added by `scripts/11_add_layer6.py`). Imputation is applied only at the modelling stage and differs by classifier: tree-based models accept NaN natively in sklearn ≥ 1.4 and pass raw columns through; logistic regression and the MLP impute using train-split-only medians followed by standardisation. In all cases the indicator columns are retained as features, so whatever value is imputed for the raw feature the model still sees the "was this missing" signal. Per-column assignments, share statistics, and policy details are recorded in `data/MISSING_DATA.md` and updated there as decisions evolve.

Row dropping is not used. Sensitivity analyses restricted to rows where a given indicator equals one may appear in the Discussion if they sharpen a result, but the main evaluation uses the full 1.21-million-row frame. The framework follows Rubin (1976) and Little & Rubin (2019), extended with a structural missingness sub-type appropriate to running / expanding features in event-time panel data.

### 5.7 Validation strategy

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

- **Feature-importance parallel to informed-trading traits.** If the features driving the `p_hat − market_implied_prob` gap resemble those documented in the informed-trading literature (within-trader bet size, cross-sectional bet size, pre-event timing, directional concentration, wallet newness; see Mitts and Ofir 2026), the Discussion will flag the parallel and note that the signal may be partially picking up informed flow. The Magamyman sanity check in Section 5.7 feeds this discussion.
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
| 6 | Trade-timestamp temporal split column in `03_consolidated_dataset.csv` | done (19 Apr, Alex-adoption pass) | 5 |
| 7 | Hybrid HF + API build script (`scripts/02_build_dataset.py`), rewritten 20 Apr to use pyarrow + fsspec chunked row-group reads with retry + reopen (duckdb httpfs hit Snappy decompression errors mid-stream on the 38.7 GB file) | done (20 Apr) | 5 |
| 8 | Run the hybrid build end-to-end for the Iran cluster (events 114242, 236884, 355299, 357625). Output: `data/03_consolidated_dataset.csv`, 1,209,787 rows × 57 cols, 806 MB; 67 HF markets + 7 API markets; no market truncated. HF stream took ~64 min; enrichment ~13 min | done (20 Apr) | 7 |
| 9 | Post-resolution filter applied (`settlement_minus_trade_sec > 0`) in modelling | open | 3 |
| 10 | Polygonscan on-chain wallet enrichment (Layer 6 collection, `scripts/03_enrich_wallets.py`) — Etherscan V2 multichain free tier, 3 keys × 6 workers (18 total). Initial pass 21 Apr evening reached 109,080 wallets with 6.88 % NOTOK failure rate (Etherscan server-side rejection on specific addresses, non-retryable inside the 5-retry budget). **Retry pass 22 Apr early AM re-attempted the 7,509 failed wallets; 7,050 recovered (93.9 % recovery rate).** Final: 108,621 ok / 459 permanent failures (0.42 %). | done (22 Apr early AM) | 2 |
| 10b | Layer 6 integration (`scripts/11_add_layer6.py`) — bisect per-wallet timestamp arrays onto each trade. **Emits 12 columns** (nine semantic features: `wallet_polygon_age_at_t_days`, `wallet_polygon_nonce_at_t`, `wallet_n_inbound_at_t`, `wallet_n_cex_deposits_at_t`, `wallet_cex_usdc_cumulative_at_t`, `days_from_first_usdc_to_t`, `wallet_funded_by_cex`, `wallet_funded_by_cex_scoped`, `wallet_enriched`; plus three log variants: `wallet_log_polygon_nonce_at_t`, `wallet_log_n_inbound_at_t`, `wallet_log_cex_usdc_cum`). In-place CSV patch with `.pre11.csv` backup. Pandas-3 datetime-unit gotcha (storage is `[us]` not `[ns]`; cast to `datetime64[ns, UTC]` before `// 10**9`). | done (22 Apr early AM) | 10 |
| 10c | Cross-market category-entropy feature (`scripts/10_wallet_category_entropy.py`) — stream HF mirror for 109K wallets' full Polymarket history, bucket event slugs into 8 categories, expanding entropy per wallet, +1 col | done (21 Apr evening) | 2 |
| 10d | Missing-data policy + indicator columns (`scripts/11b_add_missingness_flags.py`, `data/MISSING_DATA.md`) — 4 binary indicators + typology doc; §5.6 policy landed | done (21 Apr evening) | 10c |
| 10e | Preserve per-market `outcomes` array through `enrich_trades` and derive trade-level `is_yes` column (`scripts/01_polymarket_api.py` patch). Backfill for current CSV via `scripts/15_backfill_is_yes.py`. Fixes a missing YES/NO label (outcomes-array had been dropped during the meta→trade join, so downstream cohort picking had no reliable way to map `winning_outcome_index` to a YES/NO label). Backfill result on current CSV: 19 YES / 55 NO markets, 0 mismatches vs meta source; all 7 ceasefire markets resolved NO. | done (22 Apr, PR #6) | 8 |
| 10f | Experiment-cohort builder (`scripts/14_build_experiment_splits.py`) — slices small, self-contained parquets for train / val / test per the §4 split definition, loads in <1s instead of re-reading the 1 GB CSV each run | open | 10e |
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
- *Resolved 22 Apr.* Train / val / test split strategy. Replaced the earlier trade-timestamp quantile split (0.70 / 0.85 / 1.00) with a market-cohort, narrow-window design: train on 4 strike markets (Feb 25–28, ~215k trades including the Feb 28 informed-flow market); val on 1 older NO strike market (candidate Jan 23, ~39k trades, for temporally-distant within-family validation); test on a mix of non-strike markets (ceasefire + conflict-end, all NO) and held-out strike YES markets (March). Specific test-market picks TBD. Full rationale in §4 → Split row.
- *Resolved 21 Apr evening.* Treatment of NaN on running / expanding features (`wallet_prior_win_rate`, `wallet_directional_purity_in_market`, `wallet_spread_ratio`, `wallet_median_gap_in_market`, `size_vs_wallet_avg`, `pct_time_elapsed`, `wallet_market_category_entropy`, and the 12 Layer 6 on-chain columns): preserve NaN in the feature frame, attach five binary missingness indicators (`wallet_has_prior_trades`, `wallet_has_prior_trades_in_market`, `wallet_has_cross_market_history`, `market_timing_known`, `wallet_enriched`), defer imputation to the modelling stage per the classifier family. Full policy in §5.6 and `data/MISSING_DATA.md`.
- Burst-detection thresholds: number of trades `K` in a rolling window `N` minutes that triggers `wallet_is_burst`. Default `K=3, N=10min`.
- Whale threshold for `wallet_is_whale_in_market`: default 95th percentile of per-market cumulative wallet volume.
- *Investigated and closed as non-issue, 22 Apr.* Data-integrity audit of `bet_correct` across all 74 markets. A consistency check during the `is_yes` backfill initially flagged the 7 API-path ceasefire markets as potentially corrupt — all 7 had `winning_outcome_index=1` while the per-trade `outcome` column (populated only for API-path trades) disagreed with `wi` for 3 of them. Follow-up investigation showed **no corruption**: the per-trade `outcome` column is not market resolution, it is `outcomeIndex == 0` (i.e. "is this trade on the Yes-side token"), so its disagreement with `wi` was expected and harmless. Independent verification: (a) price-tail convergence to 1.0 / 0.0 per outcomeIndex confirms `winning_outcome_index` is correct for all 7 ceasefire markets (outcomeIndex=1 tail price ≈ 0.995 vs outcomeIndex=0 ≈ 0.005); (b) `bet_correct` internal-consistency invariant — within each market, all BUY trades on one outcomeIndex have mean `bet_correct` ≈ 1 and the other ≈ 0 — holds for 74/74 markets, so the training target is reliable throughout. The only persistent artefact of the original confusion was the "55 YES / 19 NO" framing in the 21 Apr handover, now corrected to 19 YES / 55 NO (see §10 row 10e). Worth keeping the invariant check in a unit test to catch genuine label-corruption if it appears in a future rebuild.
- *Resolved 22 Apr early AM.* Initial Layer 6 pass produced 6.88 % NOTOK wallet failures (vs ~1 % originally estimated). A targeted retry pass on the 7,509 failed wallets recovered 7,050 of them (93.9 % recovery) — consistent with the "intermittent server-side latency" diagnosis rather than permanent address rejection. Final parquet state: 108,621 / 109,080 wallets enriched (99.58 %), 459 permanent failures (0.42 %). Trade-level Layer 6 coverage: 1,206,050 / 1,209,787 rows (99.69 %); residual 3,737 trades carry `wallet_enriched=0` and NaN Layer 6 per §5.6. To be noted in Methodology → Known Limitations as a ~0.4 % non-coverage fraction.

## 12. Repository Layout

Everything lives under `ML/report/`. Current folder layout:

```
report/
├── ML_final_exam_paper.docx                 # The paper. Updated as the project progresses.
├── project_plan.md                          # This file. Source of truth. Read first.
├── Design.md                                # Naming and code-style conventions for scripts.
├── .env.example                             # Template for Etherscan V2 keys (copy to .env).
│
├── alex_updates_before_incorporation/       # Reference-only: Alex's narrower-scope material.
│   ├── design-decisions.md                  # Framing, features, baselines — we adopted the pre-EDA parts only.
│   ├── mldp-project-overview.md             # His 7-sub-market proposal (superseded for our scope).
│   └── SESSION-HANDOVER.md                  # Alex's end-state handover after the v2/v3 runs.
│
├── archive/                                 # Older checkpoints preserved just in case.
│   └── ML_final_exam_paper.pre08.docx       # Pre-Alex-incorporation snapshot of the paper.
│
├── assets/                                  # Static image assets used inside the docx.
│   ├── cbs_paper_image_{1,2,3}.png          # CBS cover/branding images.
│   └── trading_image.jpg                    # Motivation-section illustration.
│
├── data/                                    # Local outputs; .gitignored except committed evidence.
│   ├── 00_hf_markets_master.parquet         # HF markets metadata cache (116 MB) — join table for event_slug / category.
│   ├── 00_hf_trades_cache.parquet           # HF-path Iran trades subset (~50 MB).
│   ├── 01_markets_meta.csv                  # 74 resolved Iran markets' metadata.
│   ├── 01_prices.csv                        # Iran intraday price history.
│   ├── 02_trades.csv                        # Concatenated Iran trades (HF + API).
│   ├── 03_consolidated_dataset.csv               # Consolidated dataset (features + labels + split + missingness flags).
│   ├── 03_consolidated_dataset.pre11.csv         # Single safety backup, kept one step behind current.
│   ├── MISSING_DATA.md                      # Authoritative policy for NaN handling and indicator columns; cited by §5.6.
│   ├── wallet_enrichment.parquet            # Layer 6 Etherscan enrichment output; written by 03_enrich_wallets.py.
│   ├── enrichment_progress.json             # Live status snapshot of 03_enrich_wallets.py.
│   ├── enrichment_stdout.log                # Streaming log for the enrichment run.
│   ├── iran_strike_labeled_v2.parquet       # Alex's 7-market dataset — git-tracked as evidence, not our pipeline.
│   ├── backtest_v{2,3}_outputs/             # Alex's evidence artefacts — git-tracked, not our results.
│   └── mlp_reframed_v{2,3}_outputs/         # Same — Alex's exploration runs.
│
├── guidelines/                              # CBS course materials and reference exemplars.
│   ├── 01_Project_Guidelines_original.pdf
│   ├── 02_Project_Guidelines_extended.pdf   # Newer instructions — the one we follow.
│   ├── 03_CBS_GenAI_Guidelines.pdf
│   └── 04_Sample_Report_Face_Mask_Detection.pdf
│
├── handovers/                               # Per-session handovers; newest at top level.
│   └── handover_21_apr.md                   # Evening 21 Apr: Bucket 2 entropy done, Layer 6 enrichment in flight.
│
├── outputs/                                 # Generated artefacts from scripts.
│   └── eda/                                 # Figures (01–09), skewness table, top-correlations list, summary.txt.
│
└── scripts/                                 # Data pipeline + modelling code.
    ├── 01_polymarket_api.py                 # Gamma / CLOB / Data API client + feature-enrichment library.
    ├── 02_build_dataset.py                  # Primary entry point. Hybrid HF + API builder; writes 03_consolidated_dataset.csv.
    ├── 02_build_dataset_legacy.py           # API-only builder, superseded; kept for reference.
    ├── 03_enrich_wallets.py                 # Etherscan V2 on-chain enrichment (Layer 6 collection).
    ├── 03b_enrichment_dashboard.py          # Live HTML progress dashboard for long enrichment runs.
    ├── 04_eda.py                            # Writes EDA figures + tables to outputs/eda/.
    ├── 05_docx_fix_text.py                  # One-shot: align docx body text with project state.
    ├── 06_docx_insert_eda.py                # One-shot: insert EDA narrative + appendix into the docx.
    ├── 07_docx_restructure.py               # One-shot: page-break / References / headroom / inline-table fixes.
    ├── 08_docx_incorporate_alex.py          # One-shot: 14 pre-EDA text edits adopting Alex's sharper framing.
    ├── 09_patch_new_features.py             # One-shot Bucket 1 patcher: 3 Layer-5 features in-place.
    ├── 10_wallet_category_entropy.py        # Bucket 2 #4: HF-stream + duckdb bucketed entropy → +1 col.
    ├── 11_add_layer6.py                     # Bucket 2 #5: bisect-based Layer 6 integrator → +12 cols (NaN-aware).
    ├── 11b_add_missingness_flags.py         # Adds 4 binary missingness indicator columns per §5.6 policy.
    ├── 12_train_mlp.py                      # MLP + LogReg + RF baselines + isotonic calibration (scaffolded).
    ├── 14_build_experiment_splits.py        # Slices train/val/test cohort parquets per §4 split definition.
    └── 15_backfill_is_yes.py                # One-shot: adds outcomes + is_yes columns to existing CSV (obsolete after next rebuild).
```

**Reproduction workflow.** From `report/`:

1. **Pipeline build (first run or full rebuild):**
   - `python scripts/02_build_dataset.py` — streams HF + pulls ceasefire markets via API, writes `data/03_consolidated_dataset.csv` at the base feature set with train/val/test labels.
   - `python scripts/09_patch_new_features.py` — in-place Bucket 1 Layer 5 patch (+3 cols).
2. **Enrichment + feature expansion (needs `.env` with Etherscan V2 keys):**
   - `python scripts/10_wallet_category_entropy.py` — HF-stream cross-market history → duckdb bucketed entropy → +1 col. Resumable.
   - `python scripts/11b_add_missingness_flags.py` — derives 4 binary missingness indicators from the running/expanding feature columns → +4 cols. Fast (~1 min).
   - `python scripts/03_enrich_wallets.py` — Etherscan V2 tokentx per wallet → `data/wallet_enrichment.parquet`. Resumable via 500-wallet checkpoints.
   - `python scripts/11_add_layer6.py` — bisect per-trade Layer 6 features → +12 cols; NaN on un-enriched rows per §5.6 policy.
3. **EDA:**
   - `python scripts/04_eda.py` — writes plots and summaries into `outputs/eda/`. Figures then flow from there into the docx.
4. **Experiment cohorts (market-cohort split per §4):**
   - `python scripts/14_build_experiment_splits.py` — derives `is_yes` (or trusts column if already present from the patched 01 pipeline) and slices `data/experiments/{train,val,test_no,test_yes}.parquet` per the §4 market-cohort definition. Output parquets are tiny (~few MB each) and load in <1s.
5. **Modelling (scaffolded, pending data completion):**
   - `python scripts/12_train_mlp.py` — LogReg + RF + MLP + isotonic calibration on the cohort parquets from step 4. Outputs per-model `metrics.json`, `feature_list.json`, MLP `loss_curve.png` into `outputs/modelling/<model>/`.
6. **Report integration:**
   - `ML_final_exam_paper.docx` consumes figures from `outputs/eda/` (already wired) and will consume modelling outputs next.

The one-shot `05–09_docx_*` scripts were run once during the pre-EDA restructure pass; they are not part of the routine rebuild loop but remain in-tree as evidence of the edits applied.
