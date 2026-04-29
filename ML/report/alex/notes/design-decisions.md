# Design decisions — idea 1 (Iran strike → Iran ceasefire)

*Append-only log of decisions made during cohort design + modelling. Per-decision template:
**Decision** / **Alternatives** / **Justification** / **Implications** / **Status** (Locked / Leaning / Open / Deferred).*

Last updated: 2026-04-26 (+ retroactive carryovers D-023 to D-026 from 2026-04-22 cleanup session)

---

## Project framing

### D-001: Research question
**Status:** Locked (revised 2026-04-28, see D-034)
**Decision:** "Can a supervised model predict trade outcomes on Polymarket geopolitical event markets, and do the predictions transfer across event regimes?" Train on Iran-strike countdown markets; test on Iran-ceasefire countdown markets. Decompose what the model is actually detecting.
**Alternatives considered:**
- "Asymmetric-information / informed-flow detection across regimes" — *original claim, retired 2026-04-28*. Falsified by B1b (naive consensus rule matches top-1%; model is consensus detector, not informed-flow detector).
- "Detect mispricing within Iran-strike regime only" (Claim A) — narrower, less interesting
- Pure late-flow replication of one Mitts & Ofir finding — too narrow for ML coursework, doesn't justify model engineering
**Justification:** Drops the unsupported "informed-flow" framing while keeping the cross-regime transferability and the trade-outcome prediction targets. The decomposition (what is the model detecting?) becomes the central scientific contribution — answer: high-precision consensus detection on heavy-favourite NO bets, with a small ~5pp AUC delta over a 3-line naive consensus rule. This RQ survives all pressure tests honestly and remains interesting.
**Implications:**
- Target is `bet_correct`; model predicts winners directly
- Train and test must be different markets, market-level CV
- Report must include the decomposition (B1a, B1b, late-flow replication) as a primary section, not a footnote
- "Asymmetric information" can appear in literature review but not as a claim we make

### D-002: Cohort structure — train + test, no val
**Status:** Locked
**Decision:** Train = "US strikes Iran by [date]" canonical ladder. Test = "US x Iran ceasefire by [date]" canonical ladder. No separate val cohort.
**Alternatives considered:**
- Train + val + test (3-way split with chronologically-fitting val market)
- Single-market train and test (e.g., Feb 28 strike → Apr 7 ceasefire)
- Within-market chronological split (rejected — see D-005)
**Justification:** Strike and ceasefire are days apart, no natural calendar gap to fit a third cohort. CV-on-train is statistically stronger than a single noisy val split. Multiple markets per cohort gives variance for the transferability claim.
**Implications:** All hyperparameter tuning, calibration fitting, and threshold setting must use CV folds on train. Test is touched only once.

### D-003: Within-cohort multiple markets, market-level split
**Status:** Locked
**Decision:** Each cohort contains multiple deadline rungs of one canonical ladder. Markets are NOT split across cohorts.
**Alternatives considered:** Pool all trades, split by timestamp.
**Justification:** `bet_correct` is a terminal-state label propagated to every trade in a market — every trade in the same market shares the same outcome via potentially-leaking feature signatures. Market-level split is non-negotiable for the transferability claim.

### D-004: Canonical ladder only, no parallel question families
**Status:** Locked
**Decision:** Train = "US strikes Iran by [date]" only. Test = "US x Iran ceasefire by [date]" only.
**Excluded parallel ladders:**
- "Will the US **next** strike Iran on [date]" (different resolution mechanic)
- "Israel x Iran ceasefire by X" (different parties)
- "Will US and Iran go to war"
- "Will Khamenei be killed"
- "US strike on Iran's nuclear facilities"
**Justification:** Defensibility ("we used the primary question family"). Avoids signal duplication where one trader's bet shows up in multiple parallel markets. Volume in canonical ladder alone is ample (~$596M).

### D-005: All deadline rungs, trade-level pre-event filter
**Status:** Locked
**Decision:** Include every rung on each ladder. Apply trade-level filter: `keep iff trade_timestamp < event_time`.
- Train cutoff: `2026-02-28 06:35 UTC` (Operation Epic Fury launch)
- Test cutoff: `2026-04-07 23:59 UTC` (end-of-day Apr 7 — handles "by April 7" markets correctly)
**Justification:** Trivially-YES post-event trades are arbitrage, not informed flow. Filtering at trade level uniformly handles all rungs without arbitrary deadline-window cuts.

### D-006: Liquidity floor 500 pre-event trades
**Status:** Locked
**Decision:** Drop markets with fewer than 500 pre-event trades.
**Result:** Train kept 63/65 markets (1 dropped). Test kept 10/10.

---

## Feature engineering

### D-007: Withhold current trade price (and `log_trade_value_usd`)
**Status:** Locked (revised in v2/v3)
**Decision:** Current trade `price` is excluded from features. `log_trade_value_usd` was also dropped after v1 because `log(value) - log(size) = log(price)` recreates the price leak.
**Justification (v1 → v2):** The "p_hat independent of market price" framing requires no current-price info in features. v1 had a hidden leak via the size×value pair → top-2 features by |coef| both reconstructed price.
**Implication:** Without any price info, model is weak on long-horizon ceasefire markets (post-Apr 7 deadlines, settled YES). v2 test AUC 0.546 vs v1 (with leak) 0.567.

### D-008: Framing C — pre-trade price as feature
**Status:** Locked
**Decision:** Use `pre_trade_price` (lag-1 within market) and rolling means of price (5min/1h/24h, all `closed='left'`) as features. The current trade's own price is still excluded.
**Justification:** Pre-trade price is the most recent observable consensus, available to a counterfactual decision-maker before this trade. Including it lets the model use price context without leaking the current trade. Trading-rule edge = `p_hat - pre_trade_price` is intact and meaningful.
**Implication:** Edge represents the model's *behavioural adjustment* to the most recent market price, not an absolute mispricing claim.

### D-009: No wallet enrichment from external sources
**Status:** Locked
**Decision:** No on-chain enrichment, no Polymarket API calls. Only data inside the HuggingFace dataset.
**Implication:** Skip Layer 6/7 features from the prior pipeline (wallet polygon age, CEX funding, cross-market entropy, etc.).

### D-010: Within-HF wallet aggregates ARE allowed
**Status:** Locked (refined from D-009)
**Decision:** `maker` and `taker` columns in HF data can be used to compute prior-trade aggregates (cumulative trades, recent volume, first-trade flag). Strictly within-HF, no external lookup.
**Features:** `log_taker_prior_trades_in_market`, `taker_first_trade_in_market`, `log_taker_recent_volume_1h_in_market`, `log_maker_prior_trades_in_market`.

### D-011: Comprehensive feature set ("kitchen sink")
**Status:** Locked
**Decision:** v3 = ~60 features across 13 groups (trade-local, time, market rolling, multi-timescale OFI, microstructure, price dynamics, token-side, consensus + contrarian, payoff, microstructure literature, Polymarket-unique, on-chain, within-HF wallet).
**Justification:** Tree-based and MLP models robust to redundant features. L1 LogReg will prune. Diverse feature set helps the sweep find what matters.

### D-012: Contrarian-trade features
**Status:** Locked
**Decision:** Add `contrarian_score` (continuous, [-1, +1]), `is_long_shot_buy` (boolean, BUY at price < 0.20), `contrarian_strength` (consensus × contrarian interaction).
**Justification:** Captures Magamyman-style "bet against consensus on asymmetric payoff" pattern. Mitts & Ofir 2026's most-cited Iran case fits this exactly.

### D-013: Polymarket-unique features (binary structure)
**Status:** Locked
**Decision:** Include `implied_variance = p(1-p)`, `distance_from_boundary`, `consensus_strength`, `log_payoff_if_correct` (asymmetric payoff explicitly), `risk_reward_ratio_pre`.
**Justification:** Bounded [0,1] price space gives unique features unavailable in stock markets.

### D-014: Microstructure-literature features
**Status:** Locked
**Decision:** Include `kyle_lambda_market_static` (per-market price impact), `realized_vol_1h`, `jump_component_1h` (RV − bipower), `signed_oi_autocorr_1h`.
**Excluded (too expensive for v3):** Roll measure, rolling Kyle lambda, VPIN volume buckets.

---

## Modelling

### D-015: 5-fold GroupKFold CV, groups = market_id
**Status:** Locked
**Decision:** GroupKFold with 5 folds, groups = `market_id`, time-respecting (sort by timestamp before fitting within each fold). Inner chronological val split for MLP early stopping.
**Justification:** Multiple markets per cohort means CV folds give k independent estimates. Groups prevent within-market leakage. 5-fold gives ~13 markets per fold — chunky enough for stable AUC.

### D-016: Isotonic calibration on out-of-fold predictions
**Status:** Locked
**Decision:** Isotonic regression fit on OOF predictions from CV (not on a held-out val cohort).
**Justification:** Replaces what val would do for calibration. OOF predictions cover the full train set with each fold's predictions made on data the fold-model didn't see.

### D-017: Class balance — no resampling
**Status:** Locked
**Decision:** Don't apply SMOTE / ADASYN / class weights beyond `class_weight='balanced'` in LogReg.
**Justification:** Class balance on train (50.3% positive) and test (50.4% positive) is essentially even.

### D-018: Anti-predictive markets caused by target bug, not model failure
**Status:** Locked (post-investigation)
**Finding:** v1 markets 1466016 (AUC 0.373) and 1706788 (AUC 0.345) had wrong targets due to using dataset `end_date` (sometimes a snapshot date) instead of parsing the deadline from the question title.
**Fix:** Parse deadline from question regex; set ceasefire announcement = end-of-day Apr 7 UTC. v2 fixed 1706788 (AUC → 0.722). 1466016 still struggles for unrelated reasons (post-Apr 7 deadline; needs price feature to predict well — see D-007).

### D-019: Sweep composition — 7 supervised models
**Status:** Leaning
**Decision (proposed):** LogReg L2 (anchor), LogReg L1 (elbow), Decision Tree, Random Forest, HistGradientBoosting, PCA→LogReg pipeline, TF/Keras MLP.
**Lecture coverage:** L02, L04, L05, L09 explicitly demonstrated.

### D-020: Unsupervised arm — IsoForest + Autoencoder
**Status:** Leaning (separate script `08_unsupervised.py`)
**Decision (proposed):** Isolation Forest (L07) + TF/Keras stacked autoencoder (L11).
**Justification:** Per original `alex_adventure.md`. Reconstruction error / anomaly score as parallel signal to compare with supervised `p_hat`.

### D-020a: Isolation Forest as primary insider-trade detector
**Status:** Locked
**Decision:** Run Isolation Forest on the trade-level feature set as a *primary* insider-detection lens, not just a baseline. Anomaly score per trade = candidate insider signal.
**Justification:** Insider trades are statistically anomalous — big size against thin liquidity, contrarian direction, late timing, asymmetric payoff. IsoForest is built to find that. The unsupervised score doesn't depend on `bet_correct`, so it's a fully independent signal.
**Cross-checks to run:**
- Correlation between IsoForest anomaly score and `bet_correct` (do anomalous trades win more?)
- Agreement between IsoForest top-k and supervised top-k by |edge| (do the two lenses flag the same trades?)
- Per-market: does IsoForest find more anomalies on the canonical Magamyman-pattern markets?
**Implication:** Adds methodological breadth ("two independent lenses agree" is stronger than one supervised model). Lecture-aligned (L07).

### D-021: Hyperparameter optimisation strategy (L13)
**Status:** Locked
**Decision:** Bayesian only (Optuna or scikit-optimize) on the supervised winner. Skip Grid + Random.
**Justification:** Most efficient method. Lecture L13 covers all three; we'll briefly contrast in the report but don't need to spend compute on Grid/Random when Bayesian dominates in practice.

### D-022: TF/Keras only for neural networks
**Status:** Locked (exam constraint)
**Decision:** PyTorch is forbidden. All NN code uses `tensorflow` + `tf.keras`.

---

## Carryovers from 2026-04-22 data-cleanup session

*The following four decisions were established in an earlier exploratory session that built train/val/test cohorts on Pontus's 74-market frame and ran a 7-model sweep. The cohort design has since been superseded (see D-001 through D-006), but the methodological lessons still apply to the current design and are recorded here so we don't relearn them.*

### D-023: `side` and `outcomeIndex` are structurally banned from features
**Status:** Locked (carryover; reaffirmed for v3)
**Decision:** Never include `side` or `outcomeIndex` as features, and never include features that derive from signed position, same-side filtering, or outcomeIndex aggregates. Specifically banned: `wallet_position_size_before_trade`, `is_position_flip`, `wallet_cumvol_same_side_last_*`, `wallet_directional_purity_in_market`, `wallet_has_both_sides_in_market`, `market_buy_share_running`, `trade_size_vs_position_pct`, `is_position_exit`, and any `*_x_*` interaction that includes raw direction.
**Justification:** Within any single market, the pair `(side, outcomeIndex)` *deterministically* encodes `bet_correct` — the cell's market resolution + which side the trade is on fully determines the label. The mapping FLIPS across YES/NO resolutions: `(BUY, idx=0)` → bet_correct=1 in a YES market, =0 in a NO market. Training on a mixed-resolution cohort and testing on a different mix → catastrophic inversion. In the prior session, tree models with these features scored ROC 1.00 train / 0.79 val / **0.04 test** — perfect inverse on a test cohort with different YES/NO mix.
**Alternatives considered:** Stratify training by resolution; reformulate target as resolution-independent (e.g., trader-skill-only). Both are scope-creep relative to dropping the offending features.
**Implications:** All directional intent must be expressed via absolute / unsigned features (size magnitudes, market-implied prob, contrarian scores that don't reference the trade's own side). The taker-side direction relative to current price is fine if encoded via signed pre-trade-price-move, not via raw `side`.

### D-024: `cumsum(bet_correct)` over prior trades is a temporal leak
**Status:** Locked (carryover)
**Decision:** Any feature computing wallet-level statistics over a wallet's prior `bet_correct` values must filter priors by `resolution_ts < current_trade_timestamp`. Naive `groupby(wallet).cumsum(bet_correct)` is forbidden.
**Justification:** A wallet's Jan 15 trade in a "by Feb 28" market has `bet_correct` only determined at Feb 28 settlement. At Jan 20 (the wallet's next trade) Feb 28 hasn't happened — the value isn't observable. Naive cumsum uses future info. In the prior session this manifested as `wallet_prior_win_rate` being the strongest linear predictor (+0.23 Pearson) primarily via leak, not skill. Dropping it dropped LogReg test ROC 0.63 → 0.53.
**Implications:** Any wallet-level cumulative-skill feature requires per-row resolution-time filtering — not a clean `cumsum()`. Cheaper alternatives: count-based features (`prior_trades`, `prior_volume`) that don't reference outcome.

### D-025: Naive-market baseline ROC is a Simpson's-paradox artefact
**Status:** Locked (carryover)
**Decision:** Do not benchmark trained models against the aggregate ROC of `p_hat = market_implied_prob`. Use trade-level residual analysis or random (0.50) as the reference.
**Justification:** Within every (market × side × outcomeIndex) cell, `bet_correct` is constant (the cell's resolution + direction determines all winners and losers). Within-cell ROC is mathematically undefined. The aggregate ROC of `p_hat = market_implied_prob` comes purely from ranking BETWEEN cells via the (side, outcomeIndex) trade-arrival distribution. On the prior session's all-NO test cohort, naive-market scored ROC 0.63 — not from market efficiency or skill, but from the cohort's specific trade-direction mix. **Efficient-market null does not imply naive-baseline ROC = 0.5.** It implies *calibration*; aggregate ROC depends on the trade-direction distribution of the test cohort.
**Implications:** RQ1b cannot be answered by aggregate ROC comparison against naive-market. Use **residual edge** instead: project `edge = p_hat − market_implied_prob` orthogonal to `market_implied_prob`, and test whether the residual predicts `bet_correct` (partial correlation, residual ROC). In the prior session, partial correlation +0.06 on test (~7σ) confirmed our model added real value above market belief, even when raw edge ROC was 0.38 (also Simpson's-paradox-driven).

### D-026: Tree-model memorisation via wallet fingerprints — sanity-check `proxyWallet`-overlap
**Status:** Locked (sanity check; complements D-015)
**Decision:** D-015 locks GroupKFold(market_id) — that prevents within-market trade leakage but not wallet-level fingerprint leakage. Add a sanity check on the test set: report metrics separately for "wallet seen in training" vs "novel wallet" trades.
**Justification:** With deep unconstrained trees + wallet aggregate features (`prior_trades_in_market`, recent volumes, etc.), trees can map a wallet's feature signature → memorise that specific wallet's training-set bet_correct distribution. Even with market-level CV groups, the same wallet can recur across train markets and test markets. In the prior session, RF train ROC 0.96 vs test 0.52 was diagnosed as wallet-fingerprint memorisation.
**Mitigations available:** (a) heavy tree regularisation (`min_samples_leaf >= 200`); (b) report seen-wallet vs novel-wallet test metrics separately; (c) ablate wallet-aggregate features and re-measure ROC drop. We're not requiring `GroupKFold(proxyWallet)` because cross-regime markets are days apart and wallet recurrence is the genuine phenomenon, but seen/novel reporting keeps the claim honest.

---

## Skipped / deferred

| Item | Status | Reason |
|---|---|---|
| Magamyman wallet sanity check | Deferred | Original plan needed wallet IDs; we said no external enrichment. May reformulate as "characteristic match" if test results justify. |
| 2025 12-Day War strike markets in train | Deferred | User scoped to 2026 events only. Could add for more data later. |
| Roll measure / VPIN buckets / rolling Kyle | Deferred | Complex to implement; defer to v4 if v3 is weak. |
| RNN / LSTM / CNN / Transformers | Skipped | Not applicable to tabular trade features. |
| SVM | Skipped | Won't scale to 1M trades. |
| KNN | Skipped | Same scaling issue. |
| SMOTE / ADASYN | Skipped | Class balance already ~50/50. |

---

## Open questions

- Should sister-market dispersion be implemented at trade level (current) or 5-min bucket level?
- For test markets that are unresolved at HF snapshot time, are our deadline-inferred targets correct? (Currently checking 4 of 10 markets had `outcome_prices` resolved; 6 used inferred targets.)

## Recently locked

### D-027: Trading rule — report both edge formulations
**Status:** Locked
**Decision:** Report both `edge = p_hat - pre_trade_price` (Framing C, market-relative) and `edge = p_hat - 0.5` (baseline-free, absolute).
**Justification:** Framing C edge measures behavioural adjustment to current market price; baseline-free edge measures absolute conviction vs uniform prior. Both are meaningful for the trading-rule evaluation and reviewers can choose which they find more compelling.

---

## Result snapshots

### v1 (baseline, 16 features, leaky log_trade_value_usd)
- Train: 1.11M trades, 63 markets
- Test: 257K trades, 10 markets
- OOF AUC: 0.594 ± 0.039
- Test AUC (calibrated): 0.567
- Test Brier: 0.244, Test ECE: 0.031
- **Issue:** `log(value) - log(size) ≈ log(price)` leak. Top features were value + size with opposite signs.

### v2 (drop log_trade_value_usd, parse deadline from question)
- OOF AUC: 0.564 ± 0.062
- Test AUC (calibrated): 0.546
- Test Brier: 0.248, Test ECE: 0.023
- 1706788 jumped 0.345 → 0.722 (deadline fix worked)
- Long-horizon markets (1571566, 1484894, 1484895) dropped 0.65 → 0.33-0.42 (lost the price leak that was helping them)

### v3 (Framing C: pre-trade price + ~60 features)
- TBD — running now

### v3 (65 features, no wallet enrichment)
- OOF AUC: 0.623 (RF), 0.629 (LogReg)
- Test AUC (calibrated): 0.615 (RF), 0.629 (LogReg)
- Test ECE: 0.031
- Per-market AUC bimodal [0.05, 1.00] for trees
- 100% top-1% precision on tree models (n=2,571)

### v3.5 (70 features, +10 within-HF wallet aggregates)
- Test AUC: 0.629 (LR), 0.899 (RF), 0.893 (HistGBM)
- Top-1% by p_hat: 100% precision (RF, HistGBM, n=2,571)
- Late-flow signature confirmed: ≤14d hit 92%, ≤3d hit 98%
- **Initial reported headline: 45× ROI on $10K bankroll** — later found to be a calculation artifact (D-029).

---

## D-029: Cost-calculation bug discovered + fixed (2026-04-28)

**Status:** Locked (post-pressure-test)
**Bug:** `pre_trade_price` in v3.5 features was `price.shift(1)` within market — the previous trade's *per-token* price, not YES probability. The cost calc in `10_backtest.py` and `11_realistic_backtest.py` treated it as YES probability (`cost = pre_yes if rooting_for_yes else 1 - pre_yes`). Wrong on ~49% of trades — specifically every trade whose previous trade was on the *opposite* token.

**Verification:** For each market, `mean(price | nonusdc_side=token1) + mean(price | nonusdc_side=token2) ≈ 1.0` across all 75 markets. Confirmed prices are per-token, not YES-normalized.

**Fix:** Compute corrected `pre_yes_price` per trade by tracking last-observed token1 and token2 prices separately:
- `pre_yes = last_token1_price` (if token1 has traded)
- else `1 - last_token2_price` (if only token2 has traded)
- else `0.5` (no prior trade)

Implemented in `11_realistic_backtest.py::compute_pre_yes_price_corrected()`.

**Impact:**

What was unaffected:
- Bet_correct target derivation (uses categoricals, not prices)
- Model AUC and Brier scores
- Top-1% by p_hat precision (still 100% for RF and HistGBM)

What was affected:
- Top-1% by edge precision dropped: HistGBM 1.000 → 0.545, RF 1.000 → 0.927
- Realistic ROI on $10K bankroll: HistGBM general_ev 40.5× → **0.14× (+14%)**
- HistGBM top5pct_edge: 43.6× → **0.11× (+11%)**
- Most other strategies: 0.00× to 0.07×
- Some strategies turn negative under simple flat-stake PnL

**Implication for thesis claim:** The model identifies winning trades correctly (top-1% precision is real), but most picks are heavy-favorite bets at high cost (~0.95) with small payoffs (~5% per win). Best-case mirror return under realism is +14% on $10K bankroll, NOT +4054%. The Magamyman-style asymmetric-payoff insider pattern is not isolated by this feature set, consistent with IsoForest null result and home-run filter zero hits.

**Files updated:**
- `11_realistic_backtest.py` — compute_pre_yes_price_corrected + uses it in compute_cost_and_edge
- `10_backtest.py` — patched (D-030)
- `notes/pressure-test.md` — full pressure-test results + headline finding

---

## D-030: 10_backtest.py patched + skip-if-cached toggle (2026-04-28)

**Status:** Locked
**Decision:** Apply the same `compute_pre_yes_price_corrected` to `10_backtest.py`. Sort `test` by (market_id, timestamp) before training; reorder cached predictions via `sort_key` to keep alignment. Add a skip-if-all-preds-exist short-circuit (`RETRAIN=1` env var to force fresh).
**Justification:** Workers are deterministic (`random_state=42`); fresh runs produce identical npz files. Skip saves ~10 min per iteration during analysis.
**Impact (`10_backtest.py` now reports, RF):**
- top-1% by p_hat: 100% hit, +$5.5K (heavy-favorite picks, tiny payoffs)
- top-1% by edge: 92.5% hit, +$383K
- general_ev (edge>0.02): 45.3% hit, +$174K on 115K trades
- home_run filter: 4 picks, 0% hit (asymmetric-info pattern not detected)

LogReg's top-1% by edge collapsed from 98.2% → 0.3% — its picks were exploiting the per-token-price miscoding, not real edge.

---

## D-031: Phase 2 re-run with corrected pre_yes_price (2026-04-28)

**Status:** Locked
**Decision:** Re-run B1a/B1b in `phase2_falsification.py` against the corrected YES probability. Old results were SUSPECT.
**Findings:**

**B1a (consensus decomposition of top-1% picks):**
- RF: **100% with-consensus, 0% against-consensus.** 5 unique markets, top market 41% of picks. Mean YES prob = 2.1%.
- HistGBM: **100% with-consensus.** 4 unique markets, top market 68% of picks. Mean YES prob = 3.6%.
- LogReg: 79% with-consensus (the 21% against-consensus picks have hit rate 0.2% — they lose).

**B1b (naive consensus baseline) — FAILED:**
- naive "score = consensus_strength × is-with-consensus": **AUC 0.844, top-1% precision 100%**
- RF: AUC 0.888, top-1% 100%; HistGBM: AUC 0.887, top-1% 99.9%
- The model adds ~0.04 AUC over a 3-line naive rule. Top-1% precision is fully reproducible by naive consensus alignment.

**Implication:** The "100% top-1% precision" headline is real but largely a consensus-alignment artifact. The model's marginal contribution is modest (~5pp of AUC). The asymmetric-information detection claim is not supported by these tests.

**Defensible report framing:** Replicate Mitts & Ofir's late-flow signature (≤3d trades hit 98%); identify winning trades on held-out cohort with high precision; explicitly disclose that top picks concentrate in consensus-correct markets.

---

## D-032: Sensitivity sweep — cost_floor × copycats-N (2026-04-28)

**Status:** Locked
**Decision:** New script `12_sensitivity_sweep.py`. Sweep 5 cost_floors × 6 copycat-N values for 3 (model, strategy) targets. $10K bankroll, 5% max bet.

**Findings:**

**Cost floor barely matters in [0.001, 0.10].** Almost no trade has cost in that range; floor doesn't bind. Floor=0.20 jumps drastically (HistGBM general_ev: 13.8% → 37.5% at N=10) because it admits more trades through the edge>0.02 filter (n_executed 2,099 → 3,705) and caps loss size. Defensible to keep 0.05 as the headline floor.

**Copycats N is the dominant sensitivity:**

| Strategy | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| HistGBM general_ev | +23.7% | +20.3% | +13.8% | +4.6% | -1.7% | -7.5% |
| HistGBM top5pct_edge | +27.3% | +24.6% | +10.6% | +3.5% | +1.4% | +0.6% |
| RF general_ev | +13.4% | +15.3% | +7.4% | -4.9% | -5.3% | -4.5% |

RF general_ev is non-monotonic (peaks at N=5): at N=1 the per-trade volume cap is generous, so each bet is bigger and hits the 20% per-market concentration limit faster (only 1,152 executed). At N=5 the cap shrinks, more trades fit across markets, n_executed jumps to 1,894. After N=5 the smaller per-trade payoffs dominate. HistGBM picks spread across more markets, so concentration isn't binding at N=1.

**Implication:** Cost floor has no meaningful effect; assumed competition for fills (N) is the critical assumption.

---

## D-033: ML-paper headline uses N=1, not N=10 (2026-04-28)

**Status:** Locked
**Decision:** For the MLDP report, headline ROI numbers are reported under N=1 (no concurrent mirroring). Sensitivity to N is disclosed in a limitations section pointing at the heatmap.
**Alternatives considered:** N=10 default (per `11_realistic_backtest.py`), or report a range without a single headline.
**Justification:**
- The course grades model quality, not trading-system viability. N=1 measures "what does this model's signal extract from the data" — a clean ML metric.
- `liquidity_scaler=0.10` (i.e. N=10) was a defensive realism choice with no internal data calibration (pressure-test E2: "10× copycats appropriate — UNTESTED, picked out of thin air").
- N=1 is bounded above by available historical orderbook depth — a legitimate ML upper bound.
- The +24% / +14% / -7% sensitivity table makes the limitation explicit and quantified.

**Headline numbers (cost_floor=0.05, $10K bankroll, 5% bet, N=1):**
- HistGBM general_ev: **+23.7%** (+$2,368, n=1,338 executed)
- HistGBM top5pct_edge: **+27.3%** (+$2,732, n=601 executed)
- RF general_ev: +13.4% (+$1,338, n=1,152 executed)

**Report framing:** Lead with HistGBM general_ev (+23.7%) as the headline. Discuss N-sensitivity as a deployment limitation: N=10 → +13.8%, N=100 → −7.5%.

---

## D-034: Research question retired & rewritten (2026-04-28)

**Status:** Locked
**Decision:** Replace D-001's "informed-flow detection across regimes" claim with "supervised trade-outcome prediction with cross-regime transfer + decomposition of what the model detects." See revised D-001 above.
**Why retired:** B1b pressure test (D-031) showed a 3-line naive consensus rule matches the model's top-1% precision (100%) at AUC 0.844 vs RF 0.888. The "asymmetric-information / informed-flow" framing is not supported by the evidence. Continuing to claim it would be intellectually dishonest.
**What stays:**
- Cohort design (cross-regime: Iran-strike train, Iran-ceasefire test)
- Feature engineering (v3.5, 70 features)
- Models, calibration, sensitivity analysis
- All pressure-test findings — they become the central scientific contribution
**What changes in framing:**
- "Asymmetric information" appears only in the literature review (Mitts & Ofir as motivation), not as our claim
- Decomposition section (consensus alignment, late-flow replication, naive baseline) is promoted to a primary results section
- Negative finding (model is high-precision consensus detector) is reported as the central result, not as a limitation
- Cross-regime transfer is the secondary claim, supported by AUC + top-1% precision on held-out cohort

**Implications for Pontus discussion:** The co-report doesn't need to choose between cohorts on methodological purity grounds — it picks the cohort that best supports the (now narrower) RQ. Cross-regime cohort gives statistical power for the prediction claim. Pontus's strict cohort can serve as a robustness check against inferred-resolution risk (A2 still UNTESTED).

---

## D-035: SELL handling — ablation queued (2026-04-28)

**Status:** Open — ablation queued, not yet run
**Decision:** Run two ablations before locking the headline:
- **Ablation A — drop SELLs entirely:** retrain RF + HistGBM on train minus all SELL trades, test on test minus all SELL trades. Compare AUC + top-1% precision against current pipeline.
- **Ablation B — explicit SELL-semantic features:** add `sell_is_closing` (= 1 iff taker had a prior BUY on same nonusdc_side in same market) and `sell_is_open_short` flags, retrain, see if AUC moves.

**Rationale:** Pressure-test A1 found 36% of all SELLs are closing trades (existing positions being unwound, not fresh directional bets). Treating these as fresh directional bets via `bet_correct` is technically wrong — a closing SELL doesn't "bet" on the opposite side; it just realises P&L. At the model's confident edge, only 0.2% of top-1% SELLs are closing, so the effect on the headline is likely small. But the ablation gives us a defensible answer: either SELL-free model is the cleaner story, or the explicit features add signal, or current pipeline is robust to it.

**Decision rule:**
- If Ablation A gives a cleaner AUC / top-1% number, drop SELLs from both train and test, report SELL-free model as headline.
- If Ablation B adds material signal (>0.01 AUC), keep SELLs and add the explicit features.
- If neither moves the needle, current pipeline stands and A1 stays a NOTE in the limitations section.

**Implications:** Possible mid-stream pipeline change before report drafting. Run after A2 verification.

---

## D-036: Wallet feature scope expansion — queued (2026-04-28)

**Status:** Open — scoped, awaiting prioritisation
**Decision:** Three wallet-feature additions in priority order:

1. **`taker_resolved_winrate_history`** (within-HF, allowed) — for each trade, fraction of this taker's prior *resolved* trades that won. Requires per-market resolution-time mapping and per-taker O(n²) cumulative compute. ~2 hours. Most direct test of "does the model favour known-winner wallets?" If yes, may shift top-1% picks toward contrarian / against-consensus trades and partially recover an asymmetric-info finding.

2. **Cross-market wallet entropy** (within-HF, allowed) — Shannon entropy over categorical buckets of markets a taker has previously traded. Coarser proxy exists as `log_taker_unique_markets_traded` (#63 in feature list), but full entropy is more informative and matches Pontus's Layer 7. ~1 hour.

3. **On-chain identity features** (Pontus's Layer 6 — Polygon wallet age, nonce, CEX deposits, funding source) — **NOT allowed for our pipeline** under the no-external-API constraint. Available only via Pontus's Etherscan enrichment in the co-report. We document the limitation, defer to Pontus's contribution.

**Outcome to test:** Do (1) + (2) change the consensus-alignment of top picks? If they isolate trades by wallets with strong prior records *against* market consensus, the B1b finding could be revisited.

**Implications:** If (1) materially improves AUC or shifts the consensus-alignment story, the report's central decomposition section needs updating. Plan to run (1) first; if null, defer (2). Ordered after A2 + D-035.

---

## D-037: Rigor additions for report drafting (2026-04-28)

**Status:** Open — scoped, queued for ~90 min single-script run before report drafting
**Decision:** Add a `13_rigor_additions.py` script that produces three standard-ML-practice additions, output to `outputs/rigor/`:

1. **Bootstrap CI on test AUC + DeLong (or paired-bootstrap) test for RF vs HistGBM.** Convert point estimates into 95% CIs and a hypothesis test for the model-comparison claim. ~30 min implementation.
2. **Permutation importance per model.** Replaces the current MDI (`clf.feature_importances_`) outputs as the headline feature-importance number. Sklearn one-liner. ~30 min.
3. **Learning curves (AUC vs train fraction).** Tells us whether the +23.7% headline is data-bound or model-bound. ~30 min.

**Justification:** All three are textbook CBS MLDP techniques and standard expectations for a quantitative ML paper. Currently we report point estimates without uncertainty, MDI without permutation, and have no learning-curve analysis. These additions elevate rigor without changing the narrative.

**Deferred (Tier 2, only if drafting reveals a gap):**
- Hyperparameter tuning via GridSearchCV on RF (~1 hr)
- SHAP values for top picks (~1 hr)
- Stacking ensemble RF + HistGBM + LogReg (~1-2 hr) — would change the headline narrative mid-draft, defer
- Confusion matrix at threshold 0.5 (~10 min) — trivial inclusion, do as part of figure prep
- Platt scaling alongside isotonic (~30 min) — robustness check, not critical

**Skipped (intentionally):** MLPs / autoencoder (Pontus's territory + TF/Keras port effort), SMOTE / oversampling (no imbalance, base rate 50.4%), KNN / SVM (computational cost > value on 1.1M trades), K-means / DBSCAN (no clustering question), polynomial / interaction features (70 features plenty; trees handle interactions natively).

**Implications:** Run before report drafting. Output integrates into `Results` and `Methodology` sections of the report. Don't add new headline numbers (the AUC + ROI numbers are locked); these are *qualifiers* on existing numbers.

---

## D-038: PCA component count via scree elbow (2026-04-29)

**Status:** Locked — implemented in `v4_final_ml_pipeline/scripts/03_sweep.py::find_pca_elbow_k()`

**Decision:** Choose the number of PCA components for the PCA→LogReg pipeline (Stage 4, model 6) algorithmically via the geometric elbow method on the scree curve, not as a hardcoded constant.

**Procedure:**
1. Standardise X_train (PCA is scale-sensitive).
2. Fit PCA with `K_max = min(50, n_features) = 50` components.
3. For each candidate K ∈ [1, K_max], compute the perpendicular distance from `(K, cumvar[K])` to the chord connecting the first and last points of the cumulative-variance curve.
4. Pick K with maximum perpendicular distance. Floor at 2.
5. Save scree plot (variance ratio + cumulative-variance with 95% line and elbow marked) to `outputs/sweep_idea1/pca_logreg/scree.png`. Save full diagnostics (var_ratio array, cumvar array, var_at_elbow, var_at_k20) to `pca_selection.json`.

**Alternatives considered:**

| Option | Why rejected |
|---|---|
| Fixed K=20 (previous) | Arbitrary — no defence if asked "why 20?" |
| Variance threshold (`n_components=0.95`) | Principled but K varies with feature changes; harder to compare v3.5 vs v4 |
| Kaiser criterion (eigenvalue > 1) | Tends to over-retain; less common in modern ML pedagogy |
| CV sweep over K ∈ {5, 10, 20, 30, 50} | Predictive-task-aligned but conflates dim-reduction question with classification question; the L05 box is about *the projection*, not about end-to-end optimisation |

**Justification:** The L05 dimensionality-reduction box wants a principled, reproducible K choice. The scree elbow is the textbook geometric construction (Cattell 1966). Deterministic, no extra dependency (`kneed` not required — geometric distance to chord is one numpy line), and the 2-panel plot is a drop-in figure for the report's Methodology section.

**Implications:**
- Comparison-table row for `pca_logreg` is now driven by elbow-K, not K=20. Need to re-run Stage 4 once v4 data lands and report whichever K the elbow lands on.
- Per-PC importance output (`pca_logreg_importance`) still keys on `pc_0..pc_{K-1}` — no schema change.
- If the elbow lands very low (K=3-5, plausible on highly-collinear feature sets), the PCA→LogReg model will likely lose AUC vs the K=20 baseline. That's expected — the point of this model is the L05 box, not headline AUC. Headline still belongs to RF/HGBM/LightGBM.
- Methodology paragraph: "We selected the number of principal components by the geometric elbow on the scree curve, retaining {var_at_elbow:.0%} of variance with K = {elbow_k} components."

---

## D-039: MLP excluded from economic backtest by default (2026-04-29)

**Status:** Locked unless Stage 4 results overturn it. `_backtest_worker.py` knows about LogReg L2 / RF / HistGBM / LightGBM only — MLP is intentionally not registered.

**Decision:** Run MLP through Stage 4 classification scoring (`03_sweep.py`) but do NOT backtest it economically (Stage 8) by default. MLP earns the L09 deep-learning box on classification metrics alone.

**Rationale:**
- MLP is in the pipeline to satisfy the L09 deep-learning lecture box, not as a deployment candidate. Tabular data with 76 features is the regime where tree ensembles routinely beat feed-forward NNs.
- Economic backtests already cover RF + HGBM + LightGBM + LogReg. Adding MLP inflates the comparison table without changing the report's narrative.
- Engineering cost is non-trivial: `_backtest_worker.py` is shaped around `factory()` + `predict_proba` + isotonic, which doesn't fit MLP's training loop (per-fold scaler, inner chronological hold-out for early stopping, `keras.backend.clear_session()` between folds). Backtesting MLP needs either a parallel `_backtest_worker_keras.py` or a refactor of the worker to dual-handle. ~1 hr of architectural work.

**Conditional override (decision-tree gate):** if after Stage 4 the decision tree from the README fires — *"MLP beats best tree on test AUC AND DeLong p < 0.05"* — MLP becomes a headline candidate and we need ROI numbers to defend the call. In that scenario only, build the keras worker variant and re-run Stage 8 with MLP included.

**Implications:**
- Report Methodology paragraph: "Economic backtests are reported for the four top non-NN models (RF, HistGBM, LightGBM, LogReg L2); the MLP's classification performance is reported in Table N. The MLP was not selected for deployment evaluation given its marginal AUC vs. tree models."
- If `_backtest_worker_keras.py` is later added, log a separate decision (D-04x) extending this one rather than mutating it.

---

## D-040: Realistic backtest concentration bookkeeping bug — fixed (2026-04-29)

**Status:** Patched in `v4_final_ml_pipeline/scripts/11_realistic_backtest.py`. `12_sensitivity_sweep.py` inherits via `importlib`-imported `realistic_backtest`.

**Bug:** `open_positions[mid]` (the per-market $-committed counter used for the concentration limit) was decremented by `return_amt` on resolution. For losing bets, `return_amt = 0`, so the original `bet` amount stayed "tied up" forever. With many losses on the same market, the `max_concentration_pct × capital` cap progressively locked the strategy out of further bets in that market — even though the loss had been realised at entry and the capital was free.

**Fix:** `open_resolutions` entries are now 3-tuples `(res_ts, return_amt, entry_bet)`. `release_resolved` decrements `open_positions[mid]` by `entry_bet` (the original commitment), not `return_amt`. Same bookkeeping for wins (decrement by entry_bet, capital += payoff−gas) and losses (decrement by entry_bet, capital += 0). End-of-test `final_capital = capital + sum(open_positions.values())` now sums to the right value because all positions clear cleanly.

**Implications:**
- Stage 8.2 ROI numbers re-run on v4 preds will be modestly higher than the v3.5 numbers, *before* any wallet-feature signal kicks in. The previous v3.5 +14% headline was a conservative undercount caused by this bug.
- The methodological narrative is unchanged (still capital-aware, still N=10 copycats, still 5% max bet) — only the bookkeeping is fixed.
- When writing the report, do not compare v4 ROI against v3.5 ROI as a like-for-like apples-to-apples — the bookkeeping fix is a separate effect from the wallet features. The honest comparison is **v4-with-fix vs v3.5-with-fix**, which would require regenerating v3.5 numbers under the patched bookkeeping. Decide before drafting whether to (a) re-run v3.5 with the fix and report both as "post-fix" baselines, or (b) caveat in-text that the v4 ROI lift includes a small mechanical contribution from the bookkeeping correction.

**Discovery:** Spotted during the v4-pipeline file-by-file review (review session, 2026-04-29) by tracing what happens on the loss branch of `realistic_backtest`. The second `return_amt = 0.0` on the loss path overrides the gas-cost assignment, and `release_resolved` only ever subtracts `return_amt` from `open_positions[mid]`. Patched same session.

---

## Pressure-test summary (final, post-fix)

Phase 1 (9 quick verifications): NO FATAL FAILURES
- C1, D1, D2, D3, D4: PASS — pipeline integrity OK
- A3: PASS — answer1 == "Yes" for all 75 markets
- C2: PASS — all 4 dominant test markets confirmed NO via outcome_prices
- F2: NOTE — 1.5% true duplicates in test (after dedup on tx_hash + log_index)
- F1: NOTE — HF captures ~44% of market.volume (external-validity caveat)

Phase 2 (claim falsification, post-fix):
- **B1a: PASS by threshold rule, but RF/HistGBM are 100% with-consensus** — picks are pure consensus detection
- **B1b: FAILED — naive consensus baseline matches model on top-1% (100%) and gets within 0.04 AUC**
- A1: NOTE — 36% of all SELLs are closing trades, but only 0.2% in top-1% picks

Phase 2.5 (cost bug fix): see D-029. Bug inflated ROI ~400×.
Phase 3 (sensitivity): see D-032 / D-033. Cost-floor robust in [0.001, 0.10]; copycats-N is the dominant lever. Headline reported at N=1 for ML framing.
