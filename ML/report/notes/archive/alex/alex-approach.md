# Alex's approach — Pipeline summary for Pontus

*Single source of truth for Alex's trade-outcome prediction pipeline (cross-regime: Iran strike → Iran ceasefire). Last updated 2026-04-28. For comparison against Pontus's approach.*

---

## Research question

**Can a supervised model predict trade outcomes on Polymarket geopolitical event markets, and do the predictions transfer across event regimes?**

Train on Iran-strike countdown markets; test on held-out Iran-ceasefire countdown markets. Then **decompose what the model is actually detecting** — this decomposition is the central scientific contribution.

Motivation (literature): Mitts & Ofir 2026 documents late-concentrated informed flow before US military and diplomatic events. We initially framed this work as "asymmetric-information detection" but **retired that framing** after pressure tests showed top picks are 100% consensus-aligned and a 3-line naive consensus rule matches our top-1% precision (B1b finding). The model is a high-precision *consensus detector*, not an *informed-flow detector* — and that finding becomes the central result rather than a footnote.

---

## Cohort design

| | Train | Test |
|---|---|---|
| Question family | "US strikes Iran by [date]" | "US x Iran ceasefire by [date]" |
| Resolution event | Israel-Iran-US strike, 2026-02-28 06:35 UTC | Trump's ceasefire announcement, 2026-04-07 |
| # markets | 63 | 10 |
| # trades (pre-event) | 1.11M | 257K |
| Test base rate (bet_correct) | — | 50.4% |

**Strict rules:**
- Markets are NOT split across cohorts (`bet_correct` is a market-terminal label, so within-market splits leak).
- Pre-event filter: `keep iff trade_timestamp < event_time`. Verified `train.timestamp.max() < strike_event` and `test.timestamp.max() < ceasefire_event`.
- Liquidity floor: dropped 1 train market with <500 pre-event trades.
- 4 of 10 test markets confirmed NO via outcome_prices; 6 inferred from deadline + event timing (deadline ≥ Apr 7 → resolved YES; else NO).

---

## Feature engineering (v3.5, 70 features)

70 features across 7 families:

| Family | # | Examples | Notes |
|---|---|---|---|
| Trade-level | ~15 | `pre_trade_price`, `usd_amount`, `side_buy`, `outcome_yes` | Pre-trade price was bug-prone (see "Bug found" below) |
| Time-to-deadline | ~5 | `log_time_to_deadline_hours`, `is_late_trade` | Drives the late-flow signature |
| Rolling stats | ~12 | 1h/6h/24h windows on volume + price changes | All use `closed='left'` (verified for all 12 calls) |
| Wallet position | ~6 | `taker_position_size_before_trade`, cumcount-based | Computed via `cumsum().shift(1)` per (market, wallet) |
| Market context | ~10 | Order book depth, market age, prev-day volume | |
| Kyle lambda | ~4 | Price impact per unit volume | Measured robust to ablation |
| Other | ~18 | Various aggregates | |

**Leakage controls (all verified in pressure tests):**
- `pre_trade_price = price.shift(1)` within market — strictly excludes current row
- All `.rolling(...)` use `closed='left'` (12/12 verified)
- Wallet cumsum uses `cumsum().shift(1)` with first-row-of-group reset
- StandardScaler refit per fold (no train→val leak)
- 5-fold GroupKFold on `market_id` for CV

**Hard exclusions:** `side`, `outcomeIndex` (terminal labels), `log_trade_value_usd` (causes a `log(value) - log(size) ≈ log(price)` leak from v1).

---

## Models tested + selected

5-fold GroupKFold CV on train, isotonic calibration on OOF, final fit on full train.

| Model | OOF AUC | Test AUC (calibrated) | Top-1% by p_hat precision |
|---|---|---|---|
| LogReg L2 (scaled) | 0.611 | 0.629 | 79.0% |
| Random Forest (n=200, depth=10) | 0.880 | **0.899** | **100.0%** (n=2,571) |
| HistGBM (max_iter=200) | 0.875 | 0.887 | 99.9% |

Sweep included 7 supervised models + IsoForest unsupervised (null result).

**Selected:** Random Forest as headline. HistGBM as secondary (better economic numbers under N=1).

---

## Critical findings

### What works (verified)

1. **High precision at top-1%.** RF: 100% hit on 2,571 trades; HistGBM: 99.9%.
2. **Late-flow signature replicates Mitts & Ofir.** ≤14d trades hit 92%; ≤3d trades hit 98%.
3. **Beats baselines.** Random 1% = 51%, follow-consensus loses money on full-volume.
4. **Pipeline integrity.** Phase 1 of pressure-test: 9 leakage/integrity checks, no fatal failures.

### What does NOT work (and how we found out)

1. **Asymmetric-information detection: NOT supported.** Phase 2 falsification:
   - **B1a:** RF and HistGBM top-1% picks are **100% consensus-aligned**. Picks concentrate in 4-5 NO-resolved markets at ~2-4% YES probability — cheap NO bets.
   - **B1b:** A 3-line naive rule "score = consensus_strength × is-with-consensus" achieves AUC 0.844 and **top-1% precision 100%**. Models add only ~0.04 AUC over naive.
   - The high-precision signal is largely consensus alignment, not informed-flow detection.
2. **IsoForest unsupervised arm: NULL.** Anomaly score uncorrelated with `bet_correct` (corr -0.007). Magamyman-style asymmetric-payoff pattern is not isolated by this feature set without wallet-identity enrichment.
3. **Home-run filter: 0 hits** under any reasonable cutoff in test.

### Bug discovered + fixed (mid-analysis)

`pre_trade_price` in the v3.5 features was the previous trade's *per-token* price (each Polymarket token has its own price), NOT YES probability. We assumed it was YES probability for ~3 weeks. This inflated a reported 45× ROI to its corrected value of +14% under N=10 realism (or +24% under N=1). Verified: across all 75 markets, `mean(price | token1) + mean(price | token2) ≈ 1.0`. Fix: track last-observed token1 and token2 prices separately, reconstruct YES probability per trade. Both `10_backtest.py` and `11_realistic_backtest.py` patched.

This bug did NOT affect: AUC, calibration, top-1%-by-p_hat precision, or model training (which uses bet_correct only).

This bug DID affect: cost calculation, edge = p_hat - cost, all strategy masks using edge, all PnL math.

---

## Backtest setup + headline numbers

Two scripts, two purposes:

- `10_backtest.py` — flat-stake $100/trade, no capital constraint. Reports raw economic statistics.
- `11_realistic_backtest.py` — capital-aware execution: bankroll, per-trade size limits, concentration cap, gas, slippage, copycat liquidity sharing.

### Realism parameters (locked)

| Parameter | Value | Rationale |
|---|---|---|
| Bankroll | $10,000 | Mid-grid |
| Max bet per trade | min($100, 5% × capital, 10% × original_trade_usd × liq_scaler) | Per-trade volume cap is binding |
| Concentration limit | 20% of capital per market | Prevents single-market wipeout |
| Cost floor | 0.05 | Caps payoff at 19× per win — testable below |
| Copycat-N | 1 (ML headline) / 10 (deployment-realistic) | See sensitivity |
| Gas | $0.50 per trade | Polygon mainnet typical |
| Slippage surcharge | 5% if bet > 25% of effective trade USD | Approximation |
| Position release | At market resolution time | Tracked per-market |

### Headline numbers (cost_floor=0.05, $10K bankroll, 5% max bet, N=1)

| Strategy | Model | ROI | n_executed |
|---|---|---|---|
| general_ev (edge>0.02) | HistGBM | **+23.7%** | 1,338 |
| top5pct_edge | HistGBM | **+27.3%** | 601 |
| general_ev | RF | +13.4% | 1,152 |
| home_run (edge>0.20 + late + cheap) | any | 0% | n=0 |

### Sensitivity (Phase 3, full sweep at `outputs/backtest/sensitivity/`)

**Cost floor barely matters** in [0.001, 0.10] — no trade has cost in that range, floor doesn't bind. Floor=0.20 admits more trades and inflates ROI.

**Copycat-N is the dominant lever:**

| Strategy | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| HistGBM general_ev | +23.7% | +20.3% | +13.8% | +4.6% | -1.7% | -7.5% |
| HistGBM top5pct_edge | +27.3% | +24.6% | +10.6% | +3.5% | +1.4% | +0.6% |
| RF general_ev | +13.4% | +15.3% | +7.4% | -4.9% | -5.3% | -4.5% |

RF is non-monotonic (peaks at N=5 because at N=1 the per-trade volume cap is generous and trades hit the 20% per-market concentration limit faster).

---

## Decisions made (with alternatives considered)

| # | Decision | Alternative considered | Why this |
|---|---|---|---|
| Cohort | Iran strike → Iran ceasefire (cross-regime) | Within-strike-only (within-regime) | Cross-regime is the stronger thesis claim |
| CV | 5-fold GroupKFold on market_id | Train/val/test split, time-based CV | Market-level split is non-negotiable; multiple-market variance > single noisy val |
| Calibration | Isotonic on OOF | Platt scaling | Isotonic is non-parametric, no shape assumption |
| Best model | Random Forest | HistGBM, LogReg, NN | RF wins AUC; HistGBM close second with better economic numbers |
| ROI headline | N=1 (no copycats) | N=10 (deployment-realistic) | ML report grades model quality; N=1 is the clean upper bound on what the signal earns against historical orderbook depth |
| Cost floor | 0.05 | 0.001, 0.10, 0.20 | Insensitive in [0.001, 0.10]; 0.05 is conventional |

Full register: `notes/design-decisions.md` (33 entries, append-only).

---

## Open questions / disagreement candidates

1. **RQ has been narrowed (D-034).** No longer claiming "asymmetric-information detection." Now: "predict trade outcomes + cross-regime transfer + decompose what the model detects." The decomposition (consensus-alignment, late-flow replication, naive-baseline benchmark) is now the primary contribution. Open to discussion on whether this framing satisfies the course's expected scope.

2. **N=1 vs N=10 as headline ROI.** ML framing argues N=1 (signal upper bound). Deployment framing argues N=10 (defensive realism). We chose N=1 with sensitivity disclosure. Open to debate.

3. **HF dataset coverage.** Median ratio of `sum(usd_amount) / market.volume` is 0.44 — HF captures ~half of reported volume per market. External-validity caveat, not internal claim.

---

## Next steps (scoped, deferred)

Listed in rough order of expected impact-per-hour:

1. **A2 verification (15 min).** Confirm 6 inferred-resolution test markets via Polymarket UI / news search. Gating item before locking cross-regime cohort as headline. Currently UNTESTED in pressure-test register.

2. **SELL handling (~1-2 hours).** Per A1 finding (36% of all SELLs are closing trades, but only 0.2% in top-1% picks), the model is incidentally robust to SELL semantics at its confident edge. But it might do better if SELLs were either (a) removed entirely from train + test, or (b) feature-engineered so the model can distinguish closing-SELL from open-short. **Plan:**
   - **Ablation A:** drop all SELL trades from train and test, retrain RF + HistGBM, compare AUC and top-1% precision. If results unchanged or improved, defensible to report SELL-free model as headline.
   - **Ablation B:** add explicit `sell_is_closing` flag (= 1 iff this taker had a prior BUY on same nonusdc_side in same market) and `sell_is_open_short` flag, retrain, see if AUC moves.
   - Decision rule: if (A) gives a cleaner number, drop SELLs and report. If (B) adds material signal, keep SELLs with the explicit features. If neither, current pipeline stands.

3. **Wallet features — scope expansion (~3-4 hours).** Pontus's pipeline has on-chain identity features (Polygon wallet age, transaction nonce, CEX deposit history, funding source) and cross-market wallet entropy that ours lacks. Scope:
   - **`taker_resolved_winrate_history`** (the one already deferred): for each trade, fraction of this taker's prior *resolved* trades that won. Tests "is this a known winner" directly. Requires resolution-time mapping per market and per-taker O(n²) compute over the train+test cohort. ~2 hours.
   - **Cross-market entropy** (Pontus's Layer 7): Shannon entropy over the categories of markets a taker has previously traded. Coarser proxy already exists as `log_taker_unique_markets_traded` (#63), but full entropy is more informative. ~1 hour.
   - **On-chain identity features** (Pontus's Layer 6): wallet age (block of first activity), nonce, CEX deposit/withdrawal counts, funding source. Requires Etherscan API enrichment outside the HF dataset — **violates the no-external-API constraint** for our pipeline. *Can only be used as Pontus's contribution to the co-report, not added to ours.*
   - Outcome to test: do these features change the consensus-alignment of top picks? If they isolate trades by wallets with strong prior records *against* market consensus, the asymmetric-info finding might re-open.

4. **Magamyman/insider pattern test (~2 hours).** Subset of (3) — specifically the `taker_resolved_winrate_history` feature. If it shifts top-1% picks toward contrarian + against-consensus + cheap-side trades, the asymmetric-info claim could be partially recovered.

5. **Maduro-removal markets as additional train cohort (~2 hours).** Could compress per-market AUC bimodality, broaden train regime variety. Not the highest priority unless cohort breadth becomes a discussion point.

### Rigor additions for the report (D-037)

Three statistical-rigor additions to elevate the report before drafting. Together ~90 min as a single `13_rigor_additions.py` script writing to `outputs/rigor/`:

6. **Bootstrap CI on AUC + DeLong test (~30 min).** Convert "RF AUC 0.899" point estimate into "0.899 [lower, upper]" with 95% CI. Run a DeLong (or paired-bootstrap) test for whether RF significantly beats HistGBM (0.887). Lets the report defend "RF is the best model" rigorously instead of by point estimate alone.

7. **Permutation importance (~30 min).** Currently using MDI (`clf.feature_importances_`) which is the weak version. Permutation importance is the modern standard. One sklearn call per model. Lets us claim "kyle_lambda contributes X% of test AUC" with proper backing. Outputs go alongside existing `feature_importance.json` files.

8. **Learning curves — AUC vs train size (~30 min).** Plot whether more data would help (curve still rising) or the model has saturated (flatlined). Strong robustness signal: answers whether the +23.7% headline is data-bound or model-bound. Standard inclusion in a quantitative ML paper.

**Tier 2 (depth, not critical — skip unless drafting reveals a gap):**

9. **Hyperparameter tuning on RF + HistGBM (parked — pick a method later).** Won't move AUC much (already strong) but adds rigor for viva defense. Compute estimates on M4 Pro, ~30-45s per RF fit × 5 folds:
   - **HalvingRandomSearchCV** (sklearn) — ~45 min. Successive halving, no new dependency. Best speed/rigor tradeoff.
   - **Optuna TPE / Bayesian** — ~2.5 hrs for 50 trials. Modern best practice; learn-once tool.
   - **Manual staged tuning** (depth → leaf → features) — ~30 min. Bulletproof, easy to explain.
   - **GridSearchCV** — ~4-8 hrs for a meaningful 3-axis grid. Skip at our data scale.
   - **Defended defaults paragraph** — 0 min compute. Justify current `n_estimators=200, max_depth=10, min_samples_leaf=200` from `√(n)` reasoning + D-026 ablation. Acceptable as the report's primary defense if time-constrained.
   - All MUST use `GroupKFold(groups=market_id)` to preserve the cross-regime transfer claim.

10. **SHAP values for top picks (~1 hr).** Explain individual predictions. Lets us write "the model's confidence on these picks is driven by features X, Y, Z" with quantitative attribution. Shows interpretability awareness.

11. **Stacking ensemble (~1-2 hr).** RF + HistGBM + LogReg → meta-learner. Could push AUC slightly higher but changes the headline narrative mid-draft. Defer unless drafting hits a "best single model isn't enough" wall.

12. **Confusion matrix at threshold 0.5 (~10 min).** Trivial figure addition. Standard inclusion that reviewers expect.

13. **Platt scaling (calibration alternative, ~30 min).** Side-by-side with isotonic. Probably won't change anything but shows method awareness.

**Skipped intentionally** (course-relevant but wrong fit): MLPs / autoencoder (Pontus's territory + TF/Keras port effort), SMOTE / oversampling (no imbalance — base rate 50.4%), KNN / SVM (computational cost > value on 1.1M trades), K-means / DBSCAN (no clustering question), polynomial / interaction features (70 features is plenty, trees handle interactions natively).

---

## Comparison points for Pontus

Things to compare across our two ML approaches:

1. **Cohort design.** Are we in the same train/test markets? If not, why?
2. **Feature engineering.** What's in your feature set that's not in mine? What did you exclude that I kept?
3. **Leakage controls.** How are you handling pre-trade price, rolling windows, wallet aggregates?
4. **Model selection.** What models did you sweep? What's your best AUC?
5. **Top-k precision.** What does your top-1% by p_hat look like?
6. **Consensus alignment.** Do your top picks also concentrate in NO-resolved markets at low YES prob?
7. **Backtest economics.** Did you compute ROI? Under what realism assumptions?
8. **Headline finding.** Where do we agree, where do we differ?

---

## File map (for code-level questions)

```
alex/scripts/
  build_cohorts.py             # Produces train.parquet, test.parquet
  06b_engineer_features.py     # v3.5 70-feature engineer
  07_sweep.py                  # 7-model supervised sweep + IsoForest
  10_backtest.py               # Flat-stake economic backtest (patched)
  11_realistic_backtest.py     # Capital-aware execution (patched)
  12_sensitivity_sweep.py      # cost_floor × copycat-N grid
  pressure_tests/
    phase1_quick_wins.py       # 9 leakage/integrity checks
    phase2_falsification.py    # B1a/B1b consensus tests + A1 SELL semantics

alex/notes/
  design-decisions.md          # 33 decisions, append-only
  pressure-test.md             # Pressure-test register with verdicts
  alex-approach.md             # THIS file

alex/outputs/backtest/
  summary.json                 # 10_backtest output
  realistic/sensitivity.csv    # 11_realistic bankroll × bet sweep
  sensitivity/sweep.csv        # 12_sensitivity cost × copycats sweep
```
