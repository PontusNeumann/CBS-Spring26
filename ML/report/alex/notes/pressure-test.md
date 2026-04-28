# Pressure-test register

*Living document. Status tags: UNTESTED / SUSPECT / FALSIFIED / VERIFIED.*

The 45× ROI / 100% top-1% precision result is too clean. Cataloguing what we did and what we assumed, then ranking each assumption by how badly it could be wrong.

---

## ⚠️ HEADLINE FINDINGS (2026-04-28, fully updated)

**1. The 45× ROI was inflated ~400× by a cost-calculation bug.** Pre-trade price in the features parquet was the previous trade's *per-token* price (could be either YES or NO token), not YES probability. The cost calc in `10_backtest.py` and `11_realistic_backtest.py` treated it as YES probability — wrong on ~49% of trades. Both scripts now patched.

**2. Phase 2 re-run (post-fix) FALSIFIED the asymmetric-info claim:**
- B1a: RF and HistGBM top-1% picks are **100% consensus-aligned** (LogReg 79%). Picks concentrate in 4-5 NO-resolved markets at ~2-4% YES probability.
- B1b: Naive 3-line "score = consensus_strength × is-with-consensus" rule achieves **AUC 0.844, top-1% precision 100%** — matches RF/HistGBM. Models add only ~0.04 AUC.

**3. Sensitivity sweep:** Cost floor doesn't matter in [0.001, 0.10]. Copycat-N is the dominant ROI lever. ML report headlines N=1 (clean ML metric); deployment N-sensitivity disclosed.

**Locked headline numbers (cost_floor=0.05, $10K bankroll, 5% bet, N=1):**
- HistGBM general_ev: **+23.7%** ROI, n=1,338 executed
- HistGBM top5pct_edge: **+27.3%** ROI, n=601 executed
- RF general_ev: +13.4% ROI, n=1,152 executed
- Top-1% by p_hat precision: 100% (RF), 99.9% (HistGBM) — UNCHANGED by cost-fix

**Corrected economic interpretation:** The model identifies winning trades with high precision, but does so by detecting consensus-aligned heavy-favorite bets in markets where pricing is already correct, not by isolating asymmetric-information flow. The Magamyman-style insider pattern is not detected (IsoForest null + home-run zero hits + B1b naive baseline matches model).

See bottom of this doc for full pressure-test results (Phase 1 + Phase 2 + cost-fix + Phase 3 sensitivity).

---

## What we built (concrete pipeline)

1. **Cohort identification.** Filtered HF `SII-WANGZJ/Polymarket_data` markets by canonical regex on the `question` field. 75 markets matched (65 train, 10 test).
2. **Trade extraction.** Streamed `trades.parquet` from HF, kept rows where `market_id ∈ cohort` AND `timestamp < event_time`. 1.4M trades pre-filter.
3. **Liquidity floor.** Dropped markets with <500 pre-event trades. 1 strike market dropped → 63 train markets.
4. **Target derivation.** For each trade: `bet_correct = 1` if BUY of winning_token OR SELL of losing_token. Winning token derived from `outcome_prices` JSON (when available) or fallback to deadline + real-world event timing.
5. **Feature engineering.** 70 features in v3.5. Rolling windows use `closed='left'` to exclude current trade. Multi-key wallet aggregates use cumcount + shift(1).
6. **CV + calibration.** 5-fold GroupKFold on `market_id`. Isotonic regression fit on out-of-fold predictions. Final model fit on full train.
7. **Test scoring.** Predict on test, apply isotonic calibration, compute strategy masks (general_ev, home_run, top_1pct_by_p_hat, top_1pct_by_edge, etc.).
8. **PnL computation.** `cost = pre_trade_price` for trader's chosen side; `pnl = stake × (1−cost)/cost` if win, `−stake` if lose. Cost floored at 0.001 in v1, 0.05 in v2.
9. **Realistic backtest.** Capital tracking, per-trade cap (5% of capital, 10% of original USD × 0.1 copycat scaler), 20% concentration cap, $0.50 gas, slippage surcharge.

---

## Assumptions, ranked by suspicion

### A. Target derivation — HIGHEST SUSPICION

#### A1. `bet_correct` formula is correct
**Status:** SUSPECT
**Assumption:** `bet_correct = 1` iff `(BUY ∧ side == winning) OR (SELL ∧ side ≠ winning)`.
**Why suspect:** Polymarket SELLs aren't simple shorts. A SELL of YES at price P could be:
- A trader closing a prior BUY position (already exposed to direction; sell crystallises P&L)
- A market maker providing liquidity (no directional bet)
- An open short equivalent (bet against YES)
Our formula treats all SELLs as "betting opposite" — but a closing-sale isn't really a fresh directional bet.
**Test:** Sample 100 SELL-YES trades, check whether the same wallet had prior BUY-YES in the same market. If most are closing trades, our `bet_correct` is wrong for them.
**Impact if wrong:** A non-trivial fraction of trades have miscoded targets. Downstream metrics inflated/deflated unpredictably.

#### A2. Fallback resolution times for unresolved markets
**Status:** SUSPECT
**Assumption:** Test markets with deadline ≥ Apr 7 → resolve YES. Markets with deadline < Apr 7 → resolve NO. Resolution time is `min(deadline, event_time)`.
**Why suspect:** 6 of 10 test markets are unresolved at HF snapshot. We INFERRED their outcomes from real-world event timing (Trump's Apr 7 ceasefire announcement). If any market has a different resolution criterion (e.g., "ceasefire must hold for 7 days" vs "announcement only"), our inference is wrong.
**Test:** Spot-check the 6 unresolved markets' market.json metadata on Polymarket directly (one-off lookup, not part of HF). Verify resolution criteria match our assumption.
**Impact if wrong:** Bet_correct flipped for entire markets → AUC numbers are upper bounds at best, garbage at worst.

#### A3. token1 = YES, token2 = NO across all markets
**Status:** SUSPECT
**Assumption:** `answer1 == "Yes"` for all our markets. We use `nonusdc_side == "token1"` to mean "trader bought/sold YES token."
**Why suspect:** Some Polymarket markets have answer1=No, answer2=Yes (negative-framed questions). If any of our 75 markets are flipped, our `outcome_yes` flag is wrong for them.
**Test:** Verify `markets["answer1"] == "Yes"` for all 75 markets.
**Impact if wrong:** Direction features all flipped for affected markets.

### B. Generalisation / transfer claim — HIGH SUSPICION

#### B1. The 100% top-k precision is "asymmetric information detection"
**Status:** SUSPECT (likely FALSE per current evidence)
**Assumption:** Model identifies trades made by informed traders, not just trades on the consensus side.
**Why suspect:** Top-1% picks are concentrated in 4 NO-resolved markets. Picks are 50% BUY NO + 50% SELL YES — both "betting on the dominant side." Pre-event YES prices in these markets were already low (consensus said NO would win). The model is finding "consensus-correct trades in markets where consensus is correct" — that's market efficiency, not asymmetric information.
**Test:** 
- Run a naive baseline: "for each trade in test, predict win iff trade is on the dominant pre-event side." Compare hit rate to model.
- Decompose top-1% picks by: was the trader betting WITH or AGAINST market consensus at time of trade? If 100% with-consensus, model isn't detecting informed flow.
**Impact if wrong:** The headline "asymmetric information detection" claim is unsupported. Strategy reduces to "low-edge mirror trading the consensus."

#### B2. Train→test transfer is genuine
**Status:** SUSPECT
**Assumption:** Model trained on strike ladder transfers to ceasefire ladder. Pattern is regime-agnostic.
**Why suspect:** Both cohorts have the SAME structural pattern: "deadline-based binary markets where the underlying event eventually happens." The model learning "low pre-trade-price + buy YES = win when underlying event occurs" works in both cohorts because the structural pattern is identical. This isn't transfer of a "model of informed flow"; it's transfer of "structural pattern of binary markets."
**Test:** Run the same model on a structurally different test cohort: e.g. continuous-outcome markets, or markets where the underlying event is a presidential election (not deadline-binary). If transfer fails, claim is structural-pattern-specific not general informed-flow.
**Impact if wrong:** Generalisation claim narrowed substantially.

### C. Cohort design — MEDIUM SUSPICION

#### C1. The pre-event filter is correctly applied
**Status:** UNTESTED but likely VERIFIED
**Assumption:** All trades in train have `timestamp < 2026-02-28 06:35 UTC`. All test trades have `timestamp < 2026-04-07 23:59 UTC`.
**Test:** `train.timestamp.max()` and `test.timestamp.max()` vs cutoffs. Quick check.
**Impact if wrong:** Direct contamination — model sees post-event arb trades and "predicts" trivially.

#### C2. The 4 dominant test markets resolved NO as we assumed
**Status:** UNTESTED
**Assumption:** Markets 1466012/13/14/15 resolved NO based on `outcome_prices = ['0', '1']` or similar.
**Test:** Verify resolutions independently (Polymarket API or news search for "ceasefire announced March 2 / March 6 / etc."). The hard one is March 31 (1466015) which has `outcome_prices = ['0.0035', '0.9965']` — close to but not exactly resolved.
**Impact if wrong:** If any of these 4 actually resolved YES, our top-1% precision drops dramatically.

#### C3. Liquidity-floor exclusion didn't introduce selection bias
**Status:** VERIFIED (only 1 market dropped)
**Why low risk:** Single market with 0 volume; excluded for sanity not selection.

### D. Feature engineering / leakage — MEDIUM SUSPICION

#### D1. Pre-trade price doesn't leak future information
**Status:** UNTESTED
**Assumption:** `pre_trade_price = price.shift(1) within market_id` strictly excludes current trade.
**Test:** For random 100 trades, verify `pre_trade_price[i] == price[i-1]` within the same market.
**Impact if wrong:** Model has access to current trade's own price → trivial.

#### D2. Rolling features use closed='left' uniformly
**Status:** UNTESTED
**Assumption:** All `.rolling(...)` calls in `06b_engineer_features.py` use `closed='left'` to exclude the current row.
**Test:** Grep `06b_engineer_features.py` for all `.rolling(` and verify each has `closed='left'` or equivalent shift.
**Impact if wrong:** Subtle leakage of current trade's contribution to rolling stats.

#### D3. Wallet position cumsum correctly excludes current trade
**Status:** UNTESTED
**Assumption:** `taker_position_size_before_trade` is computed via `cumsum().shift(1)` per (market, taker), with the head-of-group shift artefact zeroed out.
**Test:** Spot-check a few wallets — manually compute their position before trade N, compare to feature value.
**Impact if wrong:** Position feature leaks current trade.

#### D4. Standardisation done within fold (no train+test mixing)
**Status:** UNTESTED
**Assumption:** StandardScaler.fit on train fold only, then transform val and test.
**Test:** Read `_backtest_worker.py` and `07_sweep.py` to verify scaler is refit per fold.
**Impact if wrong:** Test statistics leak into train via shared scaling.

### E. Backtest / economic — MEDIUM SUSPICION

#### E1. `cost = pre_trade_price` reflects actual execution cost
**Status:** SUSPECT
**Assumption:** Mirroring a trade at the SAME pre_trade_price is achievable.
**Why suspect:** When original trade hit the orderbook, it consumed liquidity at that price. The next trade — ours — fills at a worse price. Especially at extreme prices (cost < 0.10), this matters most.
**Test:** Compute average price impact in our trade history: per market, std of price changes around trades. Compare to the cost differences we're modelling.
**Impact if wrong:** Realistic ROI overstated; corrections compound with cost floor.

#### E2. 10× copycats is appropriate
**Status:** UNTESTED (assumption picked out of thin air)
**Assumption:** 10 simultaneous mirrors compete for the original trade's liquidity.
**Test:** No internal data validates this. Could vary N ∈ {1, 5, 10, 25, 50, 100} and report sensitivity.
**Impact if wrong:** ROI scales linearly with 1/N; could be 4× or 100× depending on assumed competition.

#### E3. Position release at market resolution is correct
**Status:** SUSPECT
**Assumption:** Capital tied up in a position is returned at market_resolution_time. Until then it's locked.
**Why suspect:** On Polymarket, you can sell positions before resolution (at current market price). Realistic traders close positions when edge disappears, not at resolution.
**Test:** Re-run with positions released after a fixed holding period (e.g., 1h) instead of waiting for resolution. PnL changes (slippage on close), but capital frees up faster.
**Impact if wrong:** Liquidity-bound numbers shift — could go either way.

#### E4. Cost floor at 0.05 is the right realism number
**Status:** UNTESTED (judgement call)
**Test:** Sweep cost floor ∈ {0.001, 0.01, 0.05, 0.10, 0.20}. Report ROI at each. Show how sensitive the headline is.

### F. Data integrity — LOW SUSPICION

#### F1. HF dataset is complete for our markets
**Status:** UNTESTED
**Assumption:** All trades that ever happened in our 75 markets are in the HF dataset.
**Test:** Compare market-level `volume` from `markets_subset.parquet` to `sum(usd_amount)` from our extracted trades. Mismatch = missing trades.
**Impact if wrong:** Missing high-volume trades = biased feature distributions.

#### F2. No duplicate trades in our extraction
**Status:** UNTESTED
**Test:** Check `transaction_hash` uniqueness in our train/test parquets.
**Impact if wrong:** Duplicates inflate statistics.

---

## Top 5 most-likely culprits if numbers are too good

Ranked:

1. **B1 (asymmetric-info claim).** Model likely picks consensus-correct trades, not informed-flow trades. The 100% top-k precision lives in 4 NO-resolved markets where everyone (market + model) correctly predicted NO. Strongest single threat to the headline.
2. **A2 (resolution times for unresolved markets).** 6 of 10 test markets used inferred resolution. If any are wrong, hit rate is wildly off.
3. **A1 (SELL semantics).** SELLs in our data may often be position closures, not directional bets. Treating them as "bets on opposite side" introduces noise/bias of unknown sign.
4. **D1-D4 (leakage).** Even small subtle leaks (current price encoded somewhere we missed) would explain the bimodal AUC.
5. **E1 (execution cost realism).** Even with cost floor, mirroring at the original trade price is optimistic.

---

## Recommended pressure-test program

Order of execution, ~3 hours total:

**Hour 1 — quick wins:**
- C1: verify pre-event filter (5 min)
- D1: verify pre_trade_price = price.shift(1) (10 min)
- D2: grep rolling closed='left' (5 min)
- F2: dedupe check on transaction_hash (5 min)
- A3: verify answer1==Yes for all markets (5 min)
- C2: verify resolution outcomes for top-4 markets (15 min)

**Hour 2 — claim falsification:**
- B1: contrarian-vs-consensus decomposition of top-1% picks (30 min)
- B1: naive "buy dominant side at low price" baseline comparison (30 min)

**Hour 3 — sensitivity:**
- E2 + E4: copycat-N sweep + cost-floor sweep (1 hour)

If we survive all this with 5×+ ROI, we can defend the headline. If any of A1/A2/B1 falsifies → retreat to a narrower claim.

---

## Pressure-test results (2026-04-28)

### Phase 1 — quick wins (NO FATAL FAILURES)

| Test | Status | Detail |
|---|---|---|
| C1  pre-event filter | ✓ PASS | train.max < strike cutoff; test.max < ceasefire cutoff |
| D1  pre_trade_price = price.shift(1) | ✓ PASS | 100% match across 257K trades |
| D2  rolling closed='left' | ✓ PASS | 12 of 12 calls correctly use it |
| D3  wallet position cumsum | ✓ PASS | 100% match on full-dataset recompute |
| D4  scaler refit per fold | ✓ PASS | 9 instances in active pipeline, all train-only fit |
| F2  duplicate transaction_hash | ℹ NOTE | 1.5% true duplicates in test (after dedup on (tx_hash, log_index)). Not material. |
| A3  answer1 == "Yes" | ✓ PASS | 75 of 75 markets |
| C2  4 dominant test markets resolved NO | ✓ PASS | All 4 confirmed via outcome_prices |
| F1  HF coverage vs market.volume | ℹ NOTE | Median ratio 0.44 — HF captures ~half of reported volume. External-validity caveat, not internal claim. |

### Phase 2 — claim falsification (re-run 2026-04-28 with corrected pre_yes_price)

| Test | Status | Detail |
|---|---|---|
| B1a consensus-vs-contrarian decomposition | ⚠ FAIL-IN-SPIRIT | RF top-1% picks: **100% with-consensus, 0% against** (5 markets, top market 41% of picks). HistGBM: **100% with-consensus** (4 markets, top market 68% of picks). LogReg: 79% with-consensus (the 21% against-consensus picks have 0.2% hit rate). Mean YES probability in picks: 2-4%. Picks are pure consensus detection. The script's PASS verdict (threshold >90% all-models) only triggered because LogReg was at 79%, but for the headline models the result is the failure mode B1a was designed to detect. |
| B1b naive baseline | ✗ FAIL | A 3-line "score = consensus_strength × is-with-consensus" naive rule achieves AUC 0.844 and **top-1% precision 100%** — matches RF's 100% top-1%. RF AUC 0.888, HistGBM 0.887: model adds only +0.04 AUC over naive. Top-1% precision is fully reproducible by naive consensus alignment. |
| A1  SELL semantics (closing vs open-short) | ℹ NOTE | 36% of all SELLs are position-closing trades. But in top-1% picks, only 0.2% of SELLs are closing — almost all are fresh open-shorts. Treating SELLs as fresh directional bets is OK at the model's confident edge. |

### Cost-calculation bug — discovery and fix

**Verification step (per-token price test):** For each market, computed mean price by `nonusdc_side`. Across all markets, `token1_mean + token2_mean ≈ 1.0` (range 0.98-1.03). This proves prices are per-token, not YES-normalized.

**Fix implemented in `11_realistic_backtest.py`** via `compute_pre_yes_price_corrected()`:
```python
df["_t1"] = np.where(df["nonusdc_side"]=="token1", df["price"], np.nan)
df["_t2"] = np.where(df["nonusdc_side"]=="token2", df["price"], np.nan)
df["_last_t1"] = df.groupby("market_id")["_t1"].ffill().shift(1)
df["_last_t2"] = df.groupby("market_id")["_t2"].ffill().shift(1)
# At market boundaries shift(1) leaks — reset first row of each market
first_idx = df.groupby("market_id").head(1).index
df.loc[first_idx, ["_last_t1", "_last_t2"]] = np.nan
pre_yes = df["_last_t1"].fillna(1 - df["_last_t2"]).fillna(0.5)
```

**Impact (top-k precision after correction):**

| Model | top-1% by p_hat (unchanged) | top-1% by edge OLD | top-1% by edge NEW | Δ |
|---|---|---|---|---|
| LogReg L2 | 0.790 | 0.982 | 0.004 | -0.978 |
| Random Forest | 1.000 | 1.000 | 0.927 | -0.073 |
| HistGBM | 0.999 | 1.000 | 0.545 | -0.455 |

**Realistic-backtest ROI ($10K bankroll, 10× copycats, cost floor 0.05):**

| Strategy | HistGBM old → new | LogReg old → new | RF old → new |
|---|---|---|---|
| general_ev | 40.5× → **0.14×** (+14%) | 39.0× → 0.06× (+6%) | 38.2× → 0.07× (+7%) |
| top5pct_edge | 43.6× → **0.11×** (+11%) | 35.3× → -0.04× | 45.5× → -0.01× |
| top1pct_phat | 8.7× → 0.01× | 1.96× → 0.00× | 7.2× → 0.00× |
| top1pct_edge | 9.1× → -0.005× | 4.0× → -0.01× | 13.5× → 0.00× |
| home_run | 1.76× → 0.00× | 1.67× → -0.00× | 1.76× → 0.00× |

### Verdict per pressure-test target

| Target | Status |
|---|---|
| A1  SELL semantics | NOTE — minor caveat, top-k picks unaffected |
| A2  Resolution times for unresolved test markets | NOT YET TESTED — but the 4 dominant resolved markets (C2) are all confirmed NO |
| A3  token1 = YES across markets | VERIFIED |
| B1  100% top-k is "informed flow detection" | **FALSIFIED** — RF/HistGBM 100% with-consensus; naive consensus rule matches top-1% (100%) and AUC within 0.04. Model is high-precision consensus detector, not asymmetric-info detector. |
| B2  Train→test transfer is genuine | UNTESTED in alternative cohort, but B1 falsification narrows the claim |
| C1  Pre-event filter | VERIFIED |
| C2  Dominant markets resolved as assumed | VERIFIED |
| C3  Liquidity floor selection bias | LOW RISK (1 market dropped) |
| D1  pre_trade_price doesn't leak future | VERIFIED (shift(1) correct) |
| D2  Rolling closed='left' uniform | VERIFIED |
| D3  Wallet cumsum excludes current trade | VERIFIED |
| D4  Standardisation per-fold | VERIFIED in active pipeline |
| E1  cost = pre_trade_price reflects execution | **FALSIFIED** — bug discovered; cost was wrong on ~49% of trades. Now uses pre_yes_price_corrected (10_backtest.py + 11_realistic_backtest.py). |
| E2  10× copycats appropriate | TESTED — N is the dominant ROI lever. HistGBM general_ev: +24% at N=1 → -7.5% at N=100. ML headline reports N=1; deployment limitation disclosed (D-033). |
| E3  Position release at resolution | NOT TESTED |
| E4  Cost floor 0.05 is realistic | TESTED — ROI insensitive to cost_floor in [0.001, 0.10] (almost no trade has cost in that range). Floor=0.20 inflates ROI by admitting more trades and capping losses. 0.05 is a safe headline floor. |
| F1  HF dataset complete | NOTE — ~44% volume coverage |
| F2  No duplicates | NOTE — 1.5% duplicates in test |

### Honest revised headline (post-fix, post-Phase-2-rerun, post-sensitivity)

> HistGBM classifier identifies trades likely to win on the held-out Iran ceasefire cohort with 99.9% precision in the top 1% (2,571 trades, AUC 0.887 calibrated). RF achieves 100% top-1% precision at AUC 0.888. Decomposition shows top-1% picks are 100% consensus-aligned and concentrate in 4-5 NO-resolved markets — the model is a high-precision consensus detector, not an asymmetric-information detector. A 3-line naive rule ("score = consensus_strength × is-with-consensus") achieves the same 100% top-1% precision and AUC within 0.04. The model's marginal contribution is ~0.04 AUC over the naive baseline. Late-flow signature replicates Mitts & Ofir (≤3d hit 98%). Under N=1 execution (no concurrent mirroring), HistGBM general_ev achieves +23.7% return on a $10K bankroll. ROI is insensitive to cost floor in [0.001, 0.10] but degrades sharply with copycat-N: +13.8% at N=10, -7.5% at N=100. The Magamyman-style asymmetric-payoff insider pattern is not isolated, consistent with the IsoForest null result and home-run filter zero hits on test.

The model is real but modest. The contribution is high-precision consensus detection, not informed-flow detection. The +23.7% headline holds under N=1; the deployment-realistic number is N-dependent.

---

## Phase 3 — sensitivity sweep (2026-04-28)

Two axes from pressure-test E2/E4. Outputs at `outputs/backtest/sensitivity/sweep.csv` + 3 heatmaps.

**Cost-floor sweep:** ROI is byte-identical across floors {0.001, 0.01, 0.05, 0.10} for HistGBM general_ev — almost no trade has cost in that range. Floor=0.20 inflates ROI (e.g. 13.8% → 37.5% at N=10) by admitting more trades through the edge>0.02 filter (n_executed 2,099 → 3,705) and clipping losses. Defensible to keep 0.05 as headline floor.

**Copycat-N sweep:**

| Strategy | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| HistGBM general_ev | +23.7% | +20.3% | +13.8% | +4.6% | -1.7% | -7.5% |
| HistGBM top5pct_edge | +27.3% | +24.6% | +10.6% | +3.5% | +1.4% | +0.6% |
| RF general_ev | +13.4% | +15.3% | +7.4% | -4.9% | -5.3% | -4.5% |

RF is non-monotonic (peaks at N=5): at N=1 the per-trade volume cap is generous and trades hit the 20% per-market concentration limit faster (only 1,152 executed). At N=5 the cap shrinks, more trades fit, n_executed jumps to 1,894. After that, smaller per-trade payoffs dominate. HistGBM picks spread across more markets so concentration isn't binding at N=1.

**Headline decision (D-033):** ML report uses N=1 figures; copycat-N treated as deployment limitation.

