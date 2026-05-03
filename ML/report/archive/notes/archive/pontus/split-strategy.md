# Split-strategy analysis

## Your intuition, restated

> "The model shouldn't know instantly what the outcome of the market was
> and then figuring out on a trade what market it is connected to."

Translated into ML terms: if the feature vector identifies the market, a
classifier can memorise "market X → resolution Y" (or equivalently "wallet
cluster Z tends to win in market X"), and the test ROC becomes a lookup
table rather than a transferable signal. The subtlest version of this
leakage survives even after dropping absolute-scale features.

## What Alex did, and what it tests

| Fold | Markets | Rows | What the fold tests |
|---|---|---:|---|
| Train | Feb 25-28 strike markets (4 sub-markets) | 202,082 | — |
| Val | Mar 15 conflict-end market (1) | 13,154 | Cross-family transfer to a different event family (strikes → conflict-end) |
| Test | Apr 8-18 ceasefire markets (7, all NO) | 13,414 | Cross-family transfer further, with resolution-type shift (1 YES + 3 NO → all NO) |

Strengths
- **Chronological**: train → val → test in strict time order. No future-in-past leak.
- **Cross-family**: train is the strike event cluster; test is the ceasefire cluster. Any memorisation of a single market's resolution does not transfer.
- **P0-8 drops already neutralise the coarsest market-identity shortcut.**

Weaknesses
- **Wallet overlap.** 28,974 unique wallets traded in train. A subset of those also traded in val and/or test. The stacked run showed val ROC 0.6247 but test ROC 0.5403 on `stack_all` — a 0.08 spread largely attributable to val wallets being known to the model (via Layer 6 features) while test wallets are more novel.
- **Val is a single market (Mar 15 conflict-end).** Any idiosyncrasy of that one market (liquidity regime, news flow) dominates the early-stopping signal.
- **Test is all-NO**. `bet_correct` rate is still ~50 / 50 *within* each market (half the trades are on the losing side regardless of market resolution), but any feature that correlates with resolution-type can't be stress-tested.

## The direct empirical test of your intuition (recommend as a report exhibit)

Train a classifier with the 36-feature vector as input and
`condition_id` as the target. 74-class problem. Random baseline ≈ 1 /
74 = 1.35 %. Report top-1 and top-5 accuracy on a hold-out of trades
drawn from each market.

- If top-1 accuracy ≈ random → the feature set is genuinely
  market-agnostic. Any market-cohort split is honest.
- If top-1 accuracy is high (say ≥ 30 %) → the feature set has
  residual market-identity signal and the cross-family test ROC is
  upper-bounded by how well the model can exploit that signal.

This is **the** experiment to run before claiming the cross-family
generalisation result. Ten minutes of work, one figure for the Discussion.

## Four alternative splits, ranked

### 1. Market-cohort with wallet disjointness (the cleanest strict test)

For each cohort, drop any (wallet × market) pair whose wallet appears in
another cohort. Result: train / val / test are disjoint on both dimensions.

- **Pro**: uncontaminated generalisation claim. If test ROC > 0.5, the
  model has learnt transferable patterns.
- **Con**: the shrinkage is severe. Our group-overlap analysis shows
  that ~30 % of test wallets also traded in train markets. Dropping them
  costs ~4 k test rows (plus per-wallet effects cascade into the bootstrap
  CI).
- **Verdict**: run as a robustness check at the end. Not as the primary
  evaluation (sample size too small for PR-AUC / Brier stability).

### 2. Random 20 % within-strike (intra-family signal check)

Shuffle all Feb 25-28 strike rows; hold out 20 % as test. Every market
appears in both folds. Every wallet can appear in both folds.

- **Pro**: strong within-family signal test. If ROC here >> cross-family
  ROC, the gap tells you about transfer difficulty, not about whether
  signal exists.
- **Con**: does not answer the research question ("does the model
  generalise across event families?"). It answers a weaker question.
- **Verdict**: run as a **complement** to the main split, not a
  replacement.

### 3. Per-market temporal split (within-market structure test)

For each market, hold out the last 15 % of trades as test and the
preceding 15 % as val; train on the earliest 70 %.

- **Pro**: no market-identification shortcut because every market is
  represented in every fold. Mirrors a real-time deployment (today's
  decision depends on past trades in the same market).
- **Con**: the "last 15 %" of each market is heavily concentrated in the
  post-deadline / pre-resolution regime (20 % of rows overall). The §4
  filter (`settlement_minus_trade_sec > 0`) retains the pre-resolution
  subset but this still has near-decided outcomes; `bet_correct` is
  ~easy-to-predict by then.
- **Verdict**: run as a secondary result, with the §4 filter applied.
  Caveat: ecological to the trading rule (which fires in the final 6 h
  anyway) but conflates "can we predict hard trades early" with "can we
  classify easy trades late".

### 4. Purged K-fold with embargo (de Prado 2018, *Advances in Financial Machine Learning*)

Ten-fold time-stratified split where (a) each fold's val is a contiguous
time window, (b) training data within `L` seconds of the val window is
purged (`L` = label-horizon = the market's remaining time until
resolution), and (c) an embargo period follows each val window to stop
near-boundary information bleed. Optionally grouped by proxyWallet.

- **Pro**: the canonical rigorous approach for financial-series ML.
  Purging addresses the label-lookahead problem (our `wallet_prior_win_rate_causal` already handles the specific case of prior-trade resolutions, but purged k-fold generalises it). Embargo handles microstructure autocorrelation.
- **Con**: implementation overhead (need a fold-generator, a purge
  function, and an embargo window calibrated to the Iran cluster's
  label horizon of up to ~60 days). The course curriculum does not
  specifically cover purged CV.
- **Verdict**: fits the rigorous-methodology angle of the report but is
  not strictly necessary. Mention in the Discussion as "rigorous
  extension we did not implement," cite de Prado.

## Recommended set-up for the final submission

Present **one primary split and two robustness checks** in the Results
section, plus the market-identity-audit experiment:

1. **Primary (Results § Evaluation)**: keep Alex's market-cohort split —
   train on Feb 25-28 strikes, val on Mar 15 conflict-end, test on the 7
   April ceasefires. This is what answers RQ1 most directly (cross-family
   generalisation). Report test ROC with wallet-level bootstrap CI.

2. **Robustness 1 (Results § Generalisation check)**: random 20 % of the
   Feb 25-28 strike trades held out of train. Reports "within-family
   signal exists." If the within-family ROC is ~0.70 and the cross-family
   ROC is ~0.58, the gap is the informed-flow narrative: patterns transfer
   but get noisier.

3. **Robustness 2 (Results § Wallet-novelty check)**: same market-cohort
   split, but report test ROC twice — once on the subset of test rows
   whose `proxyWallet` also appears in train, once on the wallet-novel
   subset. The wallet-novel number is the cleanest generalisation claim.

4. **Market-identity audit (Discussion or Appendix)**: train a 74-class
   classifier on the 36-feature vector targeting `condition_id`. Report
   top-1 accuracy vs. random baseline (1.35 %). If < 5 %, the feature
   set is market-agnostic and the cross-family results are credible. If
   > 20 %, flag it as a limitation and argue around it.

This combination satisfies the CBS ML rubric (test / val set discipline,
cross-validation within training, reported with uncertainty), matches
Alex's P0-6 concern precisely, and responds directly to your intuition
that "the model shouldn't be able to figure out what market it's in."

## Implementation effort

| Item | Code change | Runtime |
|---|---|---|
| 1. Primary market-cohort split | Already implemented in `14_build_experiment_splits.py` | — |
| 2. Random 20 % within-strike | ~30 lines in a new `14b_build_withinstrike_split.py` | 5 min |
| 3. Wallet-novelty metric on existing test | ~15 lines in an analysis script that reads `predictions_test.parquet` | 2 min |
| 4. 74-class market-identity audit | ~50 lines; `RandomForestClassifier(condition_id)` with GroupKFold | 10 min |

All four land in **≈ 20 minutes of code + 15 minutes of CPU**. Worth
doing before the final Results write-up.

## On v7labs' guide

The [v7labs train/val/test primer](https://www.v7labs.com/blog/train-validation-test-set)
covers the basic taxonomy (random, stratified, k-fold, stratified k-fold)
and the 80/10/10 default, but it does not address shared-entity
contamination — the failure mode that matters most for this dataset.
Their article is a reasonable intro citation for the Methodology section
on cross-validation, but the rigorous split discussion should anchor on:

- **Wolpert (1992)** for stacked generalization.
- **de Prado (2018)** for purged k-fold with embargo.
- **Scikit-learn's `GroupKFold`** documentation for grouped cross-validation.

### Sources consulted
- [Train, Validation, Test Split — V7 Labs](https://www.v7labs.com/blog/train-validation-test-set)
- *Advances in Financial Machine Learning* (Marcos López de Prado, Wiley 2018), chapter 7 on cross-validation for financial data.
- Alex's parallel observation on wallet overlap — see `alex/notes/session-learnings-2026-04-22.md` P0-6.
