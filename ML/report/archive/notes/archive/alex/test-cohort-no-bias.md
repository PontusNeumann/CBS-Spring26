# Test cohort — all NO, YES hedge deferred

*2026-04-22. Context: expanded test cohort from 1 ceasefire-extended market (692 trades) to all 7 ceasefire markets (13,414 trades). All 7 resolve NO. This note captures why we didn't add a strike-YES hedge despite the initial NO-bias concern.*

## Concern re-evaluation (2026-04-22 update)

The earlier "all-NO test is a problem" framing was overstated. The target `bet_correct` is a **per-trade** label that is approximately 50/50 within every market regardless of its final resolution:

- NO market: trades on the NO side have `bet_correct = 1`, trades on the YES side have `bet_correct = 0` → ~balanced.
- YES market: trades on the YES side have `bet_correct = 1`, trades on the NO side have `bet_correct = 0` → ~balanced.

The model isn't trying to predict "does this market resolve NO"; it's predicting "will THIS trade be on the winning side." That's a per-trade task with balanced targets within every market, regardless of resolution. Training on 3 NO + 1 YES markets and testing on 7 NO markets is fine as long as the features the model uses are genuinely market-agnostic (enforced by P0-8 drops of market-identifying features).

**Residual concerns (still worth monitoring but not blocking):**

1. **Residual market-identity leakage** — a feature that still correlates with market resolution could produce a "predict NO" shortcut. P0-8 drops `time_to_settlement_s`, `market_volume_so_far_usd`, etc. and the `size_x_time_to_settlement` interaction feature. Verify no leakage remains via feature-importance inspection on the baseline.
2. **Isotonic calibration asymmetry** — fitting calibration on NO-resolution val trades may not transfer perfectly to hypothetical YES-resolution deployments. Minor reporting concern.
3. **RQ2 feature-importance validation** — for the informed-flow framing, having some YES markets in test would let us check whether the same features drive predictions on both outcome types. Not blocking but useful.

## Original concern (kept for posterity)

If every test market resolves NO, the model could score well by accidentally learning "predict NO-side wins" rather than by learning trade-correctness in general. With all-NO test data, we can't distinguish the two.

## Why we're NOT hedging with a strike YES market yet

Strike YES markets available for test (Mar 1 – Mar 31, Dec 31 2026, June 30 2026) are all within the **strike event family** — same event (114242) as the training cohort (Feb 25-28).

Adding one as a test cohort introduces a price-arbitrage leakage channel. Within the strike series, prices are mechanically linked:

$$P(\text{strike by March 15}) \geq P(\text{strike by February 28})$$

(because anything that happens by Feb 28 also happens by March 15). A wallet trading simultaneously in Feb 28 and March 15 carries this arbitrage context in their `wallet_prior_*` features, their `wallet_market_category_entropy`, and their behavioural signatures (directional_purity, spread_ratio, etc.). Testing on a held-out March strike partially leaks Feb 28 training information via:

1. **Cross-market wallet behavioural features** — wallets that were informed on Feb 28 have characteristic feature signatures that persist into their March 15 trades.
2. **Implicit event-family signal** — `time_to_settlement_s` was dropped, but residual event-family correlation in volumes, wallet activity concentration, and sibling-price implications remains.

Ceasefire markets are a different event family entirely (different resolution mechanism, different deadline dynamics, different traders in practice). No arbitrage channel to strike markets. Cleaner cross-family test.

## The tradeoff accepted

- **Lose:** ability to falsify "model predicts NO-bias" on this test.
- **Gain:** clean cross-event-family generalisation claim without covert leakage.

## How to add a YES hedge later (deferred)

Options for a YES test cohort that doesn't inherit strike-series leakage:

1. **Extend dataset to Maduro / Biden-pardons clusters** (plan §8 deferred work). Different events, different traders, some resolve YES. Requires re-running the full pipeline on new condition_ids. High effort.
2. **Synthetic YES analogue** — flip the sign on a NO ceasefire market's trades (BUY YES ↔ BUY NO) and re-derive `bet_correct`. Dangerous — introduces its own artefacts and isn't really a YES test.
3. **Simulation-based sanity check** — measure NO-bias in model predictions on val (Jan 23, also NO) and use that as a calibration anchor. Val is NO, so if the model's p_hat distribution on val is significantly below 0.5, the NO-bias concern is real; if centred at ~0.5, less concerning.
4. **Ablation within training** — train an intentionally biased control model (e.g. only Feb 25-27 which are all NO) and measure how it scores on the all-NO test vs how the full train model scores. Delta tells us how much the YES training signal (Feb 28) actually contributes.

My preferred: **option 3 first** (cheap, quantifies the concern without new data), then **option 4** if concerning (1 more training run).

## Trigger conditions for adding YES hedge

- [ ] Model's `p_hat` distribution on val is materially below 0.5 (e.g. mean < 0.45).
- [ ] Feature importance shows wallet-in-NO-market behavioural features dominating.
- [ ] Reviewer / teammate explicitly asks for a YES test.

## Decision log

- **2026-04-22:** Test set = 7 ceasefire markets (13,414 trades, all NO). Strike-YES hedge deferred due to price-arbitrage leakage concern. Will add option-3 sanity check to results reporting. Revisit if trigger conditions fire.
