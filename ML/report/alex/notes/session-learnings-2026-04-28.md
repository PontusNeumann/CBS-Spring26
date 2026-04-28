# Session learnings — 2026-04-28

*Distilled from a full-day session: cohort pivots, v3.5 feature engineering, sweep, realistic backtest, then a thorough pressure test that discovered a cost-calc bug and collapsed the 45× ROI headline to +14%. Save future sessions from re-deriving these.*

## Hard constraints (never forget)

- TF/Keras only for neural-network code. PyTorch is forbidden by exam rules.
- Scope is `/Users/alex/Documents/Claude Projects/CBS-Spring26/ML/report/alex/` only. Don't touch `ML/report/scripts/` (Pontus's territory).
- No external wallet enrichment from Polymarket API or on-chain queries — HuggingFace dataset only.
- Within-HF wallet aggregates ARE allowed (cumcount + shift on `maker`/`taker` columns).
- Working venv at `ML/report/.venv` already has TF/Keras, sklearn, pandas, polars-compatible deps.

## Top 5 lessons that almost cost us the thesis

### 1. Verify schema assumptions at feature-engineering time, not at submission time

The `price` field in HF schema is per-token (each token has its own price), NOT YES-normalized. We assumed pre_trade_price = YES probability for ~3 weeks before catching it during pressure test. Cost was wrong on ~49% of trades, inflating reported ROI by ~400×.

**Habit to form:** before treating ANY field as a derived semantic value (e.g. "this is YES probability"), verify by computing per-category statistics. For our case: `mean(price | nonusdc_side)` per market — the test that would have caught it was a 5-line groupby.

### 2. AUC ≠ economic value

100% top-k precision is real and survives the audit. But 100% precision on heavy-favorite bets at cost 0.95 makes ~5% per win, not 19× per win. The 100% precision number can be true while the ROI number is junk. Always backtest with realistic execution math BEFORE celebrating model metrics.

### 3. Pressure-test before reporting headline numbers

The 45× ROI was reported externally in a group update before pressure testing. By the time we caught the bug, we were already invested in the framing. Better discipline: any "too-good-to-be-true" number triggers a mandatory ablation pass — drop suspect features, run a naive baseline, compare to random selection. The 5 most-likely-culprits framework in pressure-test.md is reusable.

### 4. Bimodal per-market AUC is a finding, not a bug

Trees scoring AUC 1.0 in some markets and 0.0 in others doesn't mean leakage. It means the model has learned a single transferable rule that's correct in markets where the underlying pattern holds and inverted where it doesn't. Three RF ablations (drop kyle_lambda, drop 24h-window features, drop suspect family) confirmed no single-feature leak — the bimodality is structural.

### 5. Spot-check tests can lie if row alignment is off

D3 (wallet position cumsum) initially "failed" because my sample-based test had a row-alignment bug between raw and feature parquets. Full-dataset recompute matched 100%. **Lesson:** when verifying derived features, always do a full-dataset comparison, not just a sample. Indexing across two different parquets is fragile.

## Specific project state at session end

### What works

- **Pipeline integrity:** all of pre-event filter, pre_trade_price lag, rolling closed='left', wallet position cumsum, and per-fold scaler refit are correct.
- **Model AUC:** RF 0.899, HistGBM 0.893, LogReg 0.629 (calibrated, on test). These are real.
- **Top-1% by p_hat precision:** 100% on tree models (n=2,571 trades). Real.
- **Late-flow signature:** ≤14d hit 92%, ≤3d hit 98%. Confirmed Mitts & Ofir pattern.
- **Beats baselines:** random 1% 51%, follow-consensus loses money. Real.

### What's broken or limited

- `pre_trade_price` in v3.5 features is per-token price, not YES probability. **The model TRAINED on this** — we don't know how the model interpreted it. Fix is in `11_realistic_backtest.py::compute_pre_yes_price_corrected()`.
- `10_backtest.py` still uses the buggy cost calc. Has not been patched. Cached predictions in `.scratch/backtest/preds_*.npz` are correct (model output unchanged).
- Magamyman-style asymmetric-payoff insider pattern is NOT detected by the v3.5 feature set. IsoForest unsupervised arm gave a null result (anomaly score uncorrelated with bet_correct, corr -0.007).
- Realistic ROI on $10K bankroll: +14% (HistGBM general_ev), NOT 45×. ~290× reduction after cost-fix + realism corrections.

### Three honest framings of the thesis

1. **Narrow defensible:** "Random Forest classifier identifies winning trades on the held-out Iran ceasefire cohort with 100% top-1% precision (AUC 0.899). Mirror strategy generates +14% return on $10K bankroll under realistic execution constraints. Late-flow signature confirms Mitts & Ofir's documented late-concentrated informed flow pattern."
2. **Honest negative:** "Despite identifying winning trades correctly, the picks concentrate in heavy-favorite consensus-correct bets with small payoffs. Magamyman-style asymmetric-payoff insider trades are NOT isolated by feature set without wallet identity enrichment."
3. **Open future:** "Adding `taker_resolved_winrate_history` (track-record-based wallet feature) would directly test 'is this trader a known winner', which the current feature set proxies imperfectly."

## What's open if you want to keep working

In rough priority:

1. **Apply the same cost fix to `10_backtest.py`** — currently still buggy. ~10 min. Re-run all backtest outputs.
2. **Re-run Phase 2 (B1a, B1b) with corrected pre_yes_price** — gives a clean answer on consensus-vs-contrarian decomposition. ~30 min.
3. **`taker_resolved_winrate_history` feature** — for each trade, fraction of this taker's prior resolved trades that won. Requires resolution-time mapping per market and per-taker O(n²) compute. ~2 hours.
4. **Phase 3 sensitivity sweeps** (cost floor, copycat-N) — less important now since headline is modest. 1 hour.
5. **Add Maduro-removal markets to train** — diversifies resolution-direction examples; might compress bimodal per-market AUC. 2 hours.
6. **Write up thesis report** — narrow defensible framing is ready to ship.

## Pipeline file map (current state)

```
alex/scripts/
  build_cohorts.py                — produces train.parquet, test.parquet
  06b_engineer_features.py        — v3.5 70-feature engineer (BUG: pre_trade_price is per-token)
  06_baseline_idea1.py            — v1/v2 LogReg baseline (deprecated)
  07_sweep.py                     — supervised sweep + IsoForest
  08_leakage_diagnostic.py        — RF ablation tests
  09_iso_forest.py                — standalone IsoForest run (null result)
  10_backtest.py                  — backtest (BUGGY — uses pre_trade_price as YES prob)
  11_realistic_backtest.py        — capital-constrained backtest (PATCHED, uses corrected pre_yes_price)
  _backtest_worker.py             — parallel worker for sweep
  dashboard.py                    — live HTTP dashboard, port 9876
  pressure_tests/
    phase1_quick_wins.py          — 9 verifications, all PASS or NOTE
    phase2_falsification.py       — 3 high-stakes tests, B1a/B1b inconclusive due to cost bug
    fix_cost_and_rerun.py         — diagnostic that proved the bug + measured impact

alex/notes/
  design-decisions.md             — 29 entries, append-only log
  pressure-test.md                — full register with verdicts
  feature-exclusion-list.md       — pre-existing ref
  sister-market-features.md       — pre-existing ref
  test-cohort-no-bias.md          — pre-existing ref
  session-learnings-2026-04-22.md — prior session
  session-learnings-2026-04-28.md — THIS file
  new-chat-prompt.md              — kickoff template

alex/data/
  markets_subset.parquet          — 75 markets tagged train/test
  train.parquet                   — 1.11M raw HF trades
  test.parquet                    — 257K raw HF trades
  train_features.parquet          — 1.11M × 74 cols (70 features + target + meta)
  test_features.parquet           — 257K × 74 cols
  feature_cols.json               — feature list

alex/.scratch/
  markets.parquet                 — 121MB HF markets cache
  backtest/                       — cached predictions per model (preds_*.npz)
  pressure_tests/                 — Phase 1/2 results + cost-fix comparison
```
