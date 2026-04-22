# New-chat kickoff prompt

*Copy-paste the block below into a fresh Claude Code session. Tailored to pick up the CBS MLDP exam project where the 2026-04-22 session paused, with a clean pivot toward: define cohorts → download → features → baseline → model sweep → iterate.*

---

```
I'm continuing a CBS MSc exam project on Polymarket Iran markets —
"Mispricing on Polymarkets: detecting probability asymmetries in Iran
geopolitical markets with machine learning." The previous Claude session
cleaned the data pipeline extensively and exposed structural issues with
our initial experimental design. I want to pivot and start fresh on the
modelling, working in my personal workspace at
`/Users/alex/Documents/Claude Projects/CBS-Spring26/ML/report/alex/`.

## Hard constraints

- **TensorFlow / Keras only for all neural-network code.** PyTorch is
  NOT permitted (CBS MLDP exam requirement). Pontus's existing
  `scripts/12_train_mlp.py` uses PyTorch and is unusable for
  submission — don't reference it.
- Sklearn for non-neural baselines (LogReg, RF, GBM, Isolation Forest)
  is fine.
- All new modelling code I write should live in `alex/scripts/`. Shared
  pipeline scripts in `ML/report/scripts/` are Pontus's territory — I
  can read them but shouldn't modify them from my workspace.

## Read before doing anything

1. `ML/report/FYI-pontus-and-claude.md` — short orientation. Includes
   the framework-constraint warning.
2. `ML/report/data-pipeline-issues.md` — exhaustive log of P0/P1 bugs
   found and fixed or mitigated. Critical for avoiding silent
   contamination.
3. `ML/report/alex/notes/session-learnings-2026-04-22.md` — distilled
   lessons from the previous session. Read this BEFORE touching
   features.
4. `ML/report/alex/notes/feature-exclusion-list.md` — canonical
   `NON_FEATURE_COLS` set. Never use `side`, `outcomeIndex`, or any
   direction-dependent feature in training.
5. `ML/report/project_plan.md` — §4 Split row for the cohort design;
   §5 for modelling spec.
6. `ML/report/alex/notes/sister-market-features.md` — deferred feature
   idea + trigger conditions.

## My workflow for this session

I want to go step by step, in this order:

### 1. Define train / val / test markets
Current setup (in `ML/report/scripts/14_build_experiment_splits.py`):
- Train: 4 strike markets (Feb 25-28), ~202k rows
- Val: 1 conflict-end (Mar 15), ~13k rows
- Test: 7 ceasefires (Apr 8-18), all NO, ~13k rows

Previous session's critique: val + test are both entirely NO-resolution
by construction (no non-strike YES markets exist in the current
dataset). For a more defensible generalisation claim, we should
consider pulling additional clusters from the HuggingFace mirror
`SII-WANGZJ/Polymarket_data` — e.g. Maduro / Venezuela regime-change
markets (plan §8 lists these as reserved future work with documented
insider-trading cases per Mitts & Ofir 2026). Decide first: do we
pull new clusters, and if so which? Then update the cohort definitions.

### 2. Download / build the needed data
If pulling new clusters: extend `scripts/02_build_dataset.py` with new
`event_id`s and run the HF stream pipeline. Layer 6 Etherscan
enrichment is a separate ~3h run per cluster per batch of new wallets.
Layer 7 (cross-market entropy) already covers the full Polymarket
universe — no re-run needed. Document any new data under
`alex/outputs/` or similar.

### 3. Define a BASIC feature set (~10 features)
Previous session used 44 features and hit a wall at 0.53 test ROC.
Pivot approach: strip to ~10 "obviously safe" features, then ADD
features back incrementally in Phase 2 to isolate what contributes.

Candidate basic set (from previous session's notes):
- `log_size`, `trade_value_usd`
- `wallet_prior_trades`, `wallet_prior_volume_usd`,
  `wallet_first_minus_trade_sec`
- `pct_time_elapsed`
- `market_price_vol_last_1h`
- `wallet_trades_in_market_last_10min`
- `wallet_prior_trades_in_market`
- `wallet_market_category_entropy`

All market-agnostic, all obviously no-lookahead. No Layer 6, no
direction-dependent features, no sizing ratios.

### 4. Run baselines on the basic feature set
`alex/scripts/03_baselines_sweep.py` already exists as a template.
Re-run with the basic 10-feature set. Expect modest numbers.

### 5. Test multiple models (sweep)
Non-neural: LogReg (L2, L1), RandomForest (with heavy regularisation
this time — previous session showed trees memorise wallets without
it), HistGradientBoosting, Gaussian NB, Decision Tree.

Neural (TF/Keras, REQUIRED for submission): basic MLP — 2-3 hidden
layers, SELU or ReLU, batch normalisation, Adam, early stopping.
Plan §5.1 has the spec.

### 6. Pick winner, tune
By val ROC penalising train-val gap (trees get demoted for memorising).
Tune the winner's hyperparameters via a small grid or random search.
Report test metrics only after tuning on val.

### 7. Then decide
Based on where tuned winner caps out:
- If strong (~0.60+ test ROC) → write up RQ1 as positive.
- If modest (~0.55) → add features back diagnostically (Phase 2), see
  which family moves the needle.
- If weak (~0.52) → pivot hard to RQ2 feature-importance narrative, or
  invest in richer data (Maduro cluster, sister-market features,
  correctly-recomputed `wallet_prior_win_rate`).

## What I want from you this session

- Sparring-partner mode when I'm writing code (review my diffs, catch
  bugs, push back on design choices).
- Drive mode when I say "write this for me" — produce the code
  directly. Default to sparring-partner.
- Concise updates. Don't narrate. Reference file paths and line numbers.
- Log any new findings or issues discovered in
  `ML/report/data-pipeline-issues.md`. Commit them.
- Keep commits small and focused. Open PRs against
  `PontusNeumann/CBS-Spring26` main. Acceptable to self-merge
  administrative / alex-workspace changes; flag shared-pipeline
  changes for review.

## Where we are right now

Last commit on main: see `git log` — PR #11 merged earlier today.
There's an uncommitted branch `alex/logreg-baseline` with the
LogReg baseline, sweep results, naive-market investigation, and
residual-edge analysis. Review those outputs before starting fresh:

- `alex/outputs/baselines/logreg/` — initial LogReg baseline.
- `alex/outputs/baselines/sweep/` — 7-model sweep results.
- `alex/outputs/investigations/naive_market/` — Simpson's paradox proof.
- `alex/outputs/investigations/residual_edge/` — RQ1b residual-edge
  analysis. +0.06 partial correlation on test.

Start with step 1: discuss cohort-market decisions. What am I actually
testing, and is the current setup good enough, or do we need to pull
new data?
```

---

## How to adapt this

- Delete any sections you've already completed before pasting.
- If you've decided on new clusters, add their `event_id`s explicitly in step 2.
- If you want me to drive from the start, change "sparring-partner mode" to "drive mode" in the `what I want from you` section.
