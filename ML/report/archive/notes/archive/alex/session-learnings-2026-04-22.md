# Session learnings — 2026-04-22

*Distilled lessons from the full-day data-cleaning + baseline-sweep session. Save future sessions from re-deriving these.*

## Hard constraints (never forget)

- **TF / Keras only.** CBS MLDP exam requires TensorFlow / Keras for all neural-network code. PyTorch is not permitted. Pontus's `scripts/12_train_mlp.py` uses PyTorch — it's not usable for the submission as-is.

## Data-pipeline bugs and fixes (all in `data-pipeline-issues.md`)

**Fixed upstream:**
- P0-3: `_first_lock_timestamp` was too strict → 3 markets had `NaT` resolution. Now uses robust median check.
- `is_yes` column now derived in `enrich_trades` from `outcomes[winning_outcome_index]`.

**Mitigated at training time (features dropped from `NON_FEATURE_COLS`):**
- P0-1: `wallet_is_whale_in_market` uses end-of-market p95 threshold.
- P0-2: `is_position_exit` misfires on first-ever SELL.
- P0-8: market-identifying absolute-scale features (`time_to_settlement_s`, `market_volume_so_far_usd`, etc.) + the `size_x_time_to_settlement` interaction.
- P0-9: `wallet_prior_win_rate` peeks at future market resolutions via cumsum on bet_correct.
- **P0-11: `side` and `outcomeIndex` together deterministically encode bet_correct, and the mapping FLIPS across market resolutions.** Tree models exploited the interaction and catastrophically inverted on test (ROC 0.00).
- P0-12: indirect direction-dependent features (`wallet_position_size_before_trade`, `is_position_flip`, `wallet_cumvol_same_side_last_10min`, `wallet_directional_purity_in_market`, `wallet_has_both_sides_in_market`, `market_buy_share_running`, `trade_size_vs_position_pct`).

**Documented but not yet fixed:**
- P0-4: `market_implied_prob` contaminated by trade-execution price on HF-path markets (67 of 74 markets). Only affects RQ1b trading-rule benchmark.
- P0-5: val was temporally before train → fixed by moving val to Mar 15 conflict-end.
- P0-6: wallet group leakage. Trees still memorise wallets via Layer 6 on-chain fingerprints. Need GroupKFold by `proxyWallet` for honest CV.
- P0-7: correlated errors within wallet bursts → bootstrap test CIs at wallet level.
- P0-10: HF-path trades are per-fill (ratio 1.56 rows/tx); API-path is per-order (1.00 rows/tx). Train/test granularity mismatch.

## Key diagnostic insights

### 1. The (side, outcomeIndex, resolution) triple determinism
Within any single market, `(side, outcomeIndex)` perfectly determines `bet_correct`. But the mapping flips between YES and NO markets. **Training on mixed-resolution markets creates an averaged association that doesn't generalise to single-resolution test cohorts.**
Any feature derived from `side` or `outcomeIndex` carries this contamination.

### 2. Naive-market 0.63 ROC on test is Simpson's paradox
`p_hat = market_implied_prob` scored 0.63 test ROC, making it look like our trained models (0.53) were being beaten. Investigation in `alex/outputs/investigations/naive_market/` proved:
- Within every (market × side × outcomeIndex) cell on test, `bet_correct` is constant.
- Within-cell ROC is mathematically undefined.
- The 0.63 aggregate comes purely from ranking between cells via `side × outcomeIndex` distribution.
- **Efficient-market null does not imply naive-baseline ROC = 0.5.** It implies calibration. Aggregate ROC depends on trade-direction mix.

### 3. Residual edge is the real RQ1b test
Aggregate ROC comparison is the wrong test. The plan's RQ1b is about whether `p_hat − market_implied_prob` carries information. When we compute the *residual* of edge after linearly projecting out `market_implied_prob`:
- Residual edge ROC: 0.55 (train), 0.59 (val), 0.53 (test).
- Partial correlation `edge ⊥ MIP` with `bet_correct`: +0.06 on test (~7σ).
- Signal exists. Modest but real.

### 4. Tree models memorise wallets via Layer 6 fingerprints
Even after dropping direction leaks, trees score 0.86-0.96 on train vs 0.52 on test. They use Layer 6 on-chain identity columns (polygon age, nonce, CEX history) as near-unique wallet identifiers. Untuned trees = nearest-neighbour lookup by wallet. Need heavy regularisation (`min_samples_leaf=200+`) + `GroupKFold(proxyWallet)` for honest generalisation.

### 5. All-NO test is fine; "needing YES in test" was overcooked concern
Target `bet_correct` is a per-trade label that's ~50/50 within every market regardless of resolution. The model doesn't care about per-market YES/NO mix. What *does* matter: residual-edge evaluation, calibration per subgroup.

## Cohort specs (current, leakage-free)

| Cohort | Rows | Markets | Resolution | Span |
|---|---:|---|---|---|
| Train | 202,082 | 4 strike (Feb 25-28) | 1 YES + 3 NO | through Feb 28 |
| Val | 13,154 | 1 conflict-end (Mar 15) | NO | Feb 28 → Mar 17 |
| Test | 13,414 | 7 ceasefires (Apr 8-18) | all NO | Apr 8 → Apr 19 |

Chronology: train → val → test strictly.

## Sweep results after all leakage pruning (32 features, direction-free)

| Model | Train ROC | Val ROC | Test ROC |
|---|---:|---:|---:|
| LogReg L2 | 0.556 | 0.558 | 0.532 |
| LogReg L1 | 0.555 | 0.559 | 0.532 |
| Gaussian NB | 0.544 | 0.562 | 0.529 |
| Random Forest | 0.957 | 0.606 | 0.522 |
| Extra Trees | 0.859 | 0.550 | 0.519 |
| Decision Tree | 0.707 | 0.496 | 0.496 |
| HistGBM | 0.843 | 0.600 | 0.483 |

All cluster around 0.50-0.53 test. **Signal at this feature set is weak but real.**

## What the pivot proposes

1. **Basic feature set** (~10 features) — clean, market-agnostic, obviously no-lookahead. Drop Layer 6 + sizing ratios temporarily; add back diagnostically in Phase 2.
2. **Add-back ablation** — measure which feature group contributes which ROC points. Publishable "we attributed signal to family X" narrative.
3. **Consider pulling Maduro / Biden-pardons cluster from HF** for a richer, balanced val/test with documented insider-trading cases.
4. **MLP in TensorFlow / Keras** — start fresh, don't inherit Pontus's PyTorch implementation.

## How to avoid re-deriving all this

- Read `data-pipeline-issues.md` in full before touching features.
- Use the canonical `NON_FEATURE_COLS` set in `alex/notes/feature-exclusion-list.md`.
- Never include `side` or `outcomeIndex` as features — or any feature that uses them (signed position, same-side filters, outcome-share aggregates).
- When comparing a trained model to naive market, use **residual edge**, not raw ROC.
- Trees need heavy regularisation + wallet-stratified CV or they'll memorise Layer 6 fingerprints.
- TF / Keras only. Never PyTorch.
