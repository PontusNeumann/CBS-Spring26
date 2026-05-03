# Missing-data policy

Authoritative record for how missingness is represented in
`consolidated_modeling_data.parquet` (the team's single modeling dataset)
and how it should be handled downstream. Cited by `project_plan.md` §5.6
and referenced from the paper's methodology section.

## 1. Summary

The modeling dataset contains **zero NaN values** across all 1,371,180
rows × 87 columns. Structural missingness — cases where a feature is
mathematically undefined given the prior history available at row time —
is resolved by substitution at the feature-engineering stage with a
semantically meaningful constant rather than retained as NaN. Two binary
indicator columns mark the most informative missingness species so the
model can still learn from the "was this row at the cold-start boundary"
signal.

This represents a deliberate methodological shift from an earlier policy
that preserved NaN and added five indicator columns (recorded in §6 below
for traceability). The current convention is simpler to defend in the
report and removes any need for per-model imputation logic, at the cost
of fewer explicit indicators than the prior design.

## 2. Substitution rules in the feature-engineering stage

All rules are applied in `alex/scripts/06b_engineer_features.py` before
the parquet is written. Every fill value is a constant — none are
learned from train statistics — so the rules are leakage-safe by
construction.

| Class of feature | Structural-missing case | Substituted value | Rationale |
|---|---|---|---|
| Cumulative `log_*` counts and volumes (`log_taker_prior_trades_total`, `log_taker_cumvol_in_market`, `log_taker_prior_volume_total_usd`) | Wallet's first-ever trade or first trade in market | 0 | `np.log1p(0) = 0` is the correct mathematical value, not an imputation |
| Per-(market, taker) features (`taker_directional_purity_in_market`, `taker_position_size_before_trade`, `log_taker_prior_trades_in_market`) | First trade in a market | 0 | Explicit `pos_cum.loc[first_in_mt] = 0`; first-row reset before `cumsum().shift(1)` |
| Cross-market wallet share (`taker_yes_share_global`) | Wallet's first-ever trade | 0.5 | `fillna(0.5)` — neutral prior, no directional information |
| Wallet on-chain features (12 columns: `wallet_polygon_age_at_t_days`, `wallet_polygon_nonce_at_t`, `wallet_n_inbound_at_t`, `wallet_n_cex_deposits_at_t`, `wallet_cex_usdc_cumulative_at_t`, `wallet_funded_by_cex`, `wallet_funded_by_cex_scoped`, `days_from_first_usdc_to_t`, plus three `log_*` variants) | Etherscan V2 enrichment failed (~0.31% of trades; 459 of 109,080 wallets remained non-retrievable after one retry pass) | 0 | The `wallet_enriched` binary indicator stays at 0 to mark these rows so the model can distinguish substituted-zero from observed-zero |

## 3. Indicator columns retained in the dataset

| Indicator | Covers | Defined as |
|---|---|---|
| `wallet_enriched` | all 12 wallet on-chain columns | 1 if Etherscan V2 enrichment succeeded for the wallet, 0 otherwise |
| `taker_first_trade_in_market` | per-(market, taker) features (purity, position, in-market priors) | 1 on the wallet's first trade in this market, 0 otherwise |

`wallet_enriched` carries the pipeline-missingness signal end-to-end.
`taker_first_trade_in_market` carries the structural-missingness signal
for the per-market group, which is the largest zero-density cluster in
the data (panel `01_zero_density.png` in the EDA shows this as the
single most-zeroed feature group).

What is **not** flagged with an explicit indicator:

- Wallet's first-ever trade across markets (the cold-start case for
  `log_taker_prior_trades_total`, `log_taker_cumvol_in_market`,
  `taker_yes_share_global`). The substituted values 0 / 0.5 are visible
  in the data but not bracketed by a dedicated indicator. The earlier
  policy used `wallet_has_prior_trades` for this; it is not part of the
  current dataset.
- The cross-market wallet history depth and the market-timing known/unknown
  cases. The earlier policy used `wallet_has_cross_market_history` and
  `market_timing_known`; neither is present in the current dataset.

The EDA panel `01_zero_density.png` lists the 20 features with the
highest share of exact-zero values; readers can use it to assess where
the implicit "no information" mass concentrates.

## 4. Policy for the modelling stage

Because every fill value is a constant rather than a learned statistic,
**no imputation step is required** at modelling time. The same numeric
frame is fed to every model family.

- **Tree-based (Random Forest, HistGradientBoosting).** Pass through
  unchanged. Tree splits handle the zero mass automatically.
- **Linear (Logistic Regression).** Standardise on train, apply to test.
  No imputation; no NaN to handle.
- **MLP.** Same as linear: standardise on train, apply to test.

The two indicator columns (`wallet_enriched`, `taker_first_trade_in_market`)
are included as features in all model families so the model can route
the substituted-zero cases differently from real zeros.

Row dropping is not used. Sensitivity analyses restricted to enriched
wallets (`wallet_enriched = 1`) or to non-cold-start trades
(`taker_first_trade_in_market = 0`) may appear in the Discussion if
they sharpen a result.

## 5. Paper-side documentation

The methodology section should contain a short paragraph matching this
substitution table. Suggested phrasing:

> *Numeric features were substituted at the feature-engineering stage with
> constants chosen on semantic grounds. Cumulative-history features take
> value 0 on a wallet's first trade (mathematically `log(0+1) = 0`, not an
> imputation), per-market features take value 0 on the first trade in a
> market, and `taker_yes_share_global` takes 0.5 in the absence of prior
> trades. Wallets that failed Etherscan enrichment retain 0 across the 12
> on-chain columns and are flagged by the binary indicator
> `wallet_enriched`; the first-trade-in-market case is flagged by
> `taker_first_trade_in_market`. Because every fill value is a constant
> rather than a learned statistic, no train-only fit is needed and no
> leakage is introduced (Géron, 2022, §2; Hastie, Tibshirani, & Friedman,
> 2009, §9.6).*

The earlier NaN-plus-five-indicators policy is grounded in Rubin (1976)
and Little & Rubin (2019); the current constant-substitution policy is
grounded in the SimpleImputer convention covered in CBS Lecture 2 and in
Géron (2022) §2.

## 6. Change log

- 2026-04-21 evening — initial policy. NaN preserved on every
  structural / pipeline missing case; five indicator columns introduced
  (`wallet_has_prior_trades`, `wallet_has_prior_trades_in_market`,
  `wallet_has_cross_market_history`, `market_timing_known`,
  `wallet_enriched`) by `scripts/11b_add_missingness_flags.py` and
  `scripts/11_add_layer6.py`. Per-model imputation specified for the
  modelling stage (median on train for linear/MLP, native NaN for tree
  models). Entry: Pontus N. + Claude.

- 2026-04-22 early AM — Layer 6 integration landed. Enrichment retry
  pass recovered 7,050 of 7,509 initially-failed wallets (93.9% recovery).
  Final wallet coverage 99.58%; trade-level Layer 6 coverage 99.69%
  (1,206,050 of 1,209,787 rows in Pontus's pipeline). Corrected "9
  Layer 6 features" wording to "12 Layer 6 columns (9 semantic + 3 log
  variants)" since the integrator emits all three `log1p` variants as
  separate columns. Bug fixed in `11_add_layer6.py`: pandas 3+ stores
  datetimes at microsecond resolution, not nanoseconds; the int64 →
  seconds conversion now casts to `datetime64[ns, UTC]` first so
  `// 10**9` is always correct. Entry: Pontus N. + Claude.

- 2026-04-29 — Data folder consolidation + methodology shift. The
  build-pipeline output `03_consolidated_dataset.csv` (which carried the
  five indicator columns described in the 2026-04-21 entry) was
  superseded by the team-shared `data/consolidated_modeling_data.parquet`
  (1,371,180 rows × 87 cols, train/test in a `split` column). The new
  parquet comes from Alex's `06b_engineer_features.py` pipeline, which
  does not preserve NaN — it substitutes at compute time with the
  constants tabled in §2 above. As a result, three of the five prior
  indicator columns (`wallet_has_prior_trades`,
  `wallet_has_prior_trades_in_market`, `wallet_has_cross_market_history`,
  `market_timing_known`) are not present in the modeling dataset. The
  two retained indicators are `wallet_enriched` (carried over) and
  `taker_first_trade_in_market` (new in Alex's pipeline). The
  methodology in §§1–5 was rewritten to describe the constant-
  substitution policy. Trade-off: simpler to write up and removes per-
  model imputation logic; loses some explicit signal vs the prior
  policy. The legacy `03_consolidated_dataset.csv` is preserved at
  `data/archive/pipeline/03_consolidated_dataset.csv` for traceback.
  Entry: Pontus N. + Claude.
