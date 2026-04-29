# Missing-data policy

Authoritative record for how NaN / missingness is represented in
`consolidated_modeling_data.parquet` (the team's single modeling dataset) and
how it should be handled downstream. Cited by `project_plan.md` §5.6 and
referenced from the paper's methodology section.

## 1. Typology

Missingness in the feature frame partitions into two classes:

**Structural missingness** — NaN on quantities that are mathematically
undefined given the prior history available at row time. Examples:

- `wallet_prior_win_rate` on a wallet's first trade (no prior labeled
  trades to average).
- `wallet_market_category_entropy` when the wallet has <2 prior distinct
  markets (entropy over one or zero categories is undefined).
- `wallet_directional_purity_in_market`, `wallet_spread_ratio`,
  `wallet_median_gap_in_market` on the first (wallet, market) trade.
- `size_vs_wallet_avg` on the first trade (no prior average).

For these, **NaN is the truthful value** — there is no underlying number
we failed to observe. Filling with a sentinel (0 or otherwise) would
conflate "not yet defined" with a legitimate realised value. For
`wallet_market_category_entropy`, zero is specifically a valid realised
value meaning a wallet concentrated in exactly one category; imputing 0
for undefined entropy would erase the distinction.

**Pipeline missingness** — NaN where the underlying quantity exists but
was not observed. Examples:

- All 12 Layer 6 on-chain columns (nine semantic features plus three log
  variants) on rows whose wallet failed Etherscan V2 enrichment
  (`wallet_enriched = 0`). The wallet does have an on-chain history; we
  simply could not retrieve it — in the initial pass plus one targeted
  retry pass, 459 of 109,080 wallets (0.42 %) remained permanently
  non-retrievable.
- `pct_time_elapsed` on markets missing both `resolution_ts` and
  `end_date` metadata.

For these, a 0 would be semantically wrong for most features (a real
wallet is never zero days old on Polygon; a real market does eventually
resolve). NaN plus the appropriate enrichment/pipeline flag is the
faithful representation.

## 2. Indicator columns in the dataset

One binary indicator per missingness species. Dtype int8, values in {0, 1}.
Added by `scripts/11b_add_missingness_flags.py` (all pre-existing
structural flags) and `scripts/11_add_layer6.py` (Layer 6 pipeline flag).

| Indicator | Covers | Added by | Defined |
|---|---|---|---|
| `wallet_has_prior_trades` | `wallet_prior_win_rate`, `wallet_prior_volume_usd`, `size_vs_wallet_avg` | `11b` | `wallet_prior_trades > 0` |
| `wallet_has_prior_trades_in_market` | `wallet_directional_purity_in_market`, `wallet_spread_ratio`, `wallet_median_gap_in_market` | `11b` | `wallet_prior_trades_in_market > 0` |
| `wallet_has_cross_market_history` | `wallet_market_category_entropy` | `11b` | entropy value is defined (structural NaN → 0; HF-absent pipeline NaN → 0) |
| `market_timing_known` | `pct_time_elapsed` | `11b` | `pct_time_elapsed` is defined |
| `wallet_enriched` | all 12 Layer 6 columns | `11` | Etherscan V2 enrichment succeeded |

Observed shares on the final 82-column frame (post Layer 6 integration +
retry pass, 22 Apr):

| Indicator | share of `1` |
|---|---|
| `wallet_has_prior_trades` | 90.98 % |
| `wallet_has_prior_trades_in_market` | 77.41 % |
| `wallet_has_cross_market_history` | 91.91 % |
| `market_timing_known` | 95.70 % |
| `wallet_enriched` | 99.69 % |

## 3. What the dataset stores and what it does not

**Stores:** NaN on every numeric feature column where the value is
structurally undefined or pipeline-unobserved. Indicator columns as
listed above. No imputation is applied in the feature frame itself.

**Does not store:** imputed values, model-specific transforms, or
per-fold statistics. Those belong in the modelling pipeline so per-split
leakage can be avoided.

## 4. Policy for the modelling stage

Per-model imputation is deferred to `scripts/12_train_mlp.py` and its
siblings. The indicator columns are always included as features, so
whatever imputation value is chosen for the raw feature never silently
destroys the "was this missing" signal.

Preferred defaults per model family:

- **Tree-based (Random Forest, HistGradientBoosting).** Accept NaN
  natively in sklearn ≥ 1.4. Pass the raw NaN-bearing columns through
  unchanged alongside their indicators.
- **Linear (Logistic Regression).** Median-impute each numeric feature
  using *train-split-only* statistics, then standardise. Keep the
  indicator column as a separate feature.
- **MLP.** Same as linear: median-impute on train, standardise, keep
  indicators as features.

Row dropping is not used. Sensitivity analyses (e.g. restricting to
rows where `wallet_has_prior_trades = 1`) may appear in the Discussion
if they sharpen a result, but the main evaluation uses the full frame.

## 5. Paper-side documentation

The methodology section of `ML_final_exam_paper.docx` should contain a
short paragraph matching this typology (see `project_plan.md` §5.6) and
cite Rubin (1976) or Little & Rubin (2019) for the general framework,
noting that our setting includes a structural missingness sub-type
beyond the classic MCAR/MAR/MNAR taxonomy.

## 6. Change log

- 2026-04-21 evening — initial policy. Introduced 4 missingness
  indicators, re-specified `11_add_layer6.py` to emit NaN on un-enriched
  rows rather than 0, documented the typology. Entry: Pontus N. + Claude.
- 2026-04-29 — Data folder consolidation. The build-pipeline output
  `03_consolidated_dataset.csv` was superseded by the team-shared
  `data/consolidated_modeling_data.parquet` (1,371,180 rows × 87 cols, with
  train/test in a `split` column). All five missingness indicators
  documented here (`wallet_has_prior_trades`,
  `wallet_has_prior_trades_in_market`, `wallet_has_cross_market_history`,
  `market_timing_known`, `wallet_enriched`) are present in the modeling
  parquet — they are part of the 70 core feature columns. The legacy
  `03_consolidated_dataset.csv` is preserved at
  `data/archive/pipeline/03_consolidated_dataset.csv` for traceback.
  Entry: Pontus N. + Claude.
- 2026-04-22 early AM — Layer 6 integration landed. Enrichment retry
  pass recovered 7,050 of 7,509 initially-failed wallets (93.9 %
  recovery). Final wallet coverage 99.58 %; trade-level Layer 6 coverage
  99.69 % (1,206,050 of 1,209,787 rows). `wallet_enriched` observed
  share added to §2 table. Corrected "9 Layer 6 features" wording to
  "12 Layer 6 columns (9 semantic + 3 log variants)" since the
  integrator emits all three `log1p` variants as separate columns.
  Bug found and fixed in `11_add_layer6.py`: pandas 3+ stores datetimes
  at microsecond resolution, not nanoseconds; the int64 → seconds
  conversion now casts to `datetime64[ns, UTC]` first so `// 10**9` is
  always correct. Entry: Pontus N. + Claude.
