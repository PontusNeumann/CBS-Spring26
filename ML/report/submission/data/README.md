# Modeling data

Active modeling file: `data/consolidated_modeling_data.parquet` (317.5 MB)

Backtest-only sidecar: `data/backtest_context.parquet` (20 MB)

Target: `bet_correct` (int64, binary 0/1, balanced ~50.3% positive in both train and test).

**Leakage note:** all modeling features are temporally causal. The team's train/test split is market-cohort-disjoint (63 train markets, 10 test markets, zero overlap), which is the leak-defense mechanism for the within-market direction-determinism channel flagged in the audit. `submission/scripts/01_data_prep.py` excludes the forbidden lifetime/static columns before any model is fit and validates the backtest sidecar row alignment.

## Load

```python
import pandas as pd

df = pd.read_parquet("data/consolidated_modeling_data.parquet")

META_COLS = ["split", "market_id", "ts_dt", "timestamp"]
TARGET    = "bet_correct"

train = df[df["split"] == "train"]
test  = df[df["split"] == "test"]

feature_cols = [c for c in df.columns if c not in META_COLS + [TARGET]]
X_train, y_train = train[feature_cols], train[TARGET]
X_test,  y_test  = test[feature_cols],  test[TARGET]
```

## Shape

- 1,371,180 rows total
  - 1,114,003 train (`split == "train"`)
  - 257,177 test (`split == "test"`)
- 87 columns

## Column groups

| Group           | Count | Notes |
|-----------------|-------|-------|
| Meta / id       | 5     | `split`, `market_id`, `bet_correct` (target), `ts_dt`, `timestamp` |
| Core features   | 70    | Trade-, market-, taker-history- and microstructure-derived features |
| Wallet features | 12    | Point-in-time wallet enrichment (leakage-safe), all prefixed `wallet_*` plus `days_from_first_usdc_to_t` |

`consolidated_modeling_data.info.json` contains row/column counts, build timestamp, and the full wallet feature list.

## Backtest context

`backtest_context.parquet` is not a modeling input. It exists so the economic evaluation can be reproduced without relying on archived local files. It contains:

- `usd_amount`, the true trade size used for liquidity-aware bet sizing.
- `pre_yes_price_corrected`, the corrected YES-normalized pre-trade price used for cost, edge, and naive-consensus diagnostics.
- `price`, `token_amount`, `taker`, `taker_direction`, and `nonusdc_side`, which support SELL-semantics and consensus-vs-contrarian checks.
- `row_in_split`, `split`, `market_id`, and `timestamp`, which let `01_data_prep.py` and `05_backtest.py` verify row-for-row alignment with the modeling parquet.

The sidecar is derived from the archived raw Alex train/test parquets and sorted back into the same split order as `consolidated_modeling_data.parquet`. The alignment checks are recorded in `backtest_context.info.json`.

## Folder layout

```
data/
├── consolidated_modeling_data.parquet   # USE THIS for modeling
├── consolidated_modeling_data.info.json
├── backtest_context.parquet             # USE THIS for backtest context only
├── backtest_context.info.json
├── README.md
├── constant_substitution_policy_and–audit_trail.md
└── release-manifest-2026-04-29.md
```

Only `consolidated_modeling_data.parquet` is needed for supervised modeling. `backtest_context.parquet` is required by `05_backtest.py`.

## Provenance

- Source train/test split: `archive/train_features_walletjoined.parquet`, `archive/test_features_walletjoined.parquet` (built 2026-04-28 22:14)
- Wallet enrichment table: `wallet_enrichment.parquet` (128,671 wallets, alex_extras run completed 2026-04-28, 4 retries recovered)
- Consolidation date: 2026-04-29
