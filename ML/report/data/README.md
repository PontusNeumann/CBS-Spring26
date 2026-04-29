# Modeling data

Active file: `data/consolidated_modeling_data.parquet` (317.5 MB)

Target: `bet_correct` (int64, binary 0/1, balanced ~50.3% positive in both train and test).

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

## Folder layout

```
data/
├── consolidated_modeling_data.parquet   # ← USE THIS for modeling
├── consolidated_modeling_data.info.json
├── wallet_enrichment.parquet            # wallet table joined into the modeling file
├── README.md
├── MISSING_DATA.md
└── archive/                             # Frozen traceback only — not needed for modeling
    ├── train_features_walletjoined.parquet
    ├── test_features_walletjoined.parquet
    ├── pipeline/                        # Upstream raw + intermediate pipeline files
    └── alex/                            # Alex's pre-wallet-join pipeline artifacts
```

Only `consolidated_modeling_data.parquet` is needed for modeling. Everything in `archive/` is preserved for traceback.

## Provenance

- Source train/test split: `archive/train_features_walletjoined.parquet`, `archive/test_features_walletjoined.parquet` (built 2026-04-28 22:14)
- Wallet enrichment table: `wallet_enrichment.parquet` (128,671 wallets, alex_extras run completed 2026-04-28, 4 retries recovered)
- Consolidation date: 2026-04-29
