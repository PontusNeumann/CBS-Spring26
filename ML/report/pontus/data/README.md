# Modeling data

Active file: `consolidated_modeling_data.parquet` (317.5 MB)

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

Full wallet feature list is in `consolidated_modeling_data.info.json`.

## Provenance

- Source train/test split: `archive/train_features_walletjoined.parquet`, `archive/test_features_walletjoined.parquet` (built 2026-04-28 22:14)
- Wallet enrichment table: `../../data/wallet_enrichment.parquet` (128,671 wallets, alex_extras run completed 2026-04-28, 4 retries recovered)
- Consolidation date: 2026-04-29

## Archive

`archive/` holds the original per-split parquet files prior to consolidation. Kept for reproducibility; not used by training scripts.
