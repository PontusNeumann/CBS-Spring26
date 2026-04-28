# Pontus wallet enrichment snapshot 2026-04-28

Companion data for the wallet-feature merge into Alex's idea1 cohort. Extract to `ML/report/data/` to reproduce all wallet-feature derivations on top of `alex/data/{train,test}.parquet`.

## Coverage against Alex's idea1 cohort

| Side | Wallets | Trades |
|---|---:|---:|
| Alex cohort `train + test` (1,371,180 trades) | 120,672 unique takers | 1,371,180 |
| Present in `wallet_enrichment.parquet` | 101,081 (83.8%) | 1,268,123 (**92.5%**) |
| With `fetch_status == 'ok'` | 100,694 (83.4%) | 1,264,693 (**92.2%**) |

Coverage is wallet-volume-weighted. The ~19,600 missing wallets are mostly low-trade-count newcomers; flag uncovered rows with `is_wallet_enriched = 0` rather than dropping.

## Join key

`taker` (Alex's `train.parquet` / `test.parquet`, mixed-checksum case) ↔ `wallet` (this file, mixed-checksum case). Lowercase both sides before joining.

```python
takers = pd.read_parquet("ML/report/alex/data/train.parquet", columns=["taker"])
we = pd.read_parquet("ML/report/data/wallet_enrichment.parquet")
takers["taker_lc"] = takers.taker.str.lower()
we["wallet_lc"] = we.wallet.str.lower()
merged = takers.merge(we, left_on="taker_lc", right_on="wallet_lc", how="left")
```

## `wallet_enrichment.parquet` (308 MB extracted, sha256:2ed29502381f8be1)

- Rows: 109,080 (one row per Polygon address)
- Columns: 12
- Source: PolygonScan API (`scripts/03_enrich_wallets.py`), pulled 2026-04-22

  | column | dtype | notes |
  |---|---|---|
  | wallet | str | Polygon address, mixed-checksum case |
  | polygon_first_tx_ts | float64 (unix s) | timestamp of wallet's first Polygon tx; null on fetch error |
  | funded_by_cex | float64 (0/1) | 1 iff first inbound USDC came from a known CEX hot-wallet |
  | cex_label | str | name of CEX if `funded_by_cex == 1`; null otherwise (~99.8% null) |
  | first_usdc_inbound_ts | float64 (unix s) | timestamp of first USDC inbound transfer |
  | first_usdc_inbound_amount_usd | float64 | size of that first inbound, in USDC |
  | outbound_ts | str (JSON list) | serialised list of outbound USDC transfer timestamps |
  | inbound_ts | str (JSON list) | serialised list of inbound USDC transfer timestamps |
  | cex_deposit_ts | str (JSON list) | serialised list of CEX-deposit timestamps (subset of outbound) |
  | cex_deposit_amounts_usd | str (JSON list) | matching deposit sizes |
  | n_tokentx | int64 | total token transfer count for this wallet on Polygon |
  | fetch_status | str | `ok` (108,621 wallets) or `error: …` (459 wallets — API NOTOK on retry) |

## Derived features (NOT in this tarball — recompute on Alex's cohort)

This file ships **wallet-level constants only**. The causal features used in Pontus's strict-branch model are derived per-trade by joining these constants with the trade timestamp and re-indexing prior history. Recompute from this table using:

| Feature | Formula | Script reference |
|---|---|---|
| `wallet_age_at_trade_days` | `(trade_ts - polygon_first_tx_ts) / 86400` | `scripts/03_enrich_wallets.py` |
| `wallet_funded_by_cex_scoped` | `funded_by_cex AND first_usdc_inbound_ts < trade_ts` | `scripts/03_enrich_wallets.py` |
| `wallet_usdc_cumulative_at_trade` | sum of `inbound_ts - outbound_ts` filtered to `< trade_ts` | `scripts/03_enrich_wallets.py` |
| `wallet_n_cex_deposits_before_trade` | count of `cex_deposit_ts < trade_ts` | `scripts/03_enrich_wallets.py` |
| `wallet_prior_win_rate_causal` | win-rate over wallet's resolved priors with `resolution_ts < trade_ts` | `scripts/03_enrich_wallets.py` (needs market resolution table) |
| `wallet_category_entropy` | Shannon entropy over event-category mix of wallet's prior trades | `scripts/10_wallet_category_entropy.py` |

`wallet_prior_win_rate_causal` and `wallet_category_entropy` need the full Alex-cohort trade history scan; expect that to be the overnight job. The other four are O(n) lookups from this table.
