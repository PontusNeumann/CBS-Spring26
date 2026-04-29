# Pontus wallet enrichment 2026-04-28 v2 (full-coverage)

Companion data for the wallet-feature merge into Alex's idea1 cohort.
Extract to `ML/report/data/` to reproduce all wallet-feature derivations
on top of `alex/data/{train,test}.parquet`.

**Supersedes** the original `pontus-wallet-2026-04-28` release. This v2
covers every taker in Alex's cohort with `fetch_status == ok` (zero hard
failures after three Etherscan retry passes).

## Coverage against Alex's idea1 cohort

| Side | Wallets | Trades |
|---|---:|---:|
| Alex cohort `train + test` (1,371,180 trades) | 120,672 unique takers | 1,371,180 |
| Present in `wallet_enrichment.parquet` | **120,672 (100.0%)** | **1,371,180 (100.0%)** |
| With `fetch_status == 'ok'` | **120,672 (100.0%)** | **1,371,180 (100.0%)** |

Layer-6 NaN rate on the full cohort is now **0%**. No `is_wallet_enriched=0`
fallback rows; `wallet_enriched` flag is constant 1 across the joined
parquets and may safely be dropped at training time.

## How v2 was produced

Three Etherscan retry passes against the original 2026-04-22 enrichment:

1. `enrich_missing_alex_wallets.py` — added 19,591 takers that were in
   Alex's cohort but not in Pontus's idea1 cohort. 18,257 ok / 1,334
   transient NOTOK after first pass.
2. `retry_failed_wallets.py` (round 1) — 1,793 → 120 hard-failed
   (1,673 of 1,793 recovered, 93.3%).
3. `retry_failed_wallets.py` (round 2) — 120 → 4 (116 recovered, 96.7%).
4. `retry_failed_wallets.py` (round 3) — 4 → 0 (100%).

Concurrency: 18 workers across 3 Etherscan keys, per-key cap 2.86 rps.
Total wall time across all four passes: ~2 h 10 m.

## Join key

`taker` (Alex's `train.parquet` / `test.parquet`, mixed-checksum case)
↔ `wallet` (this file, mixed-checksum case). Lowercase both sides
before joining.

```python
takers = pd.read_parquet("ML/report/alex/data/train.parquet", columns=["taker"])
we = pd.read_parquet("ML/report/data/wallet_enrichment.parquet")
takers["taker_lc"] = takers.taker.str.lower()
we["wallet_lc"] = we.wallet.str.lower()
merged = takers.merge(we, left_on="taker_lc", right_on="wallet_lc", how="left")
```

## `wallet_enrichment.parquet` (354 MB extracted, sha256:b104528d338a5284)

- Rows: **128,671** (one row per Polygon address)
- Columns: 12
- Source: PolygonScan API (`scripts/03_enrich_wallets.py`),
  pulled 2026-04-22 + 2026-04-28 extensions / retries.

  | column | dtype | notes |
  |---|---|---|
  | wallet | str | Polygon address, mixed-checksum case |
  | polygon_first_tx_ts | float64 (unix s) | first Polygon tx |
  | funded_by_cex | float64 (0/1) | first inbound USDC came from a known CEX hot wallet |
  | cex_label | str | name of CEX if `funded_by_cex == 1` (~99.8% null otherwise) |
  | first_usdc_inbound_ts | float64 (unix s) | first USDC inbound |
  | first_usdc_inbound_amount_usd | float64 | size of that first inbound |
  | outbound_ts | str (JSON list) | outbound USDC transfer timestamps |
  | inbound_ts | str (JSON list) | inbound USDC transfer timestamps |
  | cex_deposit_ts | str (JSON list) | CEX-deposit timestamps (subset of outbound) |
  | cex_deposit_amounts_usd | str (JSON list) | matching deposit sizes |
  | n_tokentx | int64 | total token transfer count for this wallet |
  | fetch_status | str | `ok` for **all 128,671** rows in v2 |

## Derived features (NOT in this tarball — recompute on Alex's cohort)

This file ships **wallet-level constants only**. The 12 causal Layer-6
features used downstream are derived per-trade by joining these constants
with the trade timestamp and bisecting prior history. The reference
implementation lives at `pontus/scripts/build_walletjoined_features.py`.

| Feature | Formula |
|---|---|
| `wallet_polygon_age_at_t_days` | `(trade_ts − polygon_first_tx_ts) / 86400` |
| `wallet_polygon_nonce_at_t` | `searchsorted(outbound_ts, trade_ts, side='left')` |
| `wallet_n_inbound_at_t` | `searchsorted(inbound_ts, trade_ts, side='left')` |
| `wallet_n_cex_deposits_at_t` | `searchsorted(cex_deposit_ts, trade_ts, side='left')` |
| `wallet_cex_usdc_cumulative_at_t` | sum of `cex_deposit_amounts_usd` up to that index |
| `days_from_first_usdc_to_t` | `(trade_ts − first_usdc_inbound_ts) / 86400` if ≥ 0 else NaN |
| `wallet_funded_by_cex_scoped` | `funded_by_cex AND first_usdc_inbound_ts < trade_ts` |
| `wallet_log_polygon_nonce_at_t`, `wallet_log_n_inbound_at_t`, `wallet_log_cex_usdc_cum` | `log1p` of the matching count/amount |
| `wallet_enriched` | constant 1 (kept in output schema for downstream compatibility) |
| `wallet_funded_by_cex` | passthrough of the time-invariant flag |

## Downstream causal regression guard

`pontus/scripts/_check_causal_joined.py` asserts the causal contract on
the joined parquets. v2 passes 51/51 checks on both train and test
splits; downstream tooling can rely on:

- `wallet_enriched == 1` for every row.
- All Layer-6 numerics ≥ 0 on enriched rows where applicable.
- `wallet_funded_by_cex_scoped ≤ wallet_funded_by_cex` always.
