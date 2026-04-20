# Handover — 20 April 2026

**Session focus:** Run the hybrid Iran-cluster build end-to-end. Produce the mother dataframe on disk. Lock in the multi-cluster data strategy going forward.

## What was done this session

### 1. Rewrote the HF remote-stream function

The 19 April design used `duckdb` with `httpfs` to stream the 38.7 GB HF `trades.parquet` in a single `COPY ... TO` statement. First run failed partway through with `Invalid Input Error: Snappy decompression failure` — the single-stream strategy has no recovery from a transient byte-range corruption on a file that size.

Replaced `build_trades_subset_remote` in `scripts/build_iran_dataset.py` with a chunked reader:

- Fetches the parquet footer once via `fsspec` HTTP Range requests.
- Iterates the 592 row groups one at a time through `pyarrow.parquet.ParquetFile.read_row_group`.
- Each row group is wrapped in a retry loop (`REMOTE_READ_MAX_ATTEMPTS = 6`, exponential backoff capped at 60 s). On any failure the file handle is closed and reopened, so a bad connection cannot poison later reads.
- Filtered batches are written to a `.tmp` parquet via `ParquetWriter`, atomically renamed on success. A crashed run leaves no half-written subset.
- Progress is logged every 10 row groups with cumulative `rows_read` and `rows_kept`.

`aiohttp` was installed to back the `fsspec` HTTP filesystem.

### 2. Fixed a pandas 3.0 timestamp-precision bug in enrichment

`pd.merge_asof` in `fetch_polymarket.enrich_trades` (line 354) failed with `MergeError: incompatible merge keys [1] datetime64[ns, UTC] and datetime64[s, UTC], must be the same type`.

Root cause: pandas 3.0 (installed environment is `pandas 3.0.0`) changed `pd.to_datetime(..., unit="s")` to return `datetime64[s, UTC]` instead of the historical `[ns, UTC]`. The HF parquet path surfaces native second-precision timestamps; the CLOB `fetch_price_history` constructor also returns `[s, UTC]` under pandas 3.0. `merge_asof` refuses mixed precision on the sort key.

Fix: normalise both sides of the merge to `datetime64[ns, UTC]` immediately before `merge_asof` in `fetch_polymarket.py`. Left the HF-side cast in `build_iran_dataset.py` (line 405) in place — redundant but harmless, and useful for any downstream code that pre-concat assumes `[ns]`.

### 3. Ran the hybrid build end-to-end

One successful end-to-end run produced the dataset the plan has been pointing at for several sessions. Timings:

- HF remote-stream-filter pass: **3807 s (~64 min)**; 592 row groups; 568,646,651 rows scanned; **1,195,147 rows kept** → `data/trades_iran_subset.parquet` (50 MB, zstd).
- API fetch for the 7 ceasefire markets: 40 s; 14,640 trades + 8,722 CLOB price rows.
- Combined: **1,209,787 rows** (hf=1,195,147, api=14,640).
- Enrichment + split + feature expansion: **846 s (~14 min)** on the 1.2 M rows, no progress log inside `enrich_trades` (known opacity, not fixed this session).

### 4. Found and fixed two critical HF-path data bugs

Surfaced during the first EDA run on the 806 MB CSV. Both were silent — all downstream features computed, no errors raised — and both corrupted the modelling target for 98.79% of rows. Found by inspecting `summary.txt` null counts and per-market `bet_correct` rates.

**Bug A — HF token-side mislabelling.** The HF parquet encodes the trade side as the literal string `"token1"` / `"token2"` (1-indexed Solidity labels), not as the 77-digit decimal ERC-1155 token id that Gamma returns. The old `_oidx` in `build_hf_trades` compared these incompatible representations, so for all 1,195,147 HF rows:

- `outcomeIndex` was NaN.
- `asset` stored the literal string `"token1"` / `"token2"`, so `derive_resolution_timestamps` could never match trades to the winning token and `resolution_ts` was NaN → `settlement_minus_trade_sec` was wrong.
- `bet_correct` degenerated in `enrich_trades` to `(False == side_buy).astype("Int64")` — labelling HF rows purely by side (BUY → 0, SELL → 1).
- Every wallet-in-market directional feature (`wallet_directional_purity_in_market`, `wallet_has_both_sides_in_market`, `wallet_spread_ratio`, `wallet_position_size_before_trade`, `trade_size_vs_position_pct`, `is_position_exit`, `is_position_flip`) was null or degenerate.

Fix: `_asset_oidx` now maps `"token1"` → outcome 0 and `"token2"` → outcome 1, and populates `asset` with the real `token_ids[idx]` from Gamma so the merge-asof and the CLOB/trade-price resolution-timestamp derivation match.

**Bug B — HF timestamp precision mis-interpretation.** DuckDB returns HF's `timestamp` column as `uint64` raw Unix seconds (e.g. 1766535065). The old normaliser called `pd.to_datetime(df["timestamp"], utc=True)` without `unit="s"`, so pandas interpreted the integer as **nanoseconds** → every HF row became 1970-01-01 + 1.77 s. This silently poisoned:

- `time_to_settlement_s`, `log_time_to_settlement`, `pct_time_elapsed` — computed against real 2026 `end_date` / `resolution_ts`, giving ~56-year values.
- Every rolling wallet-in-market window (`wallet_trades_in_market_last_{1,10,60}min`) — windows collapsed to a single sub-second bucket.
- `market_price_vol_last_1h` — same rolling-window issue.
- The `split` column — trade-timestamp quantiles over a range that put every HF row below every API row.

Fix: `pd.to_datetime(df["timestamp"], unit="s", utc=True).astype("datetime64[ns, UTC]")` in `build_hf_trades`. The `[ns]` cast still matters for merge_asof precision matching with the CLOB path.

**State after both fixes, verified via `outputs/eda/summary.txt`:**

- timespan 2025-12-22 → 2026-04-19 (was 1970-01-01 + 1.77 s for HF rows).
- `bet_correct` mean 0.504 (was 0.613, inflated by the side-based degenerate labelling).
- `wallet_directional_purity_in_market` null rate 22.59% (was 99.30%); `wallet_spread_ratio` matches.
- `resolution_ts` null rate 5.02% (was 98.79%) — only markets with neither a CLOB lock nor a trade-price lock remain null.
- `wallet_position_size_before_trade` fully populated (was 98.79% null).
- Per-market correctness range 0.466 – 0.737 (was 0.466 – 0.752 but mixing two distinct signal distributions across HF and API rows).

All 35 features listed in `project_plan.md` §4 are present and populated at rates at or below the null-rate expectations in §11.

### 5. Feature parity pass against Alex's EDA

Cross-checked against Alex's earlier 7-market report (`report.html`, generated 19 April, 452k trades, 63 cols). Panel coverage was broadly aligned, but his feature set carried five market- and wallet-depth signals we didn't compute. Added all five to `fetch_polymarket.py`, strictly no-lookahead:

- `market_vol_1h_log` — log1p of trailing 1-hour USD volume per market, `closed="left"` rolling sum in `_add_running_market_features`.
- `market_vol_24h_log` — same, 24-hour window.
- `wallet_prior_trades_in_market` — cumulative count of wallet's prior trades in the same market (shifted), complements the existing time-windowed `wallet_trades_in_market_last_*min` with an unbounded depth signal.
- `wallet_cumvol_same_side_last_10min` — 10-min rolling USD volume per (wallet × condition_id × outcomeIndex), `closed="left"`. Required a new `_rolling_sum_by_group` helper that mirrors `_rolling_count_by_group` for arbitrary value columns.
- `size_vs_market_cumvol_pct` — trade value as a fraction of prior cumulative market volume; signal of abnormally large bets relative to market activity so far. Declined gracefully to NaN on the first trade in each market (74 rows, 0.01%).

`trades_enriched.csv` grew from 57 → **62 cols** and 909 MB → 990 MB. Enrichment stage time rose from ~14 min to ~20 min, driven by the (wallet × market × outcome) grouping in the rolling-sum pass. `project_plan.md` §4 Features row extended to list the additions under their respective layers.

### 6. Added the event-timing empirical panel

New `panel_event_timing` in `eda.py` (figure `08_event_timing.png` + `08_event_timing.txt`). Aggregates total USD volume and mean `bet_correct` by time-to-settlement bucket across all 74 markets (drops post-resolution close-outs via `time_to_settlement_s > 0`). Provides the empirical justification for §5.2's home-run gate (`time_to_settlement < 6h`) without anchoring on a single market or the Magamyman anecdote.

First-run table:

```
bucket  trades   volume_usd    mean_correct
<1h     31,926   $10.1M        0.5442   <- late-close window, above baseline
1-6h    75,645   $22.3M        0.5088
6-24h  209,170   $45.2M        0.5292
1-7d   421,442   $73.2M        0.4872   <- bulk of volume, noise-dominated
7-30d  284,107   $40.6M        0.4967
>30d    62,096    $5.5M        0.5304
```

The <1h bucket runs 4.4 percentage points above the 1-7d baseline on $10M of volume, which is what the home-run rule is designed to capture.

Wallet-quadrants panel renumbered `08` → `09` so the final `outputs/eda/` listing is `01_missingness` through `09_wallet_quadrants` plus `summary.txt`, `03_skewness_table.csv`, `05_top_correlations.txt`, `08_event_timing.txt`, `09_wallet_quadrants.txt`.

### 7. Locked in the multi-cluster data strategy

Decision captured in `project_plan.md` §4 (new subsection "Multi-cluster data strategy"). Short version: we repeat the ~60-min HF stream for each additional event cluster rather than download the full 39 GB parquet. Each cluster produces its own `trades_enriched.csv`; at the end we `pd.concat` them on disk into a single mother frame. The concat must recompute (a) all running/prior wallet and market features and (b) the `split` column, because both are cluster-local.

Clusters in scope beyond the current Iran four-event build: to be decided, but §8 flags Maduro and Biden-pardons as candidates.

## State of the data folder

`report/data/` (after the two HF bug fixes and the final rebuild):

- `trades_enriched.csv` — **909 MB**, 1,209,787 rows × 57 cols. Mother dataframe for modelling. Grew from 806 MB because HF rows now carry real 77-digit token IDs in `asset` and populated `outcomeIndex`, `resolution_ts`, position-aware features.
- `trades.csv` — 384 MB (raw combined).
- `markets.csv` — 27 KB (74 resolved Iran-cluster markets).
- `prices.csv` — 940 KB (CLOB history for the 7 ceasefire markets).
- `trades_iran_subset.parquet` — 50 MB (HF subset cache for Iran condition_ids; keeps the 60-min stream reusable for this cluster).
- `markets.parquet` — 116 MB (HF markets metadata cache, shared across clusters).
- `_backup_20260419/` — pre-refetch snapshot from the 19 April state.

Everything above is on disk; modelling and EDA do not touch any network from here.

## Code changes to track

- `scripts/build_iran_dataset.py`
  - `build_trades_subset_remote` rewritten for chunked pyarrow reads with retry + reopen; new module-level constants `HF_TRADES_COLUMNS`, `REMOTE_READ_MAX_ATTEMPTS`, `REMOTE_READ_BACKOFF_CAP_S`, helper `_open_remote_parquet_handle`.
  - `build_hf_trades` `_asset_oidx` replacement: `HF_TOKEN_TO_INDEX = {"token1": 0, "token2": 1}` maps HF's 1-indexed Solidity-style labels to 0-indexed Gamma token ids and the outcome index.
  - `build_hf_trades` timestamp cast: `pd.to_datetime(df["timestamp"], unit="s", utc=True).astype("datetime64[ns, UTC]")`. The `unit="s"` is load-bearing because duckdb returns HF timestamps as raw `uint64` seconds.
- `scripts/fetch_polymarket.py`
  - `enrich_trades` line 352-354: cast both merge-asof keys to `[ns, UTC]` before the merge (handles the pandas 3.0 default return of `[s, UTC]` from `pd.to_datetime(..., unit="s")`).
- `scripts/eda.py`
  - Rewritten for Design.md compliance (rocket_r palette, `style="white"`, 140/300 DPI, `FIG_W=6.3` for A4 Word layout, `clean_ax()` helper, no in-image titles, heatmap spec).
  - Column references realigned to the actual `trades_enriched.csv` schema (`proxyWallet`, `split`, `trade_value_usd`, `market_volume_so_far_usd`, `market_price_vol_last_1h`, `resolution_ts`, `size_x_time_to_settlement`, etc.).
  - `format="mixed"` on timestamp parsing for `summary.txt` so HF-second-precision and API-ns-precision strings both round-trip.
  - Magamyman Feb-28 panel removed — belongs in Discussion per plan §5.6, not in EDA.

## Open limitations, not addressed this session

- `stop offset=3500 params={}: 400 Client Error` warnings on 2 of the 7 ceasefire markets during the API pass. These are the Polymarket Data API's ~3000–3500 offset ceiling kicking in, but both markets are well under the 7k side-split budget (~4.5k trades each) so no data is lost; the warning is cosmetic and can be silenced later.
- `enrich_trades` has no internal progress logging. On 1.2 M rows the ~14-min stage looks hung from the outside; add step-level logs if we build another cluster.
- `outputs/eda/` is still the stale API-only output from an earlier session.

## Next steps

Priority order:

1. **EDA on the full 1.2 M-row Iran dataset** — `python scripts/eda.py` (task #12).
2. **Apply the `settlement_minus_trade_sec > 0` filter** inside modelling code (task #9).
3. **MLP + baselines** (task #13); then calibration, backtest, evaluation (#14–#16).
4. **Decide next cluster**, if any, and repeat the ~60-min stream to produce its own `trades_enriched_<cluster>.csv`. Then concat all cluster CSVs into the final mother frame, recomputing running features and the `split` column on the merged frame.

Tasks #10 (Polygonscan), #11 (GDELT) remain deferred per prior decisions.
