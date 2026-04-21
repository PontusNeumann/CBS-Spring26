"""Add Layer 6 on-chain identity features via bisect on per-wallet timestamp arrays.

Reads data/03_trades_features.csv (mother dataframe, 65 cols) and
data/wallet_enrichment.parquet (per-wallet scalars + timestamp arrays from
03_enrich_wallets.py), and emits data/03_trades_features.csv with 9 new
Layer 6 columns appended (final width: 74 cols).

Features added (strictly causal — each uses only wallet events with
timestamp strictly before the trade timestamp):
  - wallet_enriched                 1 if wallet was found in enrichment set
  - wallet_polygon_age_at_t_days    (trade_ts − polygon_first_tx_ts) / 86400
  - wallet_polygon_nonce_at_t       # outbound tx strictly before trade
  - wallet_log_polygon_nonce_at_t   log1p of nonce
  - wallet_n_inbound_at_t           # inbound tx strictly before trade
  - wallet_log_n_inbound_at_t       log1p of inbound count
  - wallet_n_cex_deposits_at_t      # CEX USDC deposits strictly before trade
  - wallet_cex_usdc_cumulative_at_t sum of CEX USDC received strictly before trade
  - wallet_log_cex_usdc_cum         log1p of cumulative CEX USDC
  - days_from_first_usdc_to_t       (trade_ts − first_usdc_inbound_ts) / 86400 if >= 0
  - wallet_funded_by_cex            time-invariant flag (1 if wallet ever funded by CEX)
  - wallet_funded_by_cex_scoped     1 iff funded_by_cex AND first_usdc_inbound_ts < trade_ts

A pre-integration backup is written to data/03_trades_features.pre11.csv so
the patch can be reverted in one file copy if downstream validation flags an
issue.

Usage:
  python scripts/11_add_layer6.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data" / "03_trades_features.csv"
ENRICH_IN = ROOT / "data" / "wallet_enrichment.parquet"
BACKUP_CSV = ROOT / "data" / "03_trades_features.pre11.csv"

WALLET_COL = "proxyWallet"
TIMESTAMP_COL = "timestamp"

NEW_COLS = [
    "wallet_enriched",
    "wallet_polygon_age_at_t_days",
    "wallet_polygon_nonce_at_t",
    "wallet_log_polygon_nonce_at_t",
    "wallet_n_inbound_at_t",
    "wallet_log_n_inbound_at_t",
    "wallet_n_cex_deposits_at_t",
    "wallet_cex_usdc_cumulative_at_t",
    "wallet_log_cex_usdc_cum",
    "days_from_first_usdc_to_t",
    "wallet_funded_by_cex",
    "wallet_funded_by_cex_scoped",
]


def main() -> None:
    t0 = time.time()

    print(f"loading mother dataframe...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"  {len(df):,} rows × {len(df.columns)} cols in {time.time() - t0:.1f}s")

    existing = [c for c in NEW_COLS if c in df.columns]
    if existing:
        print(
            f"WARNING: {len(existing)} Layer 6 columns already present "
            f"({existing[:3]}...). Overwriting."
        )
        df = df.drop(columns=existing)

    print(f"loading wallet enrichment...")
    t1 = time.time()
    enrich = pd.read_parquet(ENRICH_IN)
    enrich_ok = enrich[enrich["fetch_status"] == "ok"].copy()
    print(
        f"  {len(enrich):,} total, {len(enrich_ok):,} successful "
        f"({len(enrich_ok) / max(1, len(enrich)) * 100:.1f}%) in {time.time() - t1:.1f}s"
    )

    # Index by wallet for fast lookup; convert arrays to numpy once
    print(f"indexing enrichment by wallet...")
    t2 = time.time()
    wallet_idx: dict[str, dict] = {}
    for _, r in enrich_ok.iterrows():
        wallet_idx[r["wallet"]] = {
            "polygon_first_tx_ts": int(r["polygon_first_tx_ts"])
            if not pd.isna(r["polygon_first_tx_ts"])
            else None,
            "first_usdc_inbound_ts": int(r["first_usdc_inbound_ts"])
            if not pd.isna(r["first_usdc_inbound_ts"])
            else None,
            "funded_by_cex": int(r["funded_by_cex"])
            if not pd.isna(r["funded_by_cex"])
            else 0,
            "outbound_ts": np.asarray(r["outbound_ts"], dtype=np.int64)
            if len(r["outbound_ts"])
            else np.array([], dtype=np.int64),
            "inbound_ts": np.asarray(r["inbound_ts"], dtype=np.int64)
            if len(r["inbound_ts"])
            else np.array([], dtype=np.int64),
            "cex_deposit_ts": np.asarray(r["cex_deposit_ts"], dtype=np.int64)
            if len(r["cex_deposit_ts"])
            else np.array([], dtype=np.int64),
            "cex_deposit_amounts_cumsum": np.cumsum(
                np.asarray(r["cex_deposit_amounts_usd"], dtype=np.float64)
            )
            if len(r["cex_deposit_amounts_usd"])
            else np.array([], dtype=np.float64),
        }
    print(f"  indexed {len(wallet_idx):,} wallets in {time.time() - t2:.1f}s")

    # Causal per-trade bisect
    print(f"computing per-trade Layer 6 features...")
    t3 = time.time()
    n = len(df)
    polygon_age_at_t = np.full(n, np.nan, dtype=np.float64)
    polygon_nonce_at_t = np.zeros(n, dtype=np.int32)
    n_inbound_at_t = np.zeros(n, dtype=np.int32)
    n_cex_deposits_at_t = np.zeros(n, dtype=np.int32)
    cex_usdc_cum_at_t = np.zeros(n, dtype=np.float64)
    days_from_first_usdc = np.full(n, np.nan, dtype=np.float64)
    funded_by_cex_static = np.zeros(n, dtype=np.int8)
    funded_by_cex_scoped = np.zeros(n, dtype=np.int8)
    wallet_enriched = np.zeros(n, dtype=np.int8)

    wallets = df[WALLET_COL].values
    timestamps = df[TIMESTAMP_COL].values.astype(np.int64)

    for i in range(n):
        w = wallets[i]
        t_trade = timestamps[i]
        info = wallet_idx.get(w)
        if info is None:
            continue
        wallet_enriched[i] = 1

        if info["polygon_first_tx_ts"] is not None:
            age_s = max(0, t_trade - info["polygon_first_tx_ts"])
            polygon_age_at_t[i] = age_s / 86400.0

        ob = info["outbound_ts"]
        if len(ob) > 0:
            polygon_nonce_at_t[i] = int(np.searchsorted(ob, t_trade, side="left"))

        ib = info["inbound_ts"]
        if len(ib) > 0:
            n_inbound_at_t[i] = int(np.searchsorted(ib, t_trade, side="left"))

        cd = info["cex_deposit_ts"]
        if len(cd) > 0:
            k = int(np.searchsorted(cd, t_trade, side="left"))
            n_cex_deposits_at_t[i] = k
            if k > 0:
                cex_usdc_cum_at_t[i] = float(info["cex_deposit_amounts_cumsum"][k - 1])

        if info["first_usdc_inbound_ts"] is not None:
            d = (t_trade - info["first_usdc_inbound_ts"]) / 86400.0
            if d >= 0:
                days_from_first_usdc[i] = d

        funded_by_cex_static[i] = info["funded_by_cex"]
        if (
            info["funded_by_cex"] == 1
            and info["first_usdc_inbound_ts"] is not None
            and info["first_usdc_inbound_ts"] < t_trade
        ):
            funded_by_cex_scoped[i] = 1

        if (i + 1) % 100_000 == 0:
            print(
                f"  {i + 1:,} / {n:,} rows "
                f"({(i + 1) / n * 100:.1f}%, {time.time() - t3:.0f}s)"
            )

    print(f"  done in {time.time() - t3:.1f}s")

    df["wallet_enriched"] = wallet_enriched
    df["wallet_polygon_age_at_t_days"] = polygon_age_at_t
    df["wallet_polygon_nonce_at_t"] = polygon_nonce_at_t
    df["wallet_log_polygon_nonce_at_t"] = np.log1p(polygon_nonce_at_t)
    df["wallet_n_inbound_at_t"] = n_inbound_at_t
    df["wallet_log_n_inbound_at_t"] = np.log1p(n_inbound_at_t)
    df["wallet_n_cex_deposits_at_t"] = n_cex_deposits_at_t
    df["wallet_cex_usdc_cumulative_at_t"] = cex_usdc_cum_at_t
    df["wallet_log_cex_usdc_cum"] = np.log1p(cex_usdc_cum_at_t)
    df["days_from_first_usdc_to_t"] = days_from_first_usdc
    df["wallet_funded_by_cex"] = funded_by_cex_static
    df["wallet_funded_by_cex_scoped"] = funded_by_cex_scoped

    # Fill NaNs on missing/failed wallets with a sentinel (0) + the
    # wallet_enriched flag carries the presence information.
    fill_zero = [
        "wallet_polygon_age_at_t_days",
        "days_from_first_usdc_to_t",
    ]
    for c in fill_zero:
        df[c] = df[c].fillna(0.0)

    print(f"\nLayer 6 feature summary:")
    print(df[NEW_COLS].describe().T[["mean", "50%", "min", "max"]].round(3))

    print(f"\nwallet coverage:")
    enriched_cnt = int(df["wallet_enriched"].sum())
    print(
        f"  enriched (data present): {enriched_cnt:,} / {len(df):,} "
        f"({enriched_cnt / len(df) * 100:.1f}%)"
    )
    print(f"  funded_by_cex (static): {int(df['wallet_funded_by_cex'].sum()):,} rows")
    print(
        f"  funded_by_cex (scoped):  {int(df['wallet_funded_by_cex_scoped'].sum()):,} rows"
    )

    print(f"\nbacking up to {BACKUP_CSV.name} and writing patched CSV...")
    t4 = time.time()
    shutil.copy2(FEATURES_CSV, BACKUP_CSV)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"  done in {time.time() - t4:.1f}s")

    print(
        f"\nwrote {FEATURES_CSV.name} — shape {df.shape} in {time.time() - t0:.1f}s total"
    )


if __name__ == "__main__":
    main()
