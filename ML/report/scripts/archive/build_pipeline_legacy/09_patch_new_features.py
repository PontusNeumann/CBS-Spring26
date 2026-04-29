"""Patch `data/03_consolidated_dataset.csv` with three new features without
triggering a full pipeline rebuild.

Adds (all strictly no-lookahead):
  - market_buy_share_running      (market-context)
  - wallet_median_gap_in_market   (bet-slicing cluster)
  - size_vs_market_avg            (interactions)

The canonical definitions live in `01_polymarket_api.py`; this script
reproduces them against the already-enriched CSV to avoid re-running the
20+ minute full build for three columns.

Idempotent: re-running replaces the three columns in place.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "03_consolidated_dataset.csv"
BACKUP = ROOT / "data" / "03_consolidated_dataset.pre09.csv"


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}")

    t0 = time.monotonic()
    print(f"reading {CSV.name}...")
    df = pd.read_csv(CSV, low_memory=False)
    print(f"  {len(df):,} rows x {len(df.columns)} cols"
          f"  ({time.monotonic()-t0:.1f}s)")

    if not BACKUP.exists():
        print(f"writing backup {BACKUP.name}...")
        df.to_csv(BACKUP, index=False)
        print(f"  backup written ({time.monotonic()-t0:.1f}s)")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # ---------------------------------------------------------------
    # 1. market_buy_share_running
    # ---------------------------------------------------------------
    print("adding market_buy_share_running...")
    df = df.sort_values(["condition_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    is_buy = df["side"].astype(str).str.upper().eq("BUY").astype("int64")
    prior_buys = is_buy.groupby(df["condition_id"]).cumsum() - is_buy
    n_prior = pd.to_numeric(df["market_trade_count_so_far"], errors="coerce")
    df["market_buy_share_running"] = prior_buys / n_prior.where(n_prior > 0)
    print(f"  done ({time.monotonic()-t0:.1f}s)")

    # ---------------------------------------------------------------
    # 2. wallet_median_gap_in_market
    # ---------------------------------------------------------------
    print("adding wallet_median_gap_in_market...")
    wallet_col = "proxyWallet"
    df = df.sort_values([wallet_col, "condition_id", "timestamp"],
                        kind="mergesort").reset_index(drop=True)
    gaps_sec = (df.groupby([wallet_col, "condition_id"])["timestamp"]
                  .diff().dt.total_seconds())
    df["wallet_median_gap_in_market"] = (
        gaps_sec.groupby([df[wallet_col], df["condition_id"]])
                .transform(lambda s: s.expanding().median().shift(1))
    )
    print(f"  done ({time.monotonic()-t0:.1f}s)")

    # ---------------------------------------------------------------
    # 3. size_vs_market_avg
    # ---------------------------------------------------------------
    print("adding size_vs_market_avg...")
    tv = pd.to_numeric(df["trade_value_usd"], errors="coerce").fillna(0.0)
    mv_prior = pd.to_numeric(df["market_volume_so_far_usd"], errors="coerce")
    mc_prior = pd.to_numeric(df["market_trade_count_so_far"], errors="coerce")
    avg_size_market = mv_prior / mc_prior.where(mc_prior > 0)
    df["size_vs_market_avg"] = tv / avg_size_market.where(avg_size_market > 0)
    print(f"  done ({time.monotonic()-t0:.1f}s)")

    # ---------------------------------------------------------------
    # Restore the original split-order by timestamp only, then save.
    # (`split` was written by the original build in trade-timestamp order;
    # the feature values are per-row so they don't care about final order,
    # but keeping timestamp sort makes the CSV easier to diff.)
    # ---------------------------------------------------------------
    print("sorting and writing...")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df.to_csv(CSV, index=False)
    print(f"  wrote {CSV.name}  ({len(df):,} rows x {len(df.columns)} cols,"
          f" {time.monotonic()-t0:.1f}s)")

    # Sanity report
    for col in ("market_buy_share_running",
                "wallet_median_gap_in_market",
                "size_vs_market_avg"):
        s = pd.to_numeric(df[col], errors="coerce")
        print(f"  {col:32s} non-null={s.notna().sum():,}"
              f"  min={s.min():.3g}  median={s.median():.3g}  max={s.max():.3g}")


if __name__ == "__main__":
    main()
