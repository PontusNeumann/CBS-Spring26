"""Patch `data/03_consolidated_dataset.csv` to remove two confirmed
future-looking leaks without a full pipeline rebuild.

Leaks fixed (canonical definitions now live in `01_polymarket_api.py`):

  1. `wallet_is_whale_in_market`
     Old: `cum_vol_wm >= p95(final wallet volumes per market)`. The
     threshold was the 95th percentile of end-of-market wallet totals, so
     every trade was scored against a constant that depended on trades in
     the future.
     New: strictly causal expanding p95. At each trade i in market m, the
     threshold is the 95th percentile of per-wallet running volumes using
     only trades with timestamp strictly less than i's timestamp.

  2. `time_to_settlement_s`, `log_time_to_settlement`, `pct_time_elapsed`,
     `size_x_time_to_settlement`
     Old: built from `resolution_ts` (the post-hoc CLOB lock timestamp —
     not knowable at trade time). 95% of rows used the leaky source and
     99% of those disagreed with `end_date` by a median of −0.007 days
     (quartiles ±10–73 days off).
     New: built from `end_date` only (the advertised market deadline,
     published at market creation and therefore known at trade time).
     `settlement_minus_trade_sec` keeps `resolution_ts` upstream because
     its sole role is the §4 post-resolution filter; it is not a model
     feature.

Idempotent: re-running replaces the five columns in place. A pre-patch
backup is written to `data/03_consolidated_dataset.pre_causal.csv`.

Usage:
  python scripts/16_fix_leakage.py
"""
from __future__ import annotations

import importlib.util as _ilu
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "03_consolidated_dataset.csv"
BACKUP = ROOT / "data" / "03_consolidated_dataset.pre_causal.csv"

_spec = _ilu.spec_from_file_location(
    "polymarket_api", ROOT / "scripts" / "01_polymarket_api.py"
)
_fp = _ilu.module_from_spec(_spec)
sys.modules["polymarket_api"] = _fp
_spec.loader.exec_module(_fp)

WALLET_COL = "proxyWallet"
WHALE_QUANTILE = _fp.WHALE_QUANTILE


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}")

    t0 = time.monotonic()
    print(f"reading {CSV.name}...")
    df = pd.read_csv(CSV, low_memory=False)
    print(
        f"  {len(df):,} rows × {len(df.columns)} cols "
        f"({time.monotonic() - t0:.1f}s)"
    )

    required = {
        "proxyWallet", "condition_id", "timestamp", "end_date",
        "trade_value_usd", "log_size",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"source columns missing: {missing}")

    if BACKUP.exists():
        print(f"  backup {BACKUP.name} already present — not overwriting")
    else:
        print(f"writing backup {BACKUP.name}...")
        shutil.copy2(CSV, BACKUP)
        print(f"  backup written ({time.monotonic() - t0:.1f}s)")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")

    # ------------------------------------------------------------------
    # 1. Time features from end_date (causal)
    # ------------------------------------------------------------------
    print("recomputing time_to_settlement_s, log_time_to_settlement, "
          "pct_time_elapsed from end_date...")
    t1 = time.monotonic()

    old_tts = pd.to_numeric(df.get("time_to_settlement_s"), errors="coerce")
    old_pct = pd.to_numeric(df.get("pct_time_elapsed"), errors="coerce")

    df["time_to_settlement_s"] = (df["end_date"] - df["timestamp"]).dt.total_seconds()
    df["log_time_to_settlement"] = np.log1p(
        df["time_to_settlement_s"].clip(lower=0).fillna(0)
    )
    market_start = df.groupby("condition_id")["timestamp"].transform("min")
    life_total = (df["end_date"] - market_start).dt.total_seconds()
    life_elapsed = (df["timestamp"] - market_start).dt.total_seconds()
    df["pct_time_elapsed"] = (life_elapsed / life_total.where(life_total > 0)).clip(0, 1)
    df["size_x_time_to_settlement"] = (
        pd.to_numeric(df["log_size"], errors="coerce") * df["log_time_to_settlement"]
    )

    new_tts = df["time_to_settlement_s"]
    print(
        f"  time_to_settlement_s:   mean change = "
        f"{(new_tts - old_tts).mean():.0f}s   "
        f"rows changed = {((new_tts - old_tts).abs() > 1).sum():,}"
    )
    new_pct = df["pct_time_elapsed"]
    print(
        f"  pct_time_elapsed:       mean change = "
        f"{(new_pct - old_pct).mean():.4f}   "
        f"rows changed = {((new_pct - old_pct).abs() > 1e-4).sum():,}"
    )
    print(f"  done ({time.monotonic() - t1:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Causal whale flag (expanding p95 per market)
    # ------------------------------------------------------------------
    print(
        f"recomputing wallet_is_whale_in_market (expanding p95, "
        f"q={WHALE_QUANTILE})..."
    )
    t2 = time.monotonic()
    old_whale = pd.to_numeric(df.get("wallet_is_whale_in_market"), errors="coerce")
    tv = pd.to_numeric(df["trade_value_usd"], errors="coerce").fillna(0.0)
    df["wallet_is_whale_in_market"] = _fp._causal_whale_flag(
        df, WALLET_COL, tv, WHALE_QUANTILE
    ).astype("int64")
    new_whale = df["wallet_is_whale_in_market"]
    flipped = int(((old_whale.fillna(0).astype(int) == 1) & (new_whale == 0)).sum())
    kept = int(((old_whale.fillna(0).astype(int) == 1) & (new_whale == 1)).sum())
    added = int(((old_whale.fillna(0).astype(int) == 0) & (new_whale == 1)).sum())
    print(f"  flag rate: old={old_whale.mean() * 100:.2f}%  "
          f"new={new_whale.mean() * 100:.2f}%")
    print(f"  1→0 transitions (leaky flags removed): {flipped:,}")
    print(f"  1→1 kept (genuinely above running p95):  {kept:,}")
    print(f"  0→1 added (became whale causally):       {added:,}")
    print(f"  done ({time.monotonic() - t2:.1f}s)")

    # ------------------------------------------------------------------
    # Sanity: first (wallet, market) trade should now never be whale=1,
    # because prior running vol is 0 and p95 > 0 after warm-up.
    # ------------------------------------------------------------------
    first_wm = (
        df.sort_values([WALLET_COL, "condition_id", "timestamp"])
        .groupby([WALLET_COL, "condition_id"], as_index=False)
        .head(1)
    )
    n_first_whale = int((first_wm["wallet_is_whale_in_market"] == 1).sum())
    print(
        f"  sanity: whale=1 on the first (wallet, market) trade: "
        f"{n_first_whale:,} / {len(first_wm):,}  "
        f"(expected 0)"
    )

    # ------------------------------------------------------------------
    # Write back
    # ------------------------------------------------------------------
    print("writing CSV...")
    t3 = time.monotonic()
    df["end_date"] = df["end_date"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df.to_csv(CSV, index=False)
    print(
        f"  wrote {CSV.name} "
        f"({len(df):,} rows × {len(df.columns)} cols, "
        f"{time.monotonic() - t3:.1f}s)"
    )
    print(f"total: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
