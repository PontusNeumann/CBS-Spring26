"""Patch `data/03_consolidated_dataset.csv` with a causal replacement for
`wallet_prior_win_rate`, plus its matching missingness indicator.

Motivation (see `data-pipeline-issues.md` P0-9). The current
`wallet_prior_win_rate` sums `bet_correct` over ALL prior trades by a
wallet. In our 74-market build, every `bet_correct` is populated (markets
have all resolved), but at trade time *t* the value is only observable for
priors whose `resolution_ts < t`. The feature therefore peeks at future
outcomes: Pearson r with the target is **+0.367** (the strongest linear
correlate) and drops to **+0.236** once post-t priors are excluded — a
third of the signal is future information.

Columns added (the canonical names are also produced by the updated
`_add_running_wallet_features` in `01_polymarket_api.py`):

  * ``wallet_prior_win_rate_causal`` — cumulative mean of bet_correct over
    priors with ``resolution_ts < current timestamp``. NaN when the wallet
    has no resolved priors yet (structural missingness).
  * ``wallet_has_resolved_priors`` — 1 iff ≥1 resolved prior exists at *t*.

The existing leaky `wallet_prior_win_rate` is retained in the CSV for
audit. Any modelling script should exclude it and feed the causal variant
instead.

Backup to `data/03_consolidated_dataset.pre_causal_win_rate.csv`.

Usage:
  python scripts/19_add_causal_win_rate.py
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "03_consolidated_dataset.csv"
BACKUP = ROOT / "data" / "03_consolidated_dataset.pre_causal_win_rate.csv"

WALLET_COL = "proxyWallet"


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
    for c in (WALLET_COL, "timestamp", "resolution_ts", "bet_correct"):
        if c not in df.columns:
            raise SystemExit(f"source column missing: {c}")

    if BACKUP.exists():
        print(f"  backup {BACKUP.name} already present — not overwriting")
    else:
        print(f"writing backup {BACKUP.name}...")
        shutil.copy2(CSV, BACKUP)
        print(f"  backup written ({time.monotonic() - t0:.1f}s)")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["resolution_ts"] = pd.to_datetime(
        df["resolution_ts"], utc=True, errors="coerce"
    )

    df = df.sort_values(
        [WALLET_COL, "timestamp"], kind="mergesort"
    ).reset_index(drop=True)

    ts_ns = df["timestamp"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    rts_ns = df["resolution_ts"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    bc_arr = pd.to_numeric(df["bet_correct"], errors="coerce").to_numpy()
    wallet_arr = df[WALLET_COL].to_numpy()

    n = len(df)
    causal_wr = np.full(n, np.nan, dtype=np.float64)
    has_resolved = np.zeros(n, dtype=np.int8)

    print("walking wallets and computing causal win rate...")
    t1 = time.monotonic()
    cur_wallet = None
    hist_rts: list[int] = []
    hist_bc: list[float] = []
    for i in range(n):
        w = wallet_arr[i]
        if w != cur_wallet:
            cur_wallet = w
            hist_rts = []
            hist_bc = []
        if hist_rts:
            rts_np = np.asarray(hist_rts, dtype=np.int64)
            bc_np = np.asarray(hist_bc, dtype=np.float64)
            mask = (~np.isnan(bc_np)) & (rts_np < ts_ns[i])
            if mask.any():
                causal_wr[i] = float(bc_np[mask].mean())
                has_resolved[i] = 1
        hist_rts.append(int(rts_ns[i]))
        hist_bc.append(float(bc_arr[i]))
    print(f"  done ({time.monotonic() - t1:.1f}s)")

    if "wallet_prior_win_rate_causal" in df.columns:
        df = df.drop(columns=["wallet_prior_win_rate_causal"])
    if "wallet_has_resolved_priors" in df.columns:
        df = df.drop(columns=["wallet_has_resolved_priors"])
    df["wallet_prior_win_rate_causal"] = causal_wr
    df["wallet_has_resolved_priors"] = has_resolved

    # Diagnostic — correlation with target (full + resolved-only subset)
    tgt = pd.to_numeric(df["bet_correct"], errors="coerce")
    mask_ok = tgt.notna()
    r_leaky = df.loc[mask_ok, "wallet_prior_win_rate"].corr(tgt[mask_ok])
    r_causal = df.loc[mask_ok, "wallet_prior_win_rate_causal"].corr(tgt[mask_ok])
    print("\ndiagnostic:")
    print(
        f"  Pearson r(leaky   wallet_prior_win_rate        , bet_correct) = {r_leaky:.4f}"
    )
    print(
        f"  Pearson r(causal  wallet_prior_win_rate_causal , bet_correct) = {r_causal:.4f}"
    )
    print(
        f"  leak-driven correlation component: {r_leaky - r_causal:+.4f}"
    )
    print(
        f"  causal NaN rate: {float(np.isnan(causal_wr).mean()) * 100:.2f}%  "
        f"has_resolved_priors rate: {float(has_resolved.mean()) * 100:.2f}%"
    )

    print("\nwriting CSV...")
    t2 = time.monotonic()
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df["resolution_ts"] = df["resolution_ts"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
    if "deadline_ts" in df.columns:
        df["deadline_ts"] = pd.to_datetime(
            df["deadline_ts"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df.to_csv(CSV, index=False)
    print(
        f"  wrote {CSV.name} ({len(df):,} rows × {len(df.columns)} cols, "
        f"{time.monotonic() - t2:.1f}s)"
    )
    print(f"total: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
