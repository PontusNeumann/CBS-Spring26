"""
16_patch_resolution_ts.py
One-shot backfill: recompute `resolution_ts` (and the derived
`settlement_minus_trade_sec`) for markets in `03_consolidated_dataset.csv`
whose resolution_ts is NaT.

Why this exists
---------------
The current CSV has 3 markets with `resolution_ts = NaT`:
  - `US strikes Iran by February 27, 2026?` (46,657 trades, in training cohort)
  - `US strikes Iran by February 8, 2026?`  (5,389 trades)
  - `US strikes Iran by March 3, 2026?`     (8,675 trades)

The upstream `_first_lock_timestamp` in `01_polymarket_api.py` was rejecting
valid locks when any outlier trade later dipped below LOCK_UNLOCK_FLOOR. The
patched version (this session) uses a robust median check. This script applies
the same logic on-the-fly to backfill the current CSV without requiring a full
pipeline rerun.

After the next full rebuild (where `01_polymarket_api.py`'s fix is applied),
this script is obsolete.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "03_consolidated_dataset.csv"
BACKUP = ROOT / "data" / "03_consolidated_dataset.pre_rts_patch.csv"

LOCK_THRESHOLD = 0.995
LOCK_UNLOCK_FLOOR = 0.9


def robust_first_lock(prices: np.ndarray) -> int | None:
    """Index of the first price >= LOCK_THRESHOLD whose subsequent-prices
    median stays >= LOCK_UNLOCK_FLOOR. Tolerates outliers. Returns None if no
    candidate satisfies."""
    hi = np.flatnonzero(prices >= LOCK_THRESHOLD)
    if hi.size == 0:
        return None
    for idx in hi:
        if np.median(prices[idx:]) >= LOCK_UNLOCK_FLOOR:
            return int(idx)
    return None


def derive_resolution_for_market(mdf: pd.DataFrame) -> pd.Timestamp | None:
    """Compute a resolution_ts for a single market's trade frame using the
    winning-side price-lock heuristic. Requires `outcomeIndex`,
    `winning_outcome_index`, `price`, and `timestamp` columns."""
    win_idx = mdf["winning_outcome_index"].iloc[0]
    if pd.isna(win_idx):
        return None
    winning_side = mdf[mdf["outcomeIndex"] == int(win_idx)]
    if winning_side.empty:
        return None
    ws = winning_side.sort_values("ts").reset_index(drop=True)
    prices = pd.to_numeric(ws["price"], errors="coerce").to_numpy()
    idx = robust_first_lock(prices)
    if idx is None:
        return None
    return ws.iloc[idx]["ts"]


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing: {CSV}")

    print(f"[load] {CSV} ({CSV.stat().st_size / 1e6:.0f} MB)")
    df = pd.read_csv(CSV, low_memory=False)
    print(f"[load] {len(df):,} rows x {df.shape[1]} cols")

    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["rts_parsed"] = pd.to_datetime(df["resolution_ts"], utc=True, errors="coerce")

    nat_markets = (
        df.groupby("condition_id")
        .agg(
            question=("question", "first"),
            rts=("rts_parsed", "first"),
            n=("condition_id", "size"),
        )
        .reset_index()
    )
    nat_markets = nat_markets[nat_markets["rts"].isna()].copy()
    if nat_markets.empty:
        print("[skip] no markets with NaT resolution_ts; nothing to patch")
        return
    print(f"[found] {len(nat_markets)} markets with NaT resolution_ts:")
    for _, row in nat_markets.iterrows():
        print(f"         [{int(row['n']):>7,}]  {row['question']}")

    print(f"\n[backup] writing {BACKUP.name}")
    df.drop(columns=["ts", "rts_parsed"]).to_csv(BACKUP, index=False)

    # Compute resolution_ts per NaT market via robust lock heuristic
    patches: dict[str, pd.Timestamp] = {}
    unresolved: list[str] = []
    for cid in nat_markets["condition_id"]:
        mdf = df[df["condition_id"] == cid]
        rts = derive_resolution_for_market(mdf)
        q = mdf["question"].iloc[0]
        if rts is None:
            unresolved.append(q)
            print(
                f"[fail] {q}: could not derive resolution_ts (no robust lock in data window)"
            )
        else:
            patches[cid] = rts
            print(f"[ok]   {q}: resolution_ts = {rts}")

    if not patches:
        print("\n[skip] no markets could be patched; nothing written")
        return

    # Apply patches: update resolution_ts and settlement_minus_trade_sec
    for cid, rts in patches.items():
        mask = df["condition_id"] == cid
        df.loc[mask, "resolution_ts"] = rts.strftime("%Y-%m-%d %H:%M:%S%z")
        # Recompute settlement_minus_trade_sec for these trades
        sub = df.loc[mask]
        ts = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce")
        df.loc[mask, "settlement_minus_trade_sec"] = (
            (rts - ts).dt.total_seconds().values
        )

    df = df.drop(columns=["ts", "rts_parsed"])
    print(f"\n[write] {CSV}")
    df.to_csv(CSV, index=False)
    print(
        f"[done] patched {len(patches)} market(s); {len(unresolved)} remain unresolved"
    )
    if unresolved:
        print("[note] unresolved markets (data window likely ended before true lock):")
        for q in unresolved:
            print(f"         {q}")


if __name__ == "__main__":
    main()
