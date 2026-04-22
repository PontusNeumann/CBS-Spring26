"""Patch `data/03_consolidated_dataset.csv` to use the deadline parsed from
each market's question text (e.g. "by February 27, 2026") instead of the
Gamma `endDate` field.

Motivation: Gamma returned `endDate = 2026-01-31` for 34 of 74 markets in
event 114242 — a stale batch placeholder unrelated to the market's actual
deadline. This poisoned the end_date-based time features introduced by
`17_fix_leakage.py`: `time_to_settlement_s` was negative on ~44 % of rows
and `pct_time_elapsed` was 100 % NaN on 20 markets where `end_date <
market_start`. The fix uses `parse_deadline_from_question` in
`01_polymarket_api.py`, which is provably causal (the deadline is part of
the question string, published at market creation).

Columns rewritten:
  - deadline_ts                  new — written for transparency / audit
  - time_to_settlement_s         recomputed from deadline_ts − timestamp
  - log_time_to_settlement       recomputed
  - pct_time_elapsed             recomputed with deadline_ts as life_total
  - size_x_time_to_settlement    recomputed (log_size × log_time_to_settlement)

`settlement_minus_trade_sec`, `bet_correct`, `is_yes`, resolution_ts, and
end_date are untouched. Idempotent. Writes a backup to
`data/03_consolidated_dataset.pre_deadline.csv`.

Usage:
  python scripts/18_fix_deadline.py
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
BACKUP = ROOT / "data" / "03_consolidated_dataset.pre_deadline.csv"

_spec = _ilu.spec_from_file_location(
    "polymarket_api", ROOT / "scripts" / "01_polymarket_api.py"
)
_fp = _ilu.module_from_spec(_spec)
sys.modules["polymarket_api"] = _fp
_spec.loader.exec_module(_fp)


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

    for c in ("question", "end_date", "timestamp", "condition_id", "log_size"):
        if c not in df.columns:
            raise SystemExit(f"source column missing: {c}")

    if BACKUP.exists():
        print(f"  backup {BACKUP.name} already present — not overwriting")
    else:
        print(f"writing backup {BACKUP.name}...")
        shutil.copy2(CSV, BACKUP)
        print(f"  backup written ({time.monotonic() - t0:.1f}s)")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")

    # ------------------------------------------------------------------
    # 1. Derive deadline_ts per market (one parse per unique condition_id)
    # ------------------------------------------------------------------
    print("parsing deadline_ts from question for each unique market...")
    meta = (
        df.groupby("condition_id")
        .agg(question=("question", "first"), end_date=("end_date", "first"))
        .reset_index()
    )
    fallback_year = meta["end_date"].dt.year.where(meta["end_date"].notna())
    meta["deadline_ts"] = [
        _fp.parse_deadline_from_question(q, int(fy) if pd.notna(fy) else None)
        for q, fy in zip(meta["question"], fallback_year)
    ]
    meta["deadline_ts"] = pd.to_datetime(
        meta["deadline_ts"], utc=True, errors="coerce"
    ).fillna(meta["end_date"])

    parsed_ok = int(
        (meta["deadline_ts"].notna() & (meta["deadline_ts"] != meta["end_date"])).sum()
    )
    unchanged = int((meta["deadline_ts"] == meta["end_date"]).sum())
    missing = int(meta["deadline_ts"].isna().sum())
    print(
        f"  markets: {len(meta)}   "
        f"deadline derived from question & differs from end_date: {parsed_ok}   "
        f"deadline equals end_date: {unchanged}   "
        f"no deadline: {missing}"
    )
    if missing:
        raise SystemExit(
            "some markets have no deadline and no end_date fallback — "
            "check meta input"
        )

    # ------------------------------------------------------------------
    # 2. Join deadline onto every row
    # ------------------------------------------------------------------
    if "deadline_ts" in df.columns:
        df = df.drop(columns=["deadline_ts"])
    df = df.merge(
        meta[["condition_id", "deadline_ts"]], on="condition_id", how="left"
    )

    # ------------------------------------------------------------------
    # 3. Recompute time features
    # ------------------------------------------------------------------
    print("recomputing time_to_settlement_s, log_time_to_settlement, "
          "pct_time_elapsed, size_x_time_to_settlement from deadline_ts...")
    t1 = time.monotonic()
    old_tts = pd.to_numeric(df.get("time_to_settlement_s"), errors="coerce")
    old_pct = pd.to_numeric(df.get("pct_time_elapsed"), errors="coerce")

    deadline = df["deadline_ts"]
    df["time_to_settlement_s"] = (deadline - df["timestamp"]).dt.total_seconds()
    df["log_time_to_settlement"] = np.log1p(
        df["time_to_settlement_s"].clip(lower=0).fillna(0)
    )
    market_start = df.groupby("condition_id")["timestamp"].transform("min")
    life_total = (deadline - market_start).dt.total_seconds()
    life_elapsed = (df["timestamp"] - market_start).dt.total_seconds()
    df["pct_time_elapsed"] = (
        life_elapsed / life_total.where(life_total > 0)
    ).clip(0, 1)
    df["size_x_time_to_settlement"] = (
        pd.to_numeric(df["log_size"], errors="coerce") * df["log_time_to_settlement"]
    )

    new_tts = df["time_to_settlement_s"]
    new_pct = df["pct_time_elapsed"]
    print(
        f"  time_to_settlement_s: "
        f"mean change = {(new_tts - old_tts).mean():,.0f}s  "
        f"rows changed = {((new_tts - old_tts).abs() > 1).sum():,}  "
        f"negative values before = {(old_tts < 0).sum():,}  "
        f"after = {(new_tts < 0).sum():,}"
    )
    print(
        f"  pct_time_elapsed:     "
        f"mean change = {(new_pct - old_pct).mean():.4f}  "
        f"non-null before = {old_pct.notna().sum():,}  "
        f"non-null after = {new_pct.notna().sum():,}"
    )
    print(f"  done ({time.monotonic() - t1:.1f}s)")

    # ------------------------------------------------------------------
    # 4. Sanity checks
    # ------------------------------------------------------------------
    per_m_neg = df.groupby("condition_id").apply(
        lambda g: (g["time_to_settlement_s"] < 0).mean(), include_groups=False
    )
    print(
        f"  per-market negative-time share: "
        f"min={per_m_neg.min():.3f}  max={per_m_neg.max():.3f}  "
        f"median={per_m_neg.median():.3f}"
    )
    per_m_pct_nan = df.groupby("condition_id").apply(
        lambda g: g["pct_time_elapsed"].isna().mean(), include_groups=False
    )
    all_nan = int((per_m_pct_nan >= 0.999).sum())
    print(f"  markets with 100% NaN pct_time_elapsed: {all_nan} (before: 20)")

    # ------------------------------------------------------------------
    # 5. Write back
    # ------------------------------------------------------------------
    print("writing CSV...")
    t2 = time.monotonic()
    df["end_date"] = df["end_date"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df["deadline_ts"] = df["deadline_ts"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df.to_csv(CSV, index=False)
    print(
        f"  wrote {CSV.name} "
        f"({len(df):,} rows × {len(df.columns)} cols, "
        f"{time.monotonic() - t2:.1f}s)"
    )
    print(f"total: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
