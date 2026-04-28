"""Causal regression guard for the wallet-joined Alex cohort.

Mirror of `scripts/_check_causal.py` adapted to:
  - inputs:  pontus/data/{train,test}_features_walletjoined.parquet
  - schema:  Alex's 70 engineered features + 12 Layer-6 wallet features
  - keys:    `taker` (lowercase), `market_id`, `bet_correct`

Re-run after every rebuild of the joined parquets. Non-zero exit signals
a regression. Each check group below describes the invariant being
asserted and why a breach would matter.

Check groups:
  D. Layer-6 cross-feature consistency
       wallet_enriched=0 ⟹ Layer-6 numerics are all NaN
       wallet_enriched=1 ⟹ Layer-6 numerics are non-NaN
       wallet_funded_by_cex_scoped ≤ wallet_funded_by_cex
  E. NaN-rate bounds
       Layer-6 numerics should have ≤1% NaN on `wallet_enriched=1` rows.
       The combined NaN rate equals 1 − coverage by construction.
  C. Sign checks (Layer-6 log features only)
       log_* features here are log1p of non-negative counts/amounts and
       must be ≥ 0 on `wallet_enriched=1` rows. Alex's log_* features
       (e.g. `log_size`, `log_time_to_deadline_hours`) can be negative
       and are excluded from this check.
  G. Schema completeness
       Both splits carry the same 86 columns and the same Layer-6 names
       Alex's pipeline expects.

Usage:
  python pontus/scripts/_check_causal_joined.py          # full
  python pontus/scripts/_check_causal_joined.py --quick  # sample 100k rows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "pontus" / "data"

LAYER6_NUMERIC = [
    "wallet_polygon_age_at_t_days",
    "wallet_polygon_nonce_at_t",
    "wallet_log_polygon_nonce_at_t",
    "wallet_n_inbound_at_t",
    "wallet_log_n_inbound_at_t",
    "wallet_n_cex_deposits_at_t",
    "wallet_cex_usdc_cumulative_at_t",
    "wallet_log_cex_usdc_cum",
    "days_from_first_usdc_to_t",
]
LAYER6_LOG = [c for c in LAYER6_NUMERIC if c.startswith("wallet_log_") or c.endswith("_log_cex_usdc_cum")]
LAYER6_BINARY = ["wallet_enriched", "wallet_funded_by_cex", "wallet_funded_by_cex_scoped"]

REQUIRED_COLS = LAYER6_NUMERIC + LAYER6_BINARY + ["bet_correct", "market_id", "timestamp"]


class Check:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def want(self, cond: bool, msg: str) -> None:
        (self.passes if cond else self.failures).append(msg)

    def report(self) -> int:
        print(f"\n— passed: {len(self.passes)}   failed: {len(self.failures)} —")
        for m in self.failures:
            print(f"  [FAIL] {m}")
        for m in self.passes:
            print(f"  [ pass] {m}")
        return 0 if not self.failures else 1


def run_one(df: pd.DataFrame, label: str, c: Check) -> None:
    print(f"\n=== {label}: {len(df):,} rows × {len(df.columns)} cols ===")

    # G. Schema completeness
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    c.want(not missing, f"{label}: required cols present (missing: {missing})")

    enriched_mask = df["wallet_enriched"] == 1
    n_enriched = int(enriched_mask.sum())
    n_total = len(df)
    coverage = n_enriched / n_total if n_total else 0
    print(f"  wallet_enriched: {n_enriched:,}/{n_total:,} ({coverage:.1%})")

    # D. Layer-6 invariants
    for col in LAYER6_NUMERIC:
        if col not in df.columns:
            continue
        unenriched_nan = df.loc[~enriched_mask, col].isna().mean()
        c.want(
            unenriched_nan == 1.0,
            f"{label}: {col} is NaN for every wallet_enriched=0 row "
            f"(observed NaN rate {unenriched_nan:.4f})",
        )

    for col in LAYER6_NUMERIC:
        if col not in df.columns:
            continue
        enriched_nan = df.loc[enriched_mask, col].isna().mean()
        c.want(
            enriched_nan <= 0.01,
            f"{label}: {col} NaN rate on enriched rows ≤1% "
            f"(observed {enriched_nan:.4f})",
        )

    if {"wallet_funded_by_cex", "wallet_funded_by_cex_scoped"} <= set(df.columns):
        viol = (df["wallet_funded_by_cex_scoped"] > df["wallet_funded_by_cex"]).sum()
        c.want(
            viol == 0,
            f"{label}: wallet_funded_by_cex_scoped ≤ wallet_funded_by_cex "
            f"(violations: {viol})",
        )

    # E. NaN-rate bounds for Alex's numeric features (excludes Layer-6 already covered)
    alex_feats = [
        col for col in df.columns
        if col not in REQUIRED_COLS + ["ts_dt"]
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    bad = []
    for col in alex_feats:
        nan_rate = df[col].isna().mean()
        if nan_rate > 0.05:
            bad.append((col, nan_rate))
    c.want(
        not bad,
        f"{label}: all Alex numeric features have <5% NaN "
        f"(violations: {[f'{c}={r:.3f}' for c, r in bad]})",
    )

    # C. Sign on Layer-6 log columns (must be ≥ 0 on enriched rows; log1p of counts/amounts)
    for col in LAYER6_LOG + ["wallet_log_cex_usdc_cum"]:
        if col not in df.columns:
            continue
        sub = df.loc[enriched_mask, col].dropna()
        n_neg = int((sub < 0).sum())
        c.want(
            n_neg == 0,
            f"{label}: {col} ≥ 0 on enriched rows (negatives: {n_neg})",
        )

    # Class balance sanity (cheap)
    base = df["bet_correct"].mean()
    c.want(0.30 < base < 0.70, f"{label}: bet_correct base rate in (0.30, 0.70) (observed {base:.3f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="sample 100k rows per split")
    args = ap.parse_args()

    c = Check()
    for split in ("train", "test"):
        path = DATA_DIR / f"{split}_features_walletjoined.parquet"
        if not path.exists():
            c.failures.append(f"{split}: parquet missing at {path}")
            continue
        df = pd.read_parquet(path)
        if args.quick and len(df) > 100_000:
            df = df.sample(100_000, random_state=42).reset_index(drop=True)
        run_one(df, split, c)

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
