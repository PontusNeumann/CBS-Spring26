"""Add four binary missingness-indicator columns to the mother CSV.

Implements the dataset-level policy documented in `data/MISSING_DATA.md`:
preserve NaN in the raw feature columns, and attach one binary indicator
per missingness species so downstream imputation at modelling time never
erases the "was this observation missing" signal.

Indicators added (all dtype int8, values in {0, 1}):
  - wallet_has_prior_trades              1 iff wallet_prior_trades > 0
                                         Covers structural NaN on
                                         wallet_prior_win_rate,
                                         wallet_prior_volume_usd,
                                         size_vs_wallet_avg.
  - wallet_has_prior_trades_in_market    1 iff wallet_prior_trades_in_market > 0
                                         Covers structural NaN on
                                         wallet_directional_purity_in_market,
                                         wallet_spread_ratio,
                                         wallet_median_gap_in_market.
  - wallet_has_cross_market_history      1 iff wallet_market_category_entropy
                                         is defined (not NaN, not NULL).
                                         Collapses structural (<2 prior
                                         markets) + pipeline (wallet absent
                                         from HF mirror) into one bit for
                                         modelling. The MISSING_DATA.md doc
                                         preserves the sub-typology for the
                                         methodology section.
  - market_timing_known                  1 iff pct_time_elapsed is defined.
                                         Flags markets missing both
                                         resolution_ts and end_date metadata.

`wallet_enriched` (the Layer 6 indicator) is NOT added here — it is added
by 11_add_layer6.py alongside the Layer 6 feature columns themselves.

A pre-patch backup is written to data/03_trades_features.pre11b.csv so the
patch is reversible in one file copy.

Usage:
  python scripts/11b_add_missingness_flags.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data" / "03_trades_features.csv"
BACKUP_CSV = ROOT / "data" / "03_trades_features.pre11b.csv"

NEW_COLS = [
    "wallet_has_prior_trades",
    "wallet_has_prior_trades_in_market",
    "wallet_has_cross_market_history",
    "market_timing_known",
]


def main() -> None:
    t0 = time.time()

    print(f"loading {FEATURES_CSV.name}...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"  {len(df):,} rows × {len(df.columns)} cols in {time.time() - t0:.1f}s")

    existing = [c for c in NEW_COLS if c in df.columns]
    if existing:
        print(
            f"WARNING: {len(existing)} indicator columns already present: "
            f"{existing}. Overwriting."
        )
        df = df.drop(columns=existing)

    required = {
        "wallet_prior_trades": "wallet_has_prior_trades",
        "wallet_prior_trades_in_market": "wallet_has_prior_trades_in_market",
        "wallet_market_category_entropy": "wallet_has_cross_market_history",
        "pct_time_elapsed": "market_timing_known",
    }
    missing_sources = [c for c in required if c not in df.columns]
    if missing_sources:
        raise RuntimeError(
            f"source columns missing — cannot derive indicators: "
            f"{missing_sources}"
        )

    print("deriving indicators...")

    df["wallet_has_prior_trades"] = (
        pd.to_numeric(df["wallet_prior_trades"], errors="coerce").fillna(0) > 0
    ).astype(np.int8)

    df["wallet_has_prior_trades_in_market"] = (
        pd.to_numeric(df["wallet_prior_trades_in_market"], errors="coerce").fillna(0)
        > 0
    ).astype(np.int8)

    df["wallet_has_cross_market_history"] = (
        df["wallet_market_category_entropy"].notna()
        & ~df["wallet_market_category_entropy"].apply(
            lambda x: isinstance(x, float) and np.isnan(x)
        )
    ).astype(np.int8)

    df["market_timing_known"] = df["pct_time_elapsed"].notna().astype(np.int8)

    print("\nindicator summary (share of 1s):")
    for c in NEW_COLS:
        share = df[c].mean() * 100
        n_one = int(df[c].sum())
        print(f"  {c:40s}  {n_one:>9,} / {len(df):,}  ({share:5.2f}%)")

    print(f"\nbacking up to {BACKUP_CSV.name} and writing patched CSV...")
    t1 = time.time()
    shutil.copy2(FEATURES_CSV, BACKUP_CSV)
    df.to_csv(FEATURES_CSV, index=False)
    print(
        f"  backup {BACKUP_CSV.stat().st_size / 1e6:.0f} MB; "
        f"patched CSV {FEATURES_CSV.stat().st_size / 1e6:.0f} MB; "
        f"done in {time.time() - t1:.0f}s"
    )

    print(
        f"\nwrote {FEATURES_CSV.name} — shape {df.shape} in "
        f"{time.time() - t0:.0f}s total"
    )


if __name__ == "__main__":
    main()
