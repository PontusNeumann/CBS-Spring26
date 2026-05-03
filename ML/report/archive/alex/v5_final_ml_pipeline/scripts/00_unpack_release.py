"""
00_unpack_release.py — Stage 0 adapter for Pontus's 2026-04-29 release.

Pontus ships a single consolidated parquet (`consolidated_modeling_data.parquet`)
with a `split` column distinguishing train/test. The v4 pipeline expects four
files at `alex/data/` (per the v4 contract):

  - train_features.parquet      v3.5 baseline (no wallet cols)
  - test_features.parquet       v3.5 baseline (no wallet cols)
  - train_features_v4.parquet   v3.5 + wallet
  - test_features_v4.parquet    v3.5 + wallet

This adapter takes the consolidated parquet, drops two unwanted columns
(`kyle_lambda_market_static` — definitional leak, and `wallet_funded_by_cex` —
forbidden static lifetime flag per `01_validate_schema.py` FORBIDDEN_LEAKY_COLS),
and writes the four expected parquets. v3.5 baselines are derived by stripping
all wallet columns from the v4 parquets so row alignment is identity by
construction.

Run once after pulling the release tarball into `ML/report/data/`. Idempotent —
safe to re-run after an updated release.

Inputs:
  ML/report/data/consolidated_modeling_data.parquet  (Pontus's release)

Outputs (overwrites if present):
  alex/data/train_features.parquet
  alex/data/test_features.parquet
  alex/data/train_features_v4.parquet
  alex/data/test_features_v4.parquet
  alex/data/feature_cols.json    (v3.5 list, post-kyle-drop)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from _common import DATA, ROOT

# Path to Pontus's consolidated parquet. Two locations supported:
#   1. ML/report/data/  (the release-extract location per Pontus's README)
#   2. /tmp/pontus-release/  (where we extracted it during validation)
REPO_ROOT = ROOT.parent  # ML/report
RELEASE_CANDIDATES = [
    REPO_ROOT / "data" / "consolidated_modeling_data.parquet",
    Path("/tmp/pontus-release/consolidated_modeling_data.parquet"),
]

DROP_COLS = {
    # Definitional leak: fit on each market's first half then broadcast to all
    # rows in that market. Trades in the first half see post-trade information
    # from the same market.
    "kyle_lambda_market_static",
    # Static lifetime flag — leaks if the wallet's first CEX-sourced USDC
    # arrives after t. Causal version `wallet_funded_by_cex_scoped` is kept.
    # Listed in 01_validate_schema.py FORBIDDEN_LEAKY_COLS.
    "wallet_funded_by_cex",
}

# Wallet column families. After DROP_COLS removes the static flag, these are
# the wallet columns that ride along with the v4 parquet but are stripped to
# build the v3.5 baseline.
WALLET_COLS = {
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
    "wallet_funded_by_cex_scoped",
}

META_COLS = {"split", "market_id", "ts_dt", "timestamp", "bet_correct"}


def find_consolidated() -> Path:
    for p in RELEASE_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit(
        "consolidated_modeling_data.parquet not found in any of:\n  "
        + "\n  ".join(str(p) for p in RELEASE_CANDIDATES)
        + "\n\nExtract the release tarball first:\n"
        "  cd ML/report && tar -xzf /tmp/pontus-modeling-data-2026-04-29.tar.gz -C data/"
    )


def main() -> None:
    print("=" * 60)
    print("Stage 0 — Unpacking Pontus's 2026-04-29 release")
    print("=" * 60)

    src = find_consolidated()
    print(f"  source: {src}")

    df = pd.read_parquet(src)
    print(f"  loaded: {len(df):,} rows × {len(df.columns)} cols")

    # Drop unwanted columns (idempotent — works whether or not they're present)
    drop_actually = [c for c in DROP_COLS if c in df.columns]
    if drop_actually:
        df = df.drop(columns=drop_actually)
        print(f"  dropped: {sorted(drop_actually)}")
    else:
        print("  dropped: (already absent)")

    # Sanity: split column must exist
    if "split" not in df.columns:
        raise SystemExit("expected `split` column in consolidated parquet")

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    print(f"  split: train {int(train_mask.sum()):,}, test {int(test_mask.sum()):,}")
    if int(train_mask.sum()) + int(test_mask.sum()) != len(df):
        raise SystemExit("split column has values other than train/test — investigate")

    # Determine the v3.5 vs v4 column lists
    all_cols = set(df.columns)
    wallet_present = WALLET_COLS & all_cols
    v4_feature_cols = sorted(all_cols - META_COLS)
    v3_feature_cols = sorted((all_cols - META_COLS) - WALLET_COLS)

    print(f"  v3.5 feature count: {len(v3_feature_cols)} (excludes wallet)")
    print(f"  v4 feature count:   {len(v4_feature_cols)} (v3.5 + wallet)")
    missing_wallet = WALLET_COLS - wallet_present
    if missing_wallet:
        print(f"  ⚠ wallet cols absent from release: {sorted(missing_wallet)}")

    # Write v4 parquets (full feature set including wallet)
    v4_keep = ["market_id", "timestamp", "ts_dt", "bet_correct"] + v4_feature_cols
    train_v4 = df.loc[train_mask, v4_keep].reset_index(drop=True)
    test_v4 = df.loc[test_mask, v4_keep].reset_index(drop=True)

    train_v4_path = DATA / "train_features_v4.parquet"
    test_v4_path = DATA / "test_features_v4.parquet"
    train_v4.to_parquet(train_v4_path, index=False)
    test_v4.to_parquet(test_v4_path, index=False)
    print(f"  wrote {train_v4_path.relative_to(ROOT)}: {train_v4.shape}")
    print(f"  wrote {test_v4_path.relative_to(ROOT)}: {test_v4.shape}")

    # Write v3.5 parquets (strip wallet cols — same row order by construction)
    v3_keep = ["market_id", "timestamp", "ts_dt", "bet_correct"] + v3_feature_cols
    train_v3 = train_v4[[c for c in v3_keep if c in train_v4.columns]].copy()
    test_v3 = test_v4[[c for c in v3_keep if c in test_v4.columns]].copy()

    train_v3_path = DATA / "train_features.parquet"
    test_v3_path = DATA / "test_features.parquet"
    train_v3.to_parquet(train_v3_path, index=False)
    test_v3.to_parquet(test_v3_path, index=False)
    print(f"  wrote {train_v3_path.relative_to(ROOT)}: {train_v3.shape}")
    print(f"  wrote {test_v3_path.relative_to(ROOT)}: {test_v3.shape}")

    # Update feature_cols.json with the v3.5 list (Stage 1 will extend to v4)
    feature_cols_path = DATA / "feature_cols.json"
    feature_cols_path.write_text(json.dumps(v3_feature_cols, indent=2))
    print(
        f"  wrote {feature_cols_path.relative_to(ROOT)}: "
        f"{len(v3_feature_cols)} v3.5 features"
    )

    print()
    print("=" * 60)
    print("Stage 0 done — proceed to Stage 1 (01_validate_schema.py)")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
