"""
01_validate_schema.py — Stage 1 of the v4 pipeline.

Verifies that Pontus's delivered v4 parquets match the contract before any
modelling runs against them. If anything fails, halt and ask Pontus to
re-deliver — do NOT patch the parquet ourselves.

Inputs:
  data/train_features.parquet           v3.5 baseline (already on disk)
  data/test_features.parquet            v3.5 baseline (already on disk)
  data/train_features_v4.parquet        Pontus's joined output
  data/test_features_v4.parquet         Pontus's joined output
  data/feature_cols.json                v3.5 feature list (will be updated)

Outputs:
  data/feature_cols.json                updated to include the 6 new wallet columns
  stdout: PASS / FAIL per check

Note: feature engineering happens BEFORE this pipeline. v3.5 features
(70 columns) are produced by alex/scripts/06b_engineer_features.py against
the raw HF dataset; the wallet-augmented v4 join is done by Pontus on his
side and shipped as a parquet.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from _common import DATA

EXPECTED_TRAIN_ROWS = 1_114_003
EXPECTED_TEST_ROWS = 257_177

EXPECTED_NEW_WALLET_COLS = {
    "wallet_polygon_age_at_t_days",
    "wallet_log_polygon_nonce_at_t",
    "wallet_log_n_inbound_at_t",
    "wallet_n_cex_deposits_at_t",
    "wallet_log_cex_usdc_cum",
    "wallet_funded_by_cex_scoped",
}
DIAGNOSTIC_COLS = {"wallet_enriched"}  # OK to include but NOT a feature
FORBIDDEN_LEAKY_COLS = {
    "wallet_funded_by_cex",  # static lifetime constant — leaky pre-funding
    "n_tokentx",  # lifetime total — peeks at post-trade activity
    "wallet_prior_win_rate",  # the naive (P0-9 leak) version, not the causal one
}

KEY_COLS = ["market_id", "timestamp", "taker"]


def fail(msg: str) -> "sys.NoReturn":
    print(f"  ✗ FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def main():
    print("=" * 60)
    print("Stage 1 — Schema validation of Pontus's v4 parquets")
    print("=" * 60)

    train_v3 = pd.read_parquet(DATA / "train_features.parquet")
    test_v3 = pd.read_parquet(DATA / "test_features.parquet")
    try:
        train_v4 = pd.read_parquet(DATA / "train_features_v4.parquet")
        test_v4 = pd.read_parquet(DATA / "test_features_v4.parquet")
    except FileNotFoundError as e:
        fail(f"v4 parquet missing: {e}. Pontus has not delivered yet.")

    # 1. Row counts
    if len(train_v4) != EXPECTED_TRAIN_ROWS:
        fail(f"train_v4 has {len(train_v4):,} rows, expected {EXPECTED_TRAIN_ROWS:,}")
    if len(test_v4) != EXPECTED_TEST_ROWS:
        fail(f"test_v4 has {len(test_v4):,} rows, expected {EXPECTED_TEST_ROWS:,}")
    ok(f"row counts: train {len(train_v4):,}, test {len(test_v4):,}")

    # 2. v3.5 features all present
    v3_features = set(json.loads((DATA / "feature_cols.json").read_text()))
    missing_v3 = v3_features - set(train_v4.columns)
    if missing_v3:
        fail(f"missing {len(missing_v3)} v3.5 feature columns: {sorted(missing_v3)}")
    ok(f"all {len(v3_features)} v3.5 features preserved")

    # 3. Exactly the expected new wallet columns
    new_cols = set(train_v4.columns) - set(train_v3.columns)
    missing_new = EXPECTED_NEW_WALLET_COLS - new_cols
    if missing_new:
        fail(f"missing wallet columns: {sorted(missing_new)}")
    leaky = new_cols & FORBIDDEN_LEAKY_COLS
    if leaky:
        fail(f"forbidden leaky columns present: {sorted(leaky)}")
    unexpected = new_cols - EXPECTED_NEW_WALLET_COLS - DIAGNOSTIC_COLS
    if unexpected:
        print(f"  ⚠ unexpected extras (review): {sorted(unexpected)}")
    ok(f"6 expected wallet features present, {len(unexpected)} unexpected extras")

    # 4. Row alignment: same trades in same order as v3.5
    key_v3 = train_v3[KEY_COLS].astype(str).agg("|".join, axis=1)
    key_v4 = train_v4[KEY_COLS].astype(str).agg("|".join, axis=1)
    if not (key_v3 == key_v4).all():
        n_diff = int((key_v3 != key_v4).sum())
        fail(f"train row order differs from v3.5 ({n_diff:,} mismatched rows)")
    key_v3 = test_v3[KEY_COLS].astype(str).agg("|".join, axis=1)
    key_v4 = test_v4[KEY_COLS].astype(str).agg("|".join, axis=1)
    if not (key_v3 == key_v4).all():
        n_diff = int((key_v3 != key_v4).sum())
        fail(f"test row order differs from v3.5 ({n_diff:,} mismatched rows)")
    ok("row order matches v3.5 in both train and test")

    # 5. Wallet-feature coverage (v2 release claims 100%)
    nan_rate_train = train_v4["wallet_polygon_age_at_t_days"].isna().mean()
    nan_rate_test = test_v4["wallet_polygon_age_at_t_days"].isna().mean()
    if nan_rate_train > 0.001 or nan_rate_test > 0.001:
        print(
            f"  ⚠ wallet NaN rates: train {nan_rate_train * 100:.2f}%, "
            f"test {nan_rate_test * 100:.2f}% — v2 claims 100% coverage"
        )
    else:
        ok(
            f"wallet NaN rates: train {nan_rate_train * 100:.3f}%, "
            f"test {nan_rate_test * 100:.3f}% (within 0.1%)"
        )

    # 6. Update feature_cols.json
    new_feature_list = sorted(v3_features | EXPECTED_NEW_WALLET_COLS)
    (DATA / "feature_cols.json").write_text(json.dumps(new_feature_list, indent=2))
    ok(f"feature_cols.json updated: {len(new_feature_list)} features (v3.5 70 + 6 new)")

    print()
    print("=" * 60)
    print("Stage 1 PASS — proceed to Stage 2 (causality guard)")
    print("=" * 60)


if __name__ == "__main__":
    main()
