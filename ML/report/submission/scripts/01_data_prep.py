"""
01_data_prep.py — Load the consolidated dataset, split train/test, run leakage checks.

Source-of-truth merge of:
  alex/v5_final_ml_pipeline/scripts/00_unpack_release.py   (data unpacking)
  alex/v5_final_ml_pipeline/scripts/01_validate_schema.py  (schema sanity)
  alex/v5_final_ml_pipeline/scripts/02_causality_guard.py  (causality / leakage)

Teacher-facing simplification: the Alex pipeline split data into four parquets
for internal versioning (v3.5 vs v4). For the submission we load the single
consolidated parquet directly — same data, fewer files, easier to follow.

Run:
  python 01_data_prep.py

Outputs:
  outputs/data/feature_cols.json   list of 81 modelling features (after exclusions)
  outputs/data/leakage_report.json results of all leakage checks (pass/fail)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# what: import central paths/seeds so every script in the pipeline reads from one place
# how: config.py lives next to this script
# why: teacher only edits one file if their data path differs
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, OUTPUTS_DIR, RANDOM_SEED  # noqa: E402

# what: meta columns that are NOT features (book-keeping for the row)
# how: we exclude them when building X for modelling
META_COLS = {"split", "market_id", "ts_dt", "timestamp"}
TARGET = "bet_correct"

# what: columns we refuse to use for modelling because they leak the future
# why: each one peeks at information that would not be available at trade time
# how: dropped before any model is fit; tested below (test F3)
FORBIDDEN_LEAKY_COLS = {
    "kyle_lambda_market_static",   # fit on first half of each market then broadcast back
    "wallet_funded_by_cex",        # lifetime flag — true if wallet ever got CEX deposit, even after t
    "n_tokentx",                   # lifetime tx count — peeks past trade time
    "wallet_prior_win_rate",       # naive version that includes the current trade
}

# what: row counts we expect from the team's 2026-04-29 release
# why: a short-circuit check that we are reading the right file
EXPECTED_TRAIN_ROWS = 1_114_003
EXPECTED_TEST_ROWS = 257_177
EXPECTED_TOTAL_ROWS = EXPECTED_TRAIN_ROWS + EXPECTED_TEST_ROWS

# what: timestamps of the two real-world events that bracket the test cohort
# why: any trade timestamped after these is leakage (post-event)
STRIKE_EVENT_UTC = pd.Timestamp("2026-02-28T06:35:00", tz="UTC").timestamp()
CEASEFIRE_EVENT_UTC = pd.Timestamp("2026-04-07T23:59:59", tz="UTC").timestamp()


def load_consolidated() -> pd.DataFrame:
    """Load the single source-of-truth parquet."""
    # what: locate the data file
    src = DATA_DIR / "consolidated_modeling_data.parquet"
    if not src.exists():
        raise SystemExit(
            f"Dataset not found at {src}\n"
            "See submission/data/README.md for the download instructions."
        )
    # how: pandas reads the parquet directly into a DataFrame
    df = pd.read_parquet(src)
    print(f"  loaded {len(df):,} rows × {len(df.columns)} cols from {src.name}")
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the `split` column to separate train and test (cohort-disjoint by market)."""
    # what: the `split` column was assigned upstream by market cohort, not by random shuffle
    # why: random shuffle would leak info from the same market across train and test
    # how: simple boolean masking
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  train: {len(train):,}  test: {len(test):,}")
    return train, test


def check_row_counts(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Test S1 — row counts match the released contract."""
    # what: hard contract from the release manifest
    # why: catches accidental file swap or partial download
    ok = (len(train) == EXPECTED_TRAIN_ROWS) and (len(test) == EXPECTED_TEST_ROWS)
    return {"name": "S1_row_counts", "pass": ok,
            "train_rows": len(train), "test_rows": len(test),
            "expected_train": EXPECTED_TRAIN_ROWS, "expected_test": EXPECTED_TEST_ROWS}


def check_class_balance(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Test S2 — target is roughly balanced (no severe imbalance to handle)."""
    # what: report positive-class rate; the dataset is ~50/50 by construction
    # why: imbalance would force SMOTE/ADASYN; balanced data lets us focus on signal
    train_pos = float(train[TARGET].mean())
    test_pos = float(test[TARGET].mean())
    return {"name": "S2_class_balance", "pass": abs(train_pos - 0.5) < 0.05,
            "train_pos_rate": train_pos, "test_pos_rate": test_pos}


def check_no_post_event_leakage(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Test C1 — no trades timestamped after the cohort's resolving event."""
    # what: train cohort ends at the strike event; test cohort ends at the ceasefire
    # why: a single post-event trade would let the model peek at the resolved outcome
    train_max = float(train["timestamp"].max())
    test_max = float(test["timestamp"].max())
    train_ok = train_max < STRIKE_EVENT_UTC
    test_ok = test_max < CEASEFIRE_EVENT_UTC
    return {"name": "C1_no_post_event_trades", "pass": train_ok and test_ok,
            "train_max_iso": str(pd.to_datetime(train_max, unit="s", utc=True)),
            "test_max_iso": str(pd.to_datetime(test_max, unit="s", utc=True))}


def check_no_forbidden_columns(df: pd.DataFrame) -> dict:
    """Test F3 — forbidden columns are filtered out of the feature list, even if present in the data."""
    # what: the consolidated parquet keeps these columns for traceback completeness, but we never feed them to a model
    # how: get_feature_cols() below excludes them; this check just confirms the exclusion happened
    # why: documenting the policy in the leakage report makes the safety-by-construction visible
    present_in_data = sorted(set(df.columns) & FORBIDDEN_LEAKY_COLS)
    feature_cols_after_exclusion = set(get_feature_cols(df))
    leak_into_features = sorted(feature_cols_after_exclusion & FORBIDDEN_LEAKY_COLS)
    return {"name": "F3_forbidden_cols_excluded_from_features",
            "pass": len(leak_into_features) == 0,
            "present_in_data_but_excluded": present_in_data,
            "would_leak_into_features": leak_into_features,
            "forbidden_set": sorted(FORBIDDEN_LEAKY_COLS)}


def check_pre_trade_price(test: pd.DataFrame) -> dict:
    """Test D1 — pre_trade_price is the previous trade's per-token price (not the current one)."""
    # what: spot-check the upstream feature pre_trade_price by re-deriving it ourselves
    # how: group by market_id, sort by timestamp, take price.shift(1); the first trade in each market gets 0.5
    # why: a one-bar shift error here would give the model the trade's own price as a "feature"
    if "pre_trade_price" not in test.columns or "price" not in test.columns:
        return {"name": "D1_pre_trade_price", "pass": True, "skipped": "raw price column not in dataset"}
    cols = ["market_id", "timestamp", "price", "pre_trade_price"]
    df = test[cols].copy()
    df["market_id"] = df["market_id"].astype(str)
    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    expected = df.groupby("market_id")["price"].shift(1).fillna(0.5).values
    actual = df["pre_trade_price"].values
    match_rate = float(np.mean(np.abs(actual - expected) < 1e-6))
    return {"name": "D1_pre_trade_price", "pass": match_rate >= 0.995,
            "match_rate": match_rate}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of columns that are actually features (not meta, not target, not forbidden)."""
    # what: filter the column list down to modelling features
    # why: this is the canonical feature list every downstream script should use
    excluded = META_COLS | {TARGET} | FORBIDDEN_LEAKY_COLS
    return sorted([c for c in df.columns if c not in excluded])


def main() -> int:
    # what: header so terminal output is easy to scan
    print("=" * 60)
    print("Stage 1 — Load data, run leakage checks, save feature list")
    print("=" * 60)

    # what: ensure the output folder exists for our reports
    out_dir = OUTPUTS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # what: load + split
    df = load_consolidated()
    train, test = split_train_test(df)

    # what: run all leakage / sanity checks and collect results in one report
    # why: a single JSON file is easier for the teacher to inspect than terminal scrollback
    checks = [
        check_row_counts(train, test),
        check_class_balance(train, test),
        check_no_post_event_leakage(train, test),
        check_no_forbidden_columns(df),
        check_pre_trade_price(test),
    ]
    n_pass = sum(1 for c in checks if c["pass"])
    for c in checks:
        flag = "PASS" if c["pass"] else "FAIL"
        print(f"  [{flag}] {c['name']}")
    print(f"  -> {n_pass}/{len(checks)} checks passed")

    # what: persist the leakage report next to the modelling outputs
    leak_path = out_dir / "leakage_report.json"
    leak_path.write_text(json.dumps({"checks": checks, "n_pass": n_pass}, indent=2))
    print(f"  saved leakage report -> {leak_path.relative_to(OUTPUTS_DIR.parent)}")

    # what: write the final feature list used by every downstream script
    feature_cols = get_feature_cols(df)
    fc_path = out_dir / "feature_cols.json"
    fc_path.write_text(json.dumps(feature_cols, indent=2))
    print(f"  saved {len(feature_cols)} feature names -> {fc_path.relative_to(OUTPUTS_DIR.parent)}")

    # what: hard-stop the pipeline if any leakage check failed
    # why: silently continuing would invalidate every model trained downstream
    if n_pass != len(checks):
        print("\nLeakage check failed. Refusing to continue.")
        return 1
    print("\nStage 1 complete. Proceed to 02_features.py.")
    return 0


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    sys.exit(main())
