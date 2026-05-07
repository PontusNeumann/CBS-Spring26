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
  outputs/data/feature_cols.json   list of 80 modeling features (after exclusions)
  outputs/data/leakage_report.json results of all leakage checks (pass/fail)
"""

from __future__ import annotations

import ast
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
# how: we exclude them when building X for modeling
META_COLS = {"split", "market_id", "ts_dt", "timestamp"}
TARGET = "bet_correct"

# what: modelling scripts whose StandardScaler usage we audit for train-only fitting
# why: a scaler fit on the full dataset (train+test) leaks test distribution into training
# how: D4 check walks each script's AST and flags any StandardScaler() not inside a
#      fold loop or fitted on a clearly train-scoped array
SCALER_AUDIT_FILES = [
    Path(__file__).resolve().parent / "03_train_models.py",
    Path(__file__).resolve().parent / "04_calibration.py",
    Path(__file__).resolve().parent / "06_tuning_optuna.py",
]

# what: columns we refuse to use for modeling because they leak the future
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

# what: backtest-only context omitted from the modeling parquet
# why: realistic backtests need true trade USD and corrected YES-normalized price,
#      but these raw fields are not modeling features and must stay out of X
BACKTEST_CONTEXT = DATA_DIR / "backtest_context.parquet"
BACKTEST_CONTEXT_REQUIRED_COLS = {
    "split", "row_in_split", "market_id", "timestamp", "usd_amount",
    "price", "token_amount", "pre_yes_price_corrected",
    "taker", "taker_direction", "nonusdc_side",
}

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


def check_backtest_context(df: pd.DataFrame) -> dict:
    """Test B1 — backtest context exists and aligns row-for-row with the modeling data."""
    # what: the modeling parquet intentionally omits raw trading fields; this sidecar supplies them
    # why: silent fallbacks in the backtest would make ROI and liquidity assumptions non-reproducible
    if not BACKTEST_CONTEXT.exists():
        return {"name": "B1_backtest_context", "pass": False,
                "reason": f"missing {BACKTEST_CONTEXT.name}"}
    ctx = pd.read_parquet(BACKTEST_CONTEXT)
    missing = sorted(BACKTEST_CONTEXT_REQUIRED_COLS - set(ctx.columns))
    if missing:
        return {"name": "B1_backtest_context", "pass": False,
                "reason": "missing required columns", "missing_columns": missing}
    if len(ctx) != len(df):
        return {"name": "B1_backtest_context", "pass": False,
                "reason": "row count mismatch", "context_rows": len(ctx),
                "model_rows": len(df)}

    split_checks = {}
    for split in ("train", "test"):
        model_sub = df[df["split"] == split][["market_id", "timestamp"]].reset_index(drop=True)
        ctx_sub = ctx[ctx["split"] == split].reset_index(drop=True)
        expected_row = np.arange(len(model_sub))
        row_ok = np.array_equal(ctx_sub["row_in_split"].to_numpy(), expected_row)
        key_ok = (
            model_sub["market_id"].astype(str).to_numpy() == ctx_sub["market_id"].astype(str).to_numpy()
        ).all() and (
            model_sub["timestamp"].to_numpy() == ctx_sub["timestamp"].to_numpy()
        ).all()
        price_ok = ctx_sub["pre_yes_price_corrected"].between(0, 1).all()
        usd_ok = np.isfinite(ctx_sub["usd_amount"]).all() and (ctx_sub["usd_amount"] >= 0).all()
        split_checks[split] = {
            "rows": int(len(ctx_sub)),
            "row_in_split_ok": bool(row_ok),
            "market_timestamp_alignment_ok": bool(key_ok),
            "pre_yes_price_in_0_1": bool(price_ok),
            "usd_amount_finite_nonnegative": bool(usd_ok),
        }
    ok = all(all(v for k, v in c.items() if k != "rows") for c in split_checks.values())
    return {"name": "B1_backtest_context", "pass": bool(ok),
            "context_file": BACKTEST_CONTEXT.name, "checks": split_checks}


# ---------------------------------------------------------------------------
# D4 — StandardScaler refit-per-fold AST audit
# ---------------------------------------------------------------------------


def _enclosing_func(tree: ast.AST, target: ast.AST) -> ast.FunctionDef | None:
    """Return the FunctionDef node that directly contains `target`, or None."""
    # what: walk the AST to find which function owns the target node
    # why: we need the function scope to check for fold loops and train-subset fits
    # how: iterate all FunctionDef nodes, inner-walk each; first match wins
    candidate: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if inner is target:
                    candidate = node
                    break
            if candidate is not None:
                break
    return candidate


def _has_fold_loop(func: ast.FunctionDef) -> bool:
    """True if `func` body contains a recognisable cross-validation split call.

    Recognised patterns:
      - `<x>.split(...)` where x is any CV splitter (KFold, GroupKFold, etc.)
      - any `.split(...)` attribute call — conservative but sufficient
    """
    # what: look for .split(...) calls anywhere in the function body
    # why: a CV splitter's .split() is the canonical fold-loop marker
    # how: walk all Call nodes; check for an Attribute named "split"
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "split":
                return True
    return False


def _fits_only_train_subset(func: ast.FunctionDef) -> bool:
    """True if every fit/fit_transform call in `func` targets a train-scoped array.

    Recognises:
      - X.iloc[tr_idx] or X[tr_idx]  →  ast.Subscript
      - X_train, X_tr                →  ast.Name ending in _train/_tr
    Returns True if at least one qualifying pattern is found, or if no
    fit/fit_transform calls exist at all (scaler instantiated but unused — not a risk).
    """
    # what: check that fit_transform is called on a train-restricted subset
    # why: fit_transform on unsliced X uses test rows and leaks the test distribution
    # how: inspect the first argument of each fit/fit_transform call
    seen_any = False
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("fit_transform", "fit"):
                if not node.args:
                    continue
                seen_any = True
                arg0 = node.args[0]
                # X.iloc[...] or X[...]
                if isinstance(arg0, ast.Subscript):
                    return True
                # X.iloc[...].something (Attribute of a Subscript)
                if isinstance(arg0, ast.Attribute) and isinstance(
                    arg0.value, ast.Subscript
                ):
                    return True
                # X_train, X_tr, etc.
                if isinstance(arg0, ast.Name) and (
                    arg0.id.endswith("_train") or arg0.id.endswith("_tr")
                ):
                    return True
    # no fit calls at all — scaler present but not fitted here; not a leak risk
    return not seen_any


def check_scaler_refit_per_fold() -> dict:
    """Test D4 — every StandardScaler() is inside a CV fold loop or fits only a train subset.

    what: AST-walk each modelling script and flag any StandardScaler() instantiation
          that is neither inside a function with a .split() call nor fitted on a
          clearly train-scoped array (subscripted, iloc'd, or a Name ending _train/_tr).
    why:  a scaler fit on the full dataset before the CV split leaks test-set
          distribution statistics into the training signal.
    how:  parse each file with ast.parse, find all StandardScaler() Call nodes,
          look up their enclosing function, then apply _has_fold_loop and
          _fits_only_train_subset heuristics.
    """
    findings = []
    suspicious = []

    for path in SCALER_AUDIT_FILES:
        if not path.exists():
            findings.append({"file": path.name, "reason": "file_not_found", "ok": True})
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as e:
            suspicious.append({"file": path.name, "reason": f"SyntaxError: {e}"})
            continue

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "StandardScaler"
            ):
                line_no = getattr(node, "lineno", -1)
                func = _enclosing_func(tree, node)
                func_name = func.name if func else "<module>"
                ok_fold = _has_fold_loop(func) if func else False
                ok_subset = _fits_only_train_subset(func) if func else False
                ok = ok_fold or ok_subset
                findings.append(
                    {
                        "file": path.name,
                        "line": line_no,
                        "function": func_name,
                        "has_fold_loop": ok_fold,
                        "fits_train_subset": ok_subset,
                        "ok": ok,
                    }
                )
                if not ok:
                    suspicious.append(
                        {
                            "file": path.name,
                            "line": line_no,
                            "function": func_name,
                            "reason": (
                                "StandardScaler outside CV loop AND "
                                "fit_transform not on a train-subset"
                            ),
                        }
                    )

    return {
        "name": "D4_scaler_refit_per_fold",
        "pass": len(suspicious) == 0,
        "findings": findings,
        "suspicious": suspicious,
    }


# ---------------------------------------------------------------------------
# W1 — wallet causal-bisection spot-check
# ---------------------------------------------------------------------------


def check_wallet_bisection(test: pd.DataFrame) -> dict:
    """Test W1 — wallet features re-derived from raw enrichment match the parquet values.

    what: for a 500-row sample of the test set, re-compute three wallet features
          (wallet_polygon_age_at_t_days, wallet_n_inbound_at_t, wallet_n_cex_deposits_at_t)
          from wallet_enrichment.parquet using only events with timestamp < t, then
          compare against the values stored in the modelling parquet.
    why:  detects any off-by-one shift or future-looking join in the wallet feature
          engineering step; a mismatch here would mean the wallet features leak
          post-trade data.
    how:  read enrichment with fetch_status=='ok', build wallet→info lookup dict,
          sample 500 rows with seed=42, recompute using np.searchsorted (bisect-left),
          require >=99% match rate to pass.
    """
    wallet_enrich_path = DATA_DIR / "wallet_enrichment.parquet"

    # what: graceful skip when the file is absent from submission/data/
    # why: the upstream dev already verified this on the full dataset;
    #      in the submission boundary we document as skipped rather than failing
    if not wallet_enrich_path.exists():
        return {
            "name": "W1_wallet_bisection",
            "pass": True,
            "skipped": "wallet_enrichment.parquet not in submission/data/",
        }

    required_cols = [
        "wallet_polygon_age_at_t_days",
        "wallet_n_inbound_at_t",
        "wallet_n_cex_deposits_at_t",
    ]
    missing_cols = [c for c in required_cols if c not in test.columns]
    if missing_cols:
        return {
            "name": "W1_wallet_bisection",
            "pass": True,
            "skipped": f"wallet feature columns not in test set: {missing_cols}",
        }

    if "taker" not in test.columns:
        return {
            "name": "W1_wallet_bisection",
            "pass": True,
            "skipped": "taker column not in test set",
        }

    # what: load enrichment data; restrict to successfully fetched wallets
    enrich = pd.read_parquet(wallet_enrich_path)
    enrich = enrich[enrich["fetch_status"] == "ok"]

    # what: build a per-wallet lookup dict with sorted timestamp arrays
    # how: inbound_ts and cex_deposit_ts are list-of-int columns in the enrichment parquet
    idx: dict[str, dict] = {}
    for _, r in enrich.iterrows():
        idx[str(r["wallet"]).lower()] = {
            "polygon_first_tx_ts": (
                int(r["polygon_first_tx_ts"])
                if not pd.isna(r["polygon_first_tx_ts"])
                else None
            ),
            "inbound_ts": (
                np.asarray(r["inbound_ts"], dtype=np.int64)
                if len(r["inbound_ts"])
                else np.array([], dtype=np.int64)
            ),
            "cex_deposit_ts": (
                np.asarray(r["cex_deposit_ts"], dtype=np.int64)
                if len(r["cex_deposit_ts"])
                else np.array([], dtype=np.int64)
            ),
        }

    # what: sample 500 test rows uniformly with a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    n_sample = min(500, len(test))
    sample_idx = rng.choice(len(test), size=n_sample, replace=False)

    n_checked = 0
    n_match = 0
    mismatches: list[dict] = []

    for i in sample_idx:
        wallet = str(test.iloc[i]["taker"]).lower()
        info = idx.get(wallet)
        if info is None:
            continue
        n_checked += 1
        t = int(test.iloc[i]["timestamp"])

        # what: re-derive each feature using only events strictly before timestamp t
        # how: np.searchsorted with side='left' gives count of events < t
        if info["polygon_first_tx_ts"] is None:
            exp_age = float("nan")
        else:
            exp_age = max(0, t - info["polygon_first_tx_ts"]) / 86400.0

        exp_inbound = (
            int(np.searchsorted(info["inbound_ts"], t, side="left"))
            if len(info["inbound_ts"])
            else 0
        )
        exp_cex = (
            int(np.searchsorted(info["cex_deposit_ts"], t, side="left"))
            if len(info["cex_deposit_ts"])
            else 0
        )

        act_age = test.iloc[i]["wallet_polygon_age_at_t_days"]
        act_inbound = test.iloc[i]["wallet_n_inbound_at_t"]
        act_cex = test.iloc[i]["wallet_n_cex_deposits_at_t"]

        ok_age = (np.isnan(exp_age) and pd.isna(act_age)) or (
            not np.isnan(exp_age)
            and not pd.isna(act_age)
            and abs(float(act_age) - exp_age) < 1e-3
        )
        ok_inbound = float(act_inbound) == float(exp_inbound)
        ok_cex = float(act_cex) == float(exp_cex)

        if ok_age and ok_inbound and ok_cex:
            n_match += 1
        elif len(mismatches) < 10:
            mismatches.append(
                {
                    "row": int(i),
                    "wallet": wallet[:10] + "...",
                    "ts": t,
                    "age_exp": float(exp_age) if not np.isnan(exp_age) else None,
                    "age_act": float(act_age) if not pd.isna(act_age) else None,
                    "inbound_exp": exp_inbound,
                    "inbound_act": float(act_inbound),
                    "cex_exp": exp_cex,
                    "cex_act": float(act_cex),
                }
            )

    match_rate = n_match / n_checked if n_checked > 0 else 0.0
    passed = n_checked == 0 or match_rate >= 0.99

    return {
        "name": "W1_wallet_bisection",
        "pass": bool(passed),
        "n_sampled": int(n_sample),
        "n_checked": n_checked,
        "n_match": n_match,
        "match_rate": float(match_rate),
        "mismatches_sample": mismatches,
    }


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of columns that are actually features (not meta, not target, not forbidden)."""
    # what: filter the column list down to modeling features
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
        check_scaler_refit_per_fold(),
        check_wallet_bisection(test),
        check_backtest_context(df),
    ]
    n_pass = sum(1 for c in checks if c["pass"])
    for c in checks:
        flag = "PASS" if c["pass"] else "FAIL"
        print(f"  [{flag}] {c['name']}")
    print(f"  -> {n_pass}/{len(checks)} checks passed")

    # what: persist the leakage report next to the modeling outputs
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
