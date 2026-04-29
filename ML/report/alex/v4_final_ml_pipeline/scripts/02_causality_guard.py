"""
02_causality_guard.py — Stage 2 of the v4 final pipeline.

Re-runs the Phase 1 quick-wins on v4 inputs (post-wallet-join), with three new
v4-specific checks bolted on. Halts on the first fatal failure (C1, D1, D2, D3,
D4, S1, F3, W1) — these mean leakage, target corruption, or contract violation
and the entire pipeline must be re-run.

Tests
-----
Inherited (re-anchored to v4 inputs):
  T1  C1 — pre-event timestamp filter (raw HF parquets)
  T2  D1 — pre_trade_price = price.shift(1) per market (v4 parquet)
  T3  D2 — rolling features in 06b_engineer_features + Pontus's wallet-join use
           closed='left' or strictly-prior bisect
  T4  D3 — taker_position_size_before_trade recompute spot-check (v4 parquet)
  T5  D4 — StandardScaler refit per fold across v4 modelling scripts (AST walk,
           replaces brittle regex from v3.5)
  T6  F2 — duplicate transaction_hash in raw parquets (non-fatal)
  T7  A3 — answer1 == "Yes" across markets_subset (non-fatal note if missing)
  T8  C2 — resolution status for the 4 dominant test markets (non-fatal)
  T9  F1 — HF coverage vs market.volume (non-fatal)

New (v4-specific):
  T10 S1 — schema sanity: v4 parquets have expected row counts, target column,
           and 80 features that match feature_cols.json
  T11 F3 — forbidden columns absent from v4 parquets (kyle_lambda_market_static,
           wallet_funded_by_cex, n_tokentx, wallet_prior_win_rate)
  T12 W1 — wallet causal-bisection spot-check: for a sample of rows, the wallet
           feature in the parquet matches a re-derivation from
           wallet_enrichment.parquet using only events with timestamp < t

Outputs:
  alex/.scratch/pressure_tests/causality_guard_v4_results.json
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]  # alex/
DATA = ROOT / "data"
PIPE = ROOT / "v4_final_ml_pipeline" / "scripts"
WALLET_ENRICH = ROOT.parent / "data" / "wallet_enrichment.parquet"  # ML/report/data/
SCRATCH_OUT = ROOT / ".scratch" / "pressure_tests"
SCRATCH_OUT.mkdir(parents=True, exist_ok=True)

STRIKE_EVENT_UTC_TS = pd.Timestamp("2026-02-28T06:35:00", tz="UTC").timestamp()
CEASEFIRE_EVENT_UTC_TS = pd.Timestamp("2026-04-07T23:59:59", tz="UTC").timestamp()

V4_TRAIN = DATA / "train_features_v4.parquet"
V4_TEST = DATA / "test_features_v4.parquet"
FEATURE_COLS_JSON = DATA / "feature_cols.json"

EXPECTED_TRAIN_ROWS = 1_114_003
EXPECTED_TEST_ROWS = 257_177
EXPECTED_N_FEATURES = 80

FORBIDDEN_COLS = {
    "kyle_lambda_market_static",  # definitional leak (first-half fit)
    "wallet_funded_by_cex",  # static lifetime flag (P0-9 family)
    "n_tokentx",  # lifetime total — peeks post-trade
    "wallet_prior_win_rate",  # naive (P0-9) version, not the causal one
}

# Modelling scripts whose StandardScaler usage we audit. v3.5 list was stale
# (`07_sweep.py`, `06_baseline_idea1.py` no longer exist in this pipeline).
SCALER_AUDIT_FILES = [
    PIPE / "03_sweep.py",
    PIPE / "_backtest_worker.py",
    PIPE / "05_optuna_tuning.py",
    PIPE / "_optuna_worker.py",
    PIPE / "06_phase2_falsification.py",
]

# Feature-engineering scripts whose .rolling() audits run.
ROLLING_AUDIT_FILES = [
    ROOT / "scripts" / "06b_engineer_features.py",
    ROOT.parent / "pontus" / "scripts" / "build_walletjoined_features.py",
]

results: dict = {}


def _section(name: str):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)


def _record(test_id: str, status: str, detail: dict):
    results[test_id] = {"status": status, **detail}
    icon = {"PASS": "✓", "FAIL": "✗", "NOTE": "ℹ", "ERROR": "!"}.get(status, "?")
    print(f"  {icon} [{test_id}] {status}")


# ---------------------------------------------------------------------------
# T1  C1 — pre-event timestamp filter
# ---------------------------------------------------------------------------


def t_c1_pre_event_filter():
    _section("T1  C1 — pre-event timestamp filter")
    train = pd.read_parquet(DATA / "train.parquet", columns=["timestamp"])
    test = pd.read_parquet(DATA / "test.parquet", columns=["timestamp"])
    tr_max = float(train["timestamp"].max())
    te_max = float(test["timestamp"].max())
    tr_violation = tr_max >= STRIKE_EVENT_UTC_TS
    te_violation = te_max >= CEASEFIRE_EVENT_UTC_TS

    detail = {
        "train_max_dt": str(pd.to_datetime(tr_max, unit="s", utc=True)),
        "test_max_dt": str(pd.to_datetime(te_max, unit="s", utc=True)),
        "train_violations": int(tr_violation),
        "test_violations": int(te_violation),
    }
    print(f"  train max: {detail['train_max_dt']}  (cutoff strike event)")
    print(f"  test  max: {detail['test_max_dt']}  (cutoff ceasefire event)")
    status = "FAIL" if (tr_violation or te_violation) else "PASS"
    _record("C1", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T2  D1 — pre_trade_price = price.shift(1) within market (v4 parquet)
# ---------------------------------------------------------------------------


def t_d1_pre_trade_price():
    _section("T2  D1 — pre_trade_price = price.shift(1) per market (v4)")
    test_raw = pd.read_parquet(
        DATA / "test.parquet", columns=["market_id", "timestamp", "price"]
    )
    test_v4 = pd.read_parquet(
        V4_TEST, columns=["market_id", "timestamp", "pre_trade_price"]
    )
    test_raw["market_id"] = test_raw["market_id"].astype(str)
    test_v4["market_id"] = test_v4["market_id"].astype(str)
    test_raw = test_raw.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    test_v4 = test_v4.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    expected = test_raw.groupby("market_id")["price"].shift(1).fillna(0.5).values
    actual = test_v4["pre_trade_price"].values
    diffs = np.abs(actual - expected)
    n_match = int((diffs < 1e-6).sum())
    n_total = len(diffs)
    match_rate = n_match / n_total

    detail = {
        "n_match": n_match,
        "n_total": n_total,
        "match_rate": float(match_rate),
        "max_abs_diff": float(diffs.max()),
    }
    print(f"  match rate: {match_rate:.4f} ({n_match:,}/{n_total:,})")
    print(f"  max abs diff: {diffs.max():.6f}")
    status = "PASS" if match_rate >= 0.995 else "FAIL"
    _record("D1", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T3  D2 — rolling features use closed='left'
# ---------------------------------------------------------------------------


def t_d2_rolling_audit():
    _section("T3  D2 — rolling closed='left' grep audit (06b + wallet-join)")
    suspicious = []
    safe_count = 0
    total_count = 0
    audited_files = []

    for path in ROLLING_AUDIT_FILES:
        if not path.exists():
            print(f"  ⚠ skipping (not found): {path}")
            continue
        audited_files.append(str(path.relative_to(ROOT.parent)))
        src = path.read_text()
        for m in re.finditer(r"\.rolling\([^)]*\)", src):
            total_count += 1
            line_no = src[: m.start()].count("\n") + 1
            snippet = m.group(0)
            if "closed=" in snippet and ("'left'" in snippet or '"left"' in snippet):
                safe_count += 1
            else:
                suspicious.append(
                    {"file": str(path.name), "line": line_no, "snippet": snippet}
                )

    print(f"  files audited: {audited_files}")
    print(f"  total .rolling() calls: {total_count}, safe: {safe_count}")
    if suspicious:
        for s in suspicious:
            print(f"    ✗ {s['file']}:{s['line']}  {s['snippet']}")

    detail = {
        "files_audited": audited_files,
        "total": total_count,
        "safe": safe_count,
        "suspicious": suspicious,
    }
    status = "PASS" if not suspicious else "FAIL"
    _record("D2", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T4  D3 — taker_position_size_before_trade recompute (v4 parquet)
# ---------------------------------------------------------------------------


def t_d3_wallet_position():
    _section("T4  D3 — taker_position_size_before_trade recompute (v4)")
    raw = pd.read_parquet(DATA / "test.parquet")
    feat = pd.read_parquet(
        V4_TEST,
        columns=["market_id", "timestamp", "taker_position_size_before_trade"],
    )

    raw["market_id"] = raw["market_id"].astype(str)
    feat["market_id"] = feat["market_id"].astype(str)
    raw = raw.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    feat = feat.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    if len(raw) != len(feat):
        _record(
            "D3",
            "FAIL",
            {"reason": "row count mismatch", "n_raw": len(raw), "n_feat": len(feat)},
        )
        return False

    side_buy = (
        raw["taker_direction"].astype(str).str.upper().eq("BUY").astype(int).values
    )
    outcome_yes = (raw["nonusdc_side"].astype(str) == "token1").astype(int).values
    sign = (2 * side_buy - 1) * (2 * outcome_yes - 1)
    tokens = raw["token_amount"].clip(lower=0).values
    raw["_signed_tokens"] = sign * tokens

    pos_cum = (
        raw.groupby(["market_id", "taker"])["_signed_tokens"]
        .cumsum()
        .shift(1)
        .fillna(0)
    )
    first_idx = raw.groupby(["market_id", "taker"]).head(1).index
    pos_cum.loc[first_idx] = 0
    expected = np.tanh(pos_cum.values / 1000.0)
    actual = feat["taker_position_size_before_trade"].values

    diffs = np.abs(actual - expected)
    n_match = int((diffs < 1e-6).sum())
    n_total = len(diffs)
    detail = {
        "n_match": n_match,
        "n_total": n_total,
        "match_rate": float(n_match / n_total),
        "max_abs_diff": float(diffs.max()),
    }
    print(
        f"  full-dataset match: {n_match:,}/{n_total:,} ({n_match / n_total * 100:.2f}%)"
    )
    print(f"  max abs diff: {diffs.max():.8f}")
    status = "PASS" if (n_match / n_total) >= 0.999 else "FAIL"
    _record("D3", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T5  D4 — StandardScaler refit per fold (AST walk)
# ---------------------------------------------------------------------------


def _enclosing_func_body(tree: ast.AST, target: ast.AST) -> ast.AST | None:
    """Return the FunctionDef that contains `target`, or None."""
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
    """True if `func` body contains a recognisable fold-iteration call.

    Recognised patterns:
      - `<x>.split(...)` where `<x>` is *KFold/GroupKFold/StratifiedKFold/etc.
      - the literal name `split` as the attribute end (`anything.split(...)`)
    """
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "split":
                return True
    return False


def _fits_only_train_subset(func: ast.FunctionDef) -> bool:
    """Heuristic: scaler.fit_transform is called on a subscripted/iloc'd object.

    Recognises:
      - X.iloc[tr_idx], X[tr_idx], X_train, etc.
    Returns True if at least one fit_transform target is restricted; False if
    every fit_transform call uses an unsubscripted full X.
    """
    seen_any = False
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("fit_transform", "fit"):
                if not node.args:
                    continue
                seen_any = True
                arg0 = node.args[0]
                # X.iloc[...] or X[...] => Subscript
                if isinstance(arg0, ast.Subscript):
                    return True
                # X.iloc.something => Attribute on Attribute
                if isinstance(arg0, ast.Attribute) and isinstance(
                    arg0.value, ast.Subscript
                ):
                    return True
                # X_train, X_tr, etc. => Name with "_train" suffix
                if isinstance(arg0, ast.Name) and (
                    arg0.id.endswith("_train") or arg0.id.endswith("_tr")
                ):
                    return True
    # If no fit/fit_transform calls at all, scaler is constructed but unused —
    # not a leak risk. Return True so it doesn't get flagged.
    return not seen_any


def t_d4_scaler_audit():
    _section("T5  D4 — StandardScaler refit per fold (AST walk)")
    findings = []
    suspicious = []

    for path in SCALER_AUDIT_FILES:
        if not path.exists():
            print(f"  ⚠ missing: {path.name}")
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
                func = _enclosing_func_body(tree, node)
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
                            "reason": "StandardScaler outside CV loop AND fit_transform not on a train-subset",
                        }
                    )

    print(f"  StandardScaler() instantiations: {len(findings)}")
    for f in findings:
        flag = "✓" if f["ok"] else "✗"
        print(
            f"  {flag} {f['file']}:{f['line']} in {f['function']}  "
            f"(fold_loop={f['has_fold_loop']}, train_subset={f['fits_train_subset']})"
        )
    if suspicious:
        for s in suspicious:
            print(f"    ✗ {s['file']}:{s['line']} {s['function']}  {s['reason']}")

    detail = {"findings": findings, "suspicious": suspicious}
    status = "PASS" if not suspicious else "FAIL"
    _record("D4", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T6  F2 — duplicate transaction_hash
# ---------------------------------------------------------------------------


def t_f2_dedupe_check():
    _section("T6  F2 — duplicate transaction_hash")
    train = pd.read_parquet(DATA / "train.parquet", columns=["transaction_hash"])
    test = pd.read_parquet(DATA / "test.parquet", columns=["transaction_hash"])
    tr_dups = int(train["transaction_hash"].duplicated().sum())
    te_dups = int(test["transaction_hash"].duplicated().sum())
    detail = {
        "train_total": len(train),
        "train_dups": tr_dups,
        "test_total": len(test),
        "test_dups": te_dups,
    }
    print(f"  train: {len(train):,} trades, {tr_dups:,} duplicates")
    print(f"  test:  {len(test):,} trades, {te_dups:,} duplicates")
    status = "PASS" if (tr_dups == 0 and te_dups == 0) else "NOTE"
    _record("F2", status, detail)
    return True


# ---------------------------------------------------------------------------
# T7  A3 — answer1 == "Yes" across all markets
# ---------------------------------------------------------------------------


def t_a3_answer1_check():
    _section("T7  A3 — answer1 == 'Yes' for all markets in markets_subset")
    m = pd.read_parquet(DATA / "markets_subset.parquet")
    if "answer1" not in m.columns:
        _record(
            "A3",
            "NOTE",
            {"reason": "answer1 column missing from markets_subset.parquet"},
        )
        return True
    bad = m[m["answer1"] != "Yes"]
    detail = {"n_markets": len(m), "n_bad": len(bad)}
    print(f"  total markets: {len(m)}, with answer1 != 'Yes': {len(bad)}")
    if len(bad):
        for _, r in bad.iterrows():
            print(f"    {r.get('id', '?')}: answer1={r['answer1']}")
    status = "PASS" if len(bad) == 0 else "NOTE"
    _record("A3", status, detail)
    return True


# ---------------------------------------------------------------------------
# T8  C2 — resolution status for the dominant test markets
# ---------------------------------------------------------------------------


def t_c2_dominant_resolutions():
    _section("T8  C2 — resolution for the dominant test markets")
    m = pd.read_parquet(DATA / "markets_subset.parquet")
    if "outcome_prices" not in m.columns:
        _record(
            "C2",
            "NOTE",
            {"reason": "outcome_prices column missing — skipping"},
        )
        return True
    m["id"] = m["id"].astype(str)
    dominant_ids = ["1466012", "1466013", "1466014", "1466015"]
    sub = m[m.id.isin(dominant_ids)]
    findings = []
    for _, row in sub.iterrows():
        try:
            prices = ast.literal_eval(row["outcome_prices"])
            p1, p2 = float(prices[0]), float(prices[1])
            if abs(p1 - 1.0) < 0.01:
                resolved, ok = "YES (token1)", False
            elif abs(p2 - 1.0) < 0.01:
                resolved, ok = "NO (token2)", True
            elif p2 > 0.95:
                resolved, ok = f"NO_implied (p2={p2:.4f})", True
            else:
                resolved, ok = f"AMBIGUOUS ({prices})", False
        except Exception as e:
            resolved, ok = f"PARSE_ERROR: {e}", False
        findings.append(
            {"id": row["id"], "resolved": resolved, "matches_assumption": ok}
        )
    for f in findings:
        flag = "✓" if f["matches_assumption"] else "✗"
        print(f"    {flag} {f['id']}: {f['resolved']}")
    n_ok = sum(1 for f in findings if f["matches_assumption"])
    status = "PASS" if n_ok == len(findings) and findings else "NOTE"
    _record("C2", status, {"findings": findings, "n_ok": n_ok})
    return True


# ---------------------------------------------------------------------------
# T9  F1 — HF coverage vs market.volume reconciliation
# ---------------------------------------------------------------------------


def t_f1_volume_reconciliation():
    _section("T9  F1 — HF coverage vs market.volume reconciliation")
    m = pd.read_parquet(DATA / "markets_subset.parquet")
    if "volume" not in m.columns:
        _record(
            "F1",
            "NOTE",
            {"reason": "volume column missing — skipping"},
        )
        return True
    m["id"] = m["id"].astype(str)
    train_trades = pd.read_parquet(
        DATA / "train.parquet", columns=["market_id", "usd_amount"]
    )
    test_trades = pd.read_parquet(
        DATA / "test.parquet", columns=["market_id", "usd_amount"]
    )
    all_trades = pd.concat([train_trades, test_trades])
    all_trades["market_id"] = all_trades["market_id"].astype(str)
    extracted = (
        all_trades.groupby("market_id")["usd_amount"].sum().rename("extracted_volume")
    )
    m = m.set_index("id")
    m["extracted_volume"] = extracted.reindex(m.index).fillna(0)
    m["ratio"] = m["extracted_volume"] / m["volume"].clip(lower=1)
    major = m[m.volume > 1e6].copy()
    flagged = major[(major.ratio < 0.5) | (major.ratio > 1.5)]
    print(
        f"  major markets >$1M: {len(major)}, flagged outside [0.5, 1.5]: {len(flagged)}"
    )
    detail = {
        "n_major": len(major),
        "n_flagged": len(flagged),
        "median_ratio": float(major.ratio.median()) if len(major) else None,
    }
    status = "PASS" if len(flagged) == 0 else "NOTE"
    _record("F1", status, detail)
    return True


# ---------------------------------------------------------------------------
# T10  S1 — schema sanity (v4-specific)
# ---------------------------------------------------------------------------


def t_s1_schema():
    _section("T10  S1 — v4 schema sanity (row counts + feature_cols.json)")
    train = pd.read_parquet(V4_TRAIN, columns=["bet_correct"])
    test = pd.read_parquet(V4_TEST, columns=["bet_correct"])
    fcols = json.loads(FEATURE_COLS_JSON.read_text())

    failures = []
    if len(train) != EXPECTED_TRAIN_ROWS:
        failures.append(f"train rows {len(train):,} != {EXPECTED_TRAIN_ROWS:,}")
    if len(test) != EXPECTED_TEST_ROWS:
        failures.append(f"test rows {len(test):,} != {EXPECTED_TEST_ROWS:,}")
    if len(fcols) != EXPECTED_N_FEATURES:
        failures.append(
            f"feature_cols.json has {len(fcols)} features, expected {EXPECTED_N_FEATURES}"
        )

    # feature_cols entries must all exist in the v4 parquet
    train_full = pd.read_parquet(V4_TRAIN)
    missing = [c for c in fcols if c not in train_full.columns]
    if missing:
        failures.append(f"feature_cols.json refs columns not in train_v4: {missing}")

    detail = {
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features_json": len(fcols),
        "missing_in_parquet": missing,
        "failures": failures,
    }
    print(f"  train: {len(train):,} rows, test: {len(test):,} rows")
    print(
        f"  feature_cols.json: {len(fcols)} features (expected {EXPECTED_N_FEATURES})"
    )
    if failures:
        for f in failures:
            print(f"    ✗ {f}")
    status = "FAIL" if failures else "PASS"
    _record("S1", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T11  F3 — forbidden columns absent
# ---------------------------------------------------------------------------


def t_f3_forbidden_cols():
    _section("T11  F3 — forbidden columns absent from v4 parquets")
    train_cols = set(pd.read_parquet(V4_TRAIN).columns)
    test_cols = set(pd.read_parquet(V4_TEST).columns)
    train_violations = sorted(train_cols & FORBIDDEN_COLS)
    test_violations = sorted(test_cols & FORBIDDEN_COLS)
    detail = {
        "forbidden_set": sorted(FORBIDDEN_COLS),
        "train_violations": train_violations,
        "test_violations": test_violations,
    }
    print(f"  forbidden set: {sorted(FORBIDDEN_COLS)}")
    print(f"  train violations: {train_violations or 'none'}")
    print(f"  test violations:  {test_violations or 'none'}")
    status = "PASS" if not (train_violations or test_violations) else "FAIL"
    _record("F3", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T12  W1 — wallet causal-bisection spot-check
# ---------------------------------------------------------------------------


def t_w1_wallet_bisect():
    _section("T12  W1 — wallet feature causal-bisection spot-check")
    if not WALLET_ENRICH.exists():
        _record(
            "W1",
            "NOTE",
            {"reason": f"wallet_enrichment.parquet not found at {WALLET_ENRICH}"},
        )
        print(f"  ⚠ skipping: wallet_enrichment.parquet missing at {WALLET_ENRICH}")
        return True

    raw = pd.read_parquet(
        DATA / "test.parquet", columns=["market_id", "timestamp", "taker"]
    )
    feat = pd.read_parquet(
        V4_TEST,
        columns=[
            "market_id",
            "timestamp",
            "wallet_polygon_age_at_t_days",
            "wallet_n_inbound_at_t",
            "wallet_n_cex_deposits_at_t",
        ],
    )
    raw["market_id"] = raw["market_id"].astype(str)
    feat["market_id"] = feat["market_id"].astype(str)
    raw = raw.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    feat = feat.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    if len(raw) != len(feat):
        _record(
            "W1",
            "FAIL",
            {"reason": "row count mismatch", "n_raw": len(raw), "n_feat": len(feat)},
        )
        return False

    # Build wallet lookup
    enrich = pd.read_parquet(WALLET_ENRICH)
    enrich = enrich[enrich["fetch_status"] == "ok"]
    idx: dict[str, dict] = {}
    for _, r in enrich.iterrows():
        idx[r["wallet"].lower()] = {
            "polygon_first_tx_ts": (
                int(r["polygon_first_tx_ts"])
                if not pd.isna(r["polygon_first_tx_ts"])
                else None
            ),
            "inbound_ts": np.asarray(r["inbound_ts"], dtype=np.int64)
            if len(r["inbound_ts"])
            else np.array([], dtype=np.int64),
            "cex_deposit_ts": np.asarray(r["cex_deposit_ts"], dtype=np.int64)
            if len(r["cex_deposit_ts"])
            else np.array([], dtype=np.int64),
        }

    # Sample 500 rows uniformly
    rng = np.random.RandomState(42)
    sample = rng.choice(len(raw), size=min(500, len(raw)), replace=False)

    n_checked = 0
    n_match = 0
    mismatches: list[dict] = []
    for i in sample:
        wallet = str(raw.iloc[i]["taker"]).lower()
        info = idx.get(wallet)
        if info is None:
            continue
        n_checked += 1
        t = int(raw.iloc[i]["timestamp"])

        # Recompute the three sampled features
        if info["polygon_first_tx_ts"] is None:
            exp_age = np.nan
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

        act_age = feat.iloc[i]["wallet_polygon_age_at_t_days"]
        act_inbound = feat.iloc[i]["wallet_n_inbound_at_t"]
        act_cex = feat.iloc[i]["wallet_n_cex_deposits_at_t"]

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

    detail = {
        "n_sampled": int(len(sample)),
        "n_checked": n_checked,
        "n_match": n_match,
        "match_rate": n_match / n_checked if n_checked else 0.0,
        "mismatches_sample": mismatches,
    }
    print(
        f"  sampled {len(sample)}, enriched checks: {n_checked}, "
        f"matched: {n_match} ({(n_match / n_checked * 100) if n_checked else 0:.1f}%)"
    )
    if mismatches:
        print(f"  first {len(mismatches)} mismatches:")
        for m in mismatches[:3]:
            print(f"    {m}")

    if n_checked == 0:
        status = "NOTE"
    elif n_match / n_checked >= 0.99:
        status = "PASS"
    else:
        status = "FAIL"
    _record("W1", status, detail)
    return status != "FAIL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Stage 2 — v4 causality guard")
    print("=" * 60)

    fatal = [
        ("S1", t_s1_schema),
        ("F3", t_f3_forbidden_cols),
        ("C1", t_c1_pre_event_filter),
        ("D1", t_d1_pre_trade_price),
        ("D2", t_d2_rolling_audit),
        ("D3", t_d3_wallet_position),
        ("D4", t_d4_scaler_audit),
        ("W1", t_w1_wallet_bisect),
    ]
    nonfatal = [
        ("F2", t_f2_dedupe_check),
        ("A3", t_a3_answer1_check),
        ("C2", t_c2_dominant_resolutions),
        ("F1", t_f1_volume_reconciliation),
    ]

    halted = False
    for tid, fn in fatal:
        try:
            ok = fn()
        except Exception as e:
            _record(tid, "ERROR", {"exception": str(e)})
            ok = False
        if not ok:
            print(f"\n*** FATAL FAIL on {tid}. Halting causality guard. ***")
            halted = True
            break

    if not halted:
        for tid, fn in nonfatal:
            try:
                fn()
            except Exception as e:
                _record(tid, "ERROR", {"exception": str(e)})

    out_path = SCRATCH_OUT / "causality_guard_v4_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 60)
    print("STAGE 2 SUMMARY")
    print("=" * 60)
    for tid, r in results.items():
        print(f"  {tid:5} {r['status']}")
    print(f"\nResults: {out_path}")
    if halted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
