"""
phase1_quick_wins.py — Phase 1 of the pressure-test plan.

Runs the 9 quick-verification tests that can falsify the headline if they fail.
Halts on the first fatal failure (C1, D1, D2, D3, D4) — these mean leakage or
target corruption and the entire pipeline must be re-run.

Tests:
  T1.1  C1 — pre-event timestamp filter
  T1.2  D1 — pre_trade_price = price.shift(1) within market
  T1.3  D2 — rolling features use closed='left' (grep + spot-check)
  T1.4  D3 — wallet position cumsum spot-check
  T1.5  D4 — StandardScaler refit per fold (code audit)
  T1.6  F2 — duplicate transaction_hash check
  T1.7  A3 — answer1 == "Yes" across all 75 markets
  T1.8  C2 — resolution status for 4 dominant test markets
  T1.9  F1 — HF coverage vs market.volume reconciliation

Outputs:
  - prints PASS/FAIL/NOTE for each test
  - alex/.scratch/pressure_tests/phase1_results.json
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
SCRATCH_OUT = ROOT / ".scratch" / "pressure_tests"
SCRATCH_OUT.mkdir(parents=True, exist_ok=True)

STRIKE_EVENT_UTC_TS = pd.Timestamp("2026-02-28T06:35:00", tz="UTC").timestamp()
CEASEFIRE_EVENT_UTC_TS = pd.Timestamp("2026-04-07T23:59:59", tz="UTC").timestamp()

results: dict = {}


def _section(name: str):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)


def _record(test_id: str, status: str, detail: dict):
    results[test_id] = {"status": status, **detail}
    icon = {"PASS": "✓", "FAIL": "✗", "NOTE": "ℹ"}.get(status, "?")
    print(f"  {icon} [{test_id}] {status}")


# ---------------------------------------------------------------------------
# T1.1  C1 — pre-event timestamp filter
# ---------------------------------------------------------------------------


def t1_1_pre_event_filter():
    _section("T1.1  C1 — pre-event timestamp filter")
    # Use raw trade parquets (timestamp lives there)
    train = pd.read_parquet(DATA / "train.parquet", columns=["timestamp"])
    test = pd.read_parquet(DATA / "test.parquet", columns=["timestamp"])
    tr_max = float(train["timestamp"].max())
    te_max = float(test["timestamp"].max())
    tr_violation = tr_max >= STRIKE_EVENT_UTC_TS
    te_violation = te_max >= CEASEFIRE_EVENT_UTC_TS

    detail = {
        "train_max_ts": tr_max,
        "train_max_dt": str(pd.to_datetime(tr_max, unit="s", utc=True)),
        "strike_event_ts": STRIKE_EVENT_UTC_TS,
        "test_max_ts": te_max,
        "test_max_dt": str(pd.to_datetime(te_max, unit="s", utc=True)),
        "ceasefire_event_ts": CEASEFIRE_EVENT_UTC_TS,
        "train_violations": int(tr_violation),
        "test_violations": int(te_violation),
    }
    print(
        f"  train max: {detail['train_max_dt']} (cutoff {pd.to_datetime(STRIKE_EVENT_UTC_TS, unit='s', utc=True)})"
    )
    print(
        f"  test  max: {detail['test_max_dt']} (cutoff {pd.to_datetime(CEASEFIRE_EVENT_UTC_TS, unit='s', utc=True)})"
    )
    status = "FAIL" if (tr_violation or te_violation) else "PASS"
    _record("C1", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T1.2  D1 — pre_trade_price = price.shift(1) within market
# ---------------------------------------------------------------------------


def t1_2_pre_trade_price():
    _section("T1.2  D1 — pre_trade_price lag correctness")
    test_raw = pd.read_parquet(
        DATA / "test.parquet", columns=["market_id", "timestamp", "price"]
    )
    test_feat = pd.read_parquet(
        DATA / "test_features.parquet",
        columns=["market_id", "timestamp", "pre_trade_price"],
    )
    # Both files come from the same source; sort identically
    test_raw["market_id"] = test_raw["market_id"].astype(str)
    test_feat["market_id"] = test_feat["market_id"].astype(str)
    test_raw = test_raw.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    test_feat = test_feat.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    # Per-market shift(1)
    expected_pre = test_raw.groupby("market_id")["price"].shift(1)
    expected_pre = expected_pre.fillna(0.5)  # feature engineering uses 0.5 fallback

    actual = test_feat["pre_trade_price"].values
    expected = expected_pre.values
    diffs = np.abs(actual - expected)
    n_match = int((diffs < 1e-6).sum())
    n_total = len(diffs)
    match_rate = n_match / n_total

    # Sample 200 trades to print specific examples
    sample = np.random.RandomState(42).choice(
        n_total, size=min(200, n_total), replace=False
    )
    n_sample_match = int((diffs[sample] < 1e-6).sum())

    detail = {
        "n_match": n_match,
        "n_total": n_total,
        "match_rate": float(match_rate),
        "sample_size": len(sample),
        "sample_match": n_sample_match,
        "max_abs_diff": float(diffs.max()),
        "mean_abs_diff": float(diffs.mean()),
    }
    print(f"  match rate: {match_rate:.4f} ({n_match:,}/{n_total:,})")
    print(f"  max abs diff: {diffs.max():.6f}")
    # Pass = ≥99.5% match (allow 0.5% for first-trade fallback edge cases)
    status = "PASS" if match_rate >= 0.995 else "FAIL"
    _record("D1", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T1.3  D2 — rolling features use closed='left'
# ---------------------------------------------------------------------------


def t1_3_rolling_audit():
    _section("T1.3  D2 — rolling closed='left' grep audit")
    src = (ROOT / "scripts" / "06b_engineer_features.py").read_text()

    # Find all .rolling( calls and check for closed= kwarg
    rolling_calls = []
    for m in re.finditer(r"\.rolling\([^)]*\)", src):
        line_no = src[: m.start()].count("\n") + 1
        snippet = m.group(0)
        rolling_calls.append({"line": line_no, "snippet": snippet})

    safe = []
    suspicious = []
    for call in rolling_calls:
        if "closed=" in call["snippet"]:
            if "'left'" in call["snippet"] or '"left"' in call["snippet"]:
                safe.append(call)
            else:
                suspicious.append({**call, "reason": "closed= but not 'left'"})
        else:
            suspicious.append({**call, "reason": "no closed= kwarg"})

    detail = {
        "total_rolling_calls": len(rolling_calls),
        "safe": safe,
        "suspicious": suspicious,
    }
    print(f"  total .rolling() calls: {len(rolling_calls)}")
    print(f"  with closed='left': {len(safe)}")
    print(f"  suspicious: {len(suspicious)}")
    if suspicious:
        for s in suspicious:
            print(f"    line {s['line']}: {s['snippet']} ← {s['reason']}")
    status = "PASS" if not suspicious else "FAIL"
    _record("D2", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T1.4  D3 — wallet position cumsum spot-check
# ---------------------------------------------------------------------------


def t1_4_wallet_position():
    _section("T1.4  D3 — wallet position cumsum (full-dataset recompute)")
    raw = pd.read_parquet(DATA / "test.parquet")
    feat = pd.read_parquet(DATA / "test_features.parquet")

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
        "mean_abs_diff": float(diffs.mean()),
    }
    print(
        f"  full-dataset match: {n_match:,}/{n_total:,} ({n_match / n_total * 100:.2f}%)"
    )
    print(f"  max abs diff: {diffs.max():.8f}")
    status = "PASS" if (n_match / n_total) >= 0.999 else "FAIL"
    _record("D3", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T1.5  D4 — scaler refit per fold (code audit)
# ---------------------------------------------------------------------------


def t1_5_scaler_audit():
    _section("T1.5  D4 — StandardScaler refit per fold")
    files = [
        "_backtest_worker.py",
        "07_sweep.py",
        "10_backtest.py",
        "06_baseline_idea1.py",
    ]
    findings = []
    for f in files:
        p = ROOT / "scripts" / f
        if not p.exists():
            continue
        src = p.read_text()
        # Find StandardScaler() instantiations
        for m in re.finditer(r"StandardScaler\(\)", src):
            line_no = src[: m.start()].count("\n") + 1
            # Look for context — is there a fit_transform on full data, or per-fold?
            lines = src.splitlines()
            ctx_start = max(0, line_no - 6)
            ctx_end = min(len(lines), line_no + 4)
            ctx = "\n".join(
                f"    {i + 1:4}: {lines[i]}" for i in range(ctx_start, ctx_end)
            )
            findings.append({"file": f, "line": line_no, "context": ctx})

    print(f"  StandardScaler() instantiations found: {len(findings)}")
    suspicious = []
    for f in findings:
        # Heuristic: if the surrounding lines mention 'fit_transform(X_train)' or 'fit_transform(X.iloc[tr]',
        # that's a fold-local fit. If we see fit_transform(X) without slicing, that's risky.
        if re.search(r"fit_transform\(X\)\s*$", f["context"], re.MULTILINE):
            suspicious.append({**f, "reason": "fit_transform(X) — possibly full-data"})

    if suspicious:
        for s in suspicious:
            print(f"  ⚠ {s['file']}:{s['line']} — {s['reason']}")
            print(s["context"])
    else:
        print(
            "  No suspicious patterns. Each StandardScaler appears scoped to a fold or train-only fit."
        )

    detail = {"n_instantiations": len(findings), "suspicious": suspicious}
    status = "PASS" if not suspicious else "FAIL"
    _record("D4", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T1.6  F2 — duplicate transaction_hash check
# ---------------------------------------------------------------------------


def t1_6_dedupe_check():
    _section("T1.6  F2 — duplicate transaction_hash")
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
    return True  # not fatal


# ---------------------------------------------------------------------------
# T1.7  A3 — answer1 == "Yes" across all 75 markets
# ---------------------------------------------------------------------------


def t1_7_answer1_check():
    _section("T1.7  A3 — answer1 == 'Yes' for all 75 markets")
    m = pd.read_parquet(
        DATA / "markets_subset.parquet",
        columns=["id", "question", "answer1", "answer2"],
    )
    bad = m[m["answer1"] != "Yes"]
    detail = {
        "n_markets": len(m),
        "n_bad": len(bad),
        "bad_markets": bad[["id", "question", "answer1", "answer2"]].to_dict(
            orient="records"
        )
        if len(bad)
        else [],
    }
    print(f"  total markets: {len(m)}")
    print(f"  with answer1 != 'Yes': {len(bad)}")
    if len(bad):
        print("  flipped markets:")
        for _, r in bad.iterrows():
            print(f"    {r['id']}: {r['question']} (answer1={r['answer1']})")
    status = "PASS" if len(bad) == 0 else "FAIL"
    _record("A3", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T1.8  C2 — resolution status for 4 dominant test markets
# ---------------------------------------------------------------------------


def t1_8_dominant_resolutions():
    _section("T1.8  C2 — resolution for 4 dominant test markets")
    m = pd.read_parquet(DATA / "markets_subset.parquet")
    m["id"] = m["id"].astype(str)
    dominant_ids = ["1466012", "1466013", "1466014", "1466015"]
    sub = m[m.id.isin(dominant_ids)][
        ["id", "question", "outcome_prices", "answer1", "answer2"]
    ]

    findings = []
    for _, row in sub.iterrows():
        try:
            prices = ast.literal_eval(row["outcome_prices"])
            p1, p2 = float(prices[0]), float(prices[1])
            if abs(p1 - 1.0) < 0.01:
                resolved = "YES (token1)"
                ok = False  # we assumed NO
            elif abs(p2 - 1.0) < 0.01:
                resolved = "NO (token2)"
                ok = True  # matches assumption
            elif p2 > 0.95:
                resolved = f"NO_implied (p2={p2:.4f})"
                ok = True  # close enough
            else:
                resolved = f"AMBIGUOUS (prices={prices})"
                ok = False
        except Exception as e:
            resolved = f"PARSE_ERROR: {e}"
            ok = False
        findings.append(
            {
                "id": row["id"],
                "question": row["question"],
                "outcome_prices": row["outcome_prices"],
                "resolved": resolved,
                "matches_assumption": ok,
            }
        )

    print("  market resolutions:")
    for f in findings:
        flag = "✓" if f["matches_assumption"] else "✗"
        print(f"    {flag} {f['id']} ({f['question']}): {f['resolved']}")
    n_ok = sum(1 for f in findings if f["matches_assumption"])
    detail = {"findings": findings, "n_ok": n_ok, "n_total": len(findings)}
    status = "PASS" if n_ok == len(findings) else "FAIL"
    _record("C2", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T1.9  F1 — HF coverage vs market.volume reconciliation
# ---------------------------------------------------------------------------


def t1_9_volume_reconciliation():
    _section("T1.9  F1 — HF coverage vs market.volume")
    m = pd.read_parquet(
        DATA / "markets_subset.parquet", columns=["id", "question", "volume", "cohort"]
    )
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
    m["extracted_volume"] = extracted
    m["extracted_volume"] = m["extracted_volume"].fillna(0)
    m["ratio"] = m["extracted_volume"] / m["volume"].clip(lower=1)

    # Only flag major markets (>$1M volume) where ratio is far from 1
    major = m[m.volume > 1e6].copy()
    flagged = major[(major.ratio < 0.5) | (major.ratio > 1.5)]

    print(f"  major markets (>${1}M volume): {len(major)}")
    print(f"  flagged (ratio outside [0.5, 1.5]): {len(flagged)}")
    if len(flagged) > 0:
        for idx, row in flagged.iterrows():
            print(
                f"    {idx}: vol=${row['volume'] / 1e6:.1f}M, extracted=${row['extracted_volume'] / 1e6:.1f}M, ratio={row['ratio']:.2f}"
            )
    print(f"  median ratio (major markets): {major.ratio.median():.2f}")

    detail = {
        "n_major": len(major),
        "n_flagged": len(flagged),
        "median_ratio": float(major.ratio.median()),
        "flagged": flagged.reset_index()[
            ["id", "volume", "extracted_volume", "ratio"]
        ].to_dict(orient="records"),
    }
    status = "PASS" if len(flagged) == 0 else "NOTE"
    _record("F1", status, detail)
    return True  # not fatal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Phase 1 — quick wins")
    print("=" * 60)

    fatal_tests = [
        ("C1", t1_1_pre_event_filter),
        ("D1", t1_2_pre_trade_price),
        ("D2", t1_3_rolling_audit),
        ("D3", t1_4_wallet_position),
        ("D4", t1_5_scaler_audit),
    ]
    nonfatal_tests = [
        ("F2", t1_6_dedupe_check),
        ("A3", t1_7_answer1_check),
        ("C2", t1_8_dominant_resolutions),
        ("F1", t1_9_volume_reconciliation),
    ]

    halted = False
    for tid, fn in fatal_tests:
        try:
            ok = fn()
        except Exception as e:
            _record(tid, "ERROR", {"exception": str(e)})
            ok = False
        if not ok:
            print(f"\n*** FATAL FAIL on {tid}. Halting Phase 1. ***")
            halted = True
            break

    if not halted:
        for tid, fn in nonfatal_tests:
            try:
                fn()
            except Exception as e:
                _record(tid, "ERROR", {"exception": str(e)})

    out_path = SCRATCH_OUT / "phase1_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    for tid, r in results.items():
        print(f"  {tid:5} {r['status']}")
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
