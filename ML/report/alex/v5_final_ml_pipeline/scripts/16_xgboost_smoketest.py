"""
16_xgboost_smoketest.py — quick eval of XGBoost vs other tree models.

No tuning. Default hyperparameters baked into _backtest_worker.make_xgboost.
Reports: test AUC (raw + calibrated), top-1% precision, per-market AUC range,
and headline-scenario realistic-backtest ROI across 10 strategies.

Outputs:
  outputs/v5/backtest/xgboost_smoketest.csv
  outputs/v5/backtest/xgboost_smoketest.md
"""

from __future__ import annotations

import json
import sys
import warnings
from importlib import import_module as _imp
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    DATA,
    ROOT,
    SCRATCH as SCRATCH_BASE,
    compute_cost_and_edge,
    compute_pre_yes_price_corrected,
    general_ev_rule,
    home_run_rule,
    market_resolution_time,
    top_k_mask,
)

_rb11 = _imp("11_realistic_backtest")
realistic_backtest = _rb11.realistic_backtest

warnings.filterwarnings("ignore")

OUT = ROOT / "outputs" / "v5" / "backtest"
OUT.mkdir(parents=True, exist_ok=True)
PREDS = SCRATCH_BASE / "backtest"

TEST_PARQUET = "test_features_v4.parquet"
EXPECTED_N_FEATURES = 64
COMPARE_MODELS = ["hist_gbm", "lightgbm", "random_forest", "mlp_sklearn", "xgboost"]


def main():
    print("=" * 60)
    print("XGBoost smoke test (default hyperparameters)")
    print("=" * 60)

    test = pd.read_parquet(DATA / TEST_PARQUET)
    test_raw = pd.read_parquet(DATA / "test.parquet")
    markets = pd.read_parquet(DATA / "markets_subset.parquet")
    res_times = market_resolution_time(markets)

    test_raw["market_id"] = test_raw["market_id"].astype(str)
    test["market_id"] = test["market_id"].astype(str)
    test_raw = test_raw.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    test_orig = test.copy()
    test_orig["_orig_idx"] = np.arange(len(test_orig))
    sort_key = test_orig.sort_values(["market_id", "timestamp"])["_orig_idx"].values
    test = test.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    assert len(test) == len(test_raw)
    test["usd_amount"] = test_raw["usd_amount"].values
    test["pre_yes_price_corrected"] = compute_pre_yes_price_corrected(test_raw)

    bet_correct = test["bet_correct"].astype(int).values
    timestamps = test["timestamp"].values
    market_ids = test["market_id"].values
    usd_amount = test["usd_amount"].values
    n_test = len(test)
    time_to_deadline_sec = (
        np.exp(test["log_time_to_deadline_hours"].values) - 1
    ) * 3600

    # 1. Quick AUC eval per model
    print("\n[1/2] AUC + top-1% precision (test cohort, 257K trades, 10 markets)")
    print("=" * 60)
    auc_rows = []
    for m in COMPARE_MODELS:
        path = PREDS / f"preds_{m}.npz"
        if not path.exists():
            print(f"  ✗ {m}: preds missing at {path}")
            continue
        d = np.load(path)
        raw = d["raw"][sort_key]
        cal = d["cal"][sort_key]

        auc_raw = float(roc_auc_score(bet_correct, raw))
        auc_cal = float(roc_auc_score(bet_correct, cal))

        # Top-1% precision (rank-based, calibration-invariant)
        k = max(1, int(n_test * 0.01))
        top_idx = np.argpartition(cal, -k)[-k:]
        top1_prec = float(bet_correct[top_idx].mean())

        # Per-market AUC range
        per_market = []
        for mid in test["market_id"].unique():
            mask = (test["market_id"] == mid).values
            try:
                per_market.append(float(roc_auc_score(bet_correct[mask], cal[mask])))
            except ValueError:
                continue

        row = {
            "model": m,
            "auc_raw": auc_raw,
            "auc_cal": auc_cal,
            "top1pct_precision": top1_prec,
            "per_market_auc_min": min(per_market) if per_market else None,
            "per_market_auc_max": max(per_market) if per_market else None,
        }
        auc_rows.append(row)
        print(
            f"  {m:15s}  raw={auc_raw:.4f}  cal={auc_cal:.4f}  "
            f"top1%={top1_prec:.3f}  per-market=[{min(per_market):.2f}, {max(per_market):.2f}]"
        )

    # 2. Headline-scenario realistic backtest (xgboost only — quick check)
    print("\n[2/2] Realistic-backtest ROI ($10K, 5%, no copycats, ls=1.0) per strategy")
    print("=" * 60)
    backtest_rows = []
    for m in COMPARE_MODELS:
        path = PREDS / f"preds_{m}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        cal = d["cal"][sort_key]
        cost, edge, _ = compute_cost_and_edge(test, cal)

        masks = {
            "general_ev": general_ev_rule(edge),
            "home_run": home_run_rule(edge, cost, time_to_deadline_sec),
            "top1pct_phat": top_k_mask(cal, 0.01),
            "top1pct_edge": top_k_mask(edge, 0.01),
            "top5pct_edge": top_k_mask(edge, 0.05),
            "phat_gt_0.9": cal > 0.9,
            "phat_gt_0.95": cal > 0.95,
            "phat_gt_0.99": cal > 0.99,
            "general_ev_cheap": general_ev_rule(edge) & (cost < 0.30),
            "general_ev_late": general_ev_rule(edge) & (time_to_deadline_sec < 86400),
        }
        for s_name, mask in masks.items():
            r = realistic_backtest(
                signal_mask=mask,
                cost=cost,
                bet_correct=bet_correct,
                timestamps=timestamps,
                market_ids=market_ids,
                usd_amount=usd_amount,
                market_res_times=res_times,
                initial_capital=10_000,
                max_bet_pct_capital=0.05,
                liquidity_scaler=1.0,
            )
            backtest_rows.append(
                {
                    "model": m,
                    "strategy": s_name,
                    "n_executed": r["n_executed"],
                    "final_capital": r["final_capital"],
                    "roi": r["roi"],
                    "max_drawdown": r["max_drawdown"],
                }
            )
        print(f"  ✓ {m}")

    bt = pd.DataFrame(backtest_rows)
    bt.to_csv(OUT / "xgboost_smoketest.csv", index=False)
    print(f"\n  → {OUT / 'xgboost_smoketest.csv'}")

    # Compact markdown summary
    auc_df = pd.DataFrame(auc_rows)
    pivot_roi = bt.pivot_table(
        index="strategy", columns="model", values="roi", aggfunc="first"
    ).reindex(
        [
            "phat_gt_0.99",
            "phat_gt_0.95",
            "phat_gt_0.9",
            "top1pct_phat",
            "top1pct_edge",
            "top5pct_edge",
            "general_ev",
            "general_ev_late",
            "general_ev_cheap",
            "home_run",
        ],
        columns=[m for m in COMPARE_MODELS if m in bt.model.values],
    )

    md_lines = ["# XGBoost smoke test", ""]
    md_lines.append("## AUC + top-1% precision (cleaned 64-feat schema, test cohort)")
    md_lines.append("")
    md_lines.append("| model | raw AUC | cal AUC | top-1% precision | per-market AUC |")
    md_lines.append("|---|---:|---:|---:|---|")
    for _, r in auc_df.iterrows():
        rng = (
            f"[{r['per_market_auc_min']:.2f}, {r['per_market_auc_max']:.2f}]"
            if r["per_market_auc_min"] is not None
            else "—"
        )
        md_lines.append(
            f"| {r['model']} | {r['auc_raw']:.4f} | {r['auc_cal']:.4f} | "
            f"{r['top1pct_precision']:.3f} | {rng} |"
        )

    md_lines += [
        "",
        "## Realistic-backtest ROI ($10K, 5% bet, no copycats, 31-day cross-regime test)",
        "",
        "| strategy | " + " | ".join(pivot_roi.columns) + " |",
        "|---" + " | ---:" * len(pivot_roi.columns) + "|",
    ]
    for s in pivot_roi.index:
        cells = []
        for m in pivot_roi.columns:
            v = pivot_roi.loc[s, m]
            cells.append("—" if pd.isna(v) else f"{v * 100:+.1f}%")
        md_lines.append(f"| `{s}` | " + " | ".join(cells) + " |")

    md_lines += [
        "",
        "## Caveats",
        "",
        "- XGBoost trained 2026-04-30 on the 64-feature cleaned schema (D-042 cohort-flip features removed).",
        "- The hist_gbm / lightgbm / random_forest cached preds in `.scratch/backtest/` are dated Apr 27 — **before** the schema cleanup. Their raw AUCs are inflated by cohort-flip features. XGBoost is being compared against a stronger-than-real baseline.",
        "- mlp_sklearn cached preds are from Apr 29 (post-cleanup, fair comparison).",
        "- No hyperparameter tuning. Defaults: n_estimators=400, max_depth=8, learning_rate=0.05, min_child_weight=200, tree_method=hist.",
    ]
    (OUT / "xgboost_smoketest.md").write_text("\n".join(md_lines) + "\n")
    print(f"  → {OUT / 'xgboost_smoketest.md'}")
    print("\nDONE")


if __name__ == "__main__":
    main()
