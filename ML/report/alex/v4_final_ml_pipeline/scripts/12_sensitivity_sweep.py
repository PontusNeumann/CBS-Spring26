"""
12_sensitivity_sweep.py — robustness of the +14% headline to realism choices.

Sweeps two axes that pressure-test.md flagged as untested (E2, E4):
  - cost_floor ∈ {0.001, 0.01, 0.05, 0.10, 0.20}   (max payoff = 1/floor − 1)
  - copycat-N  ∈ {1, 5, 10, 25, 50, 100}           (1/N share of original trade volume)

For each (floor, N) cell: realistic backtest on $10K bankroll, 5% max bet,
on HistGBM general_ev (the +14% headline strategy). Also runs RF general_ev
and HistGBM top5pct_edge for cross-check.

Outputs:
  alex/outputs/backtest/sensitivity/
    sweep.csv             — all cells (model × strategy × floor × copycats)
    heatmap_<m>_<s>.png   — ROI heatmap per (model, strategy)
    summary.json          — pivot tables
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Pipeline-shared utilities
from _common import (
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

# Capital-aware backtester from sibling script. The number-prefixed module
# name `11_realistic_backtest` is not a valid Python identifier, so we go via
# importlib instead of a plain `from ... import`.
from importlib import import_module as _imp

_rb11 = _imp("11_realistic_backtest")
realistic_backtest = _rb11.realistic_backtest

warnings.filterwarnings("ignore")

SCRATCH = SCRATCH_BASE / "backtest"
OUT = ROOT / "outputs" / "backtest" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

# v4 contract — fail fast if pointed at v3.5 parquets or pre-Stage-1 schema.
TEST_PARQUET = "test_features_v4.parquet"
EXPECTED_N_FEATURES = 76  # 70 v3.5 + 6 wallet

# Sweep grids
COST_FLOORS = [0.001, 0.01, 0.05, 0.10, 0.20]
COPYCAT_NS = [1, 5, 10, 25, 50, 100]

# Fixed bankroll + bet sizing for the sweep
INITIAL_CAPITAL = 10_000
MAX_BET_PCT = 0.05

# (model, strategy) pairs to sweep — keep small to bound runtime.
# LightGBM included so whichever boosting model wins Stage 4's decision-tree
# gate ("LightGBM beats HGBM by > 0.5pp") gets a sensitivity grid.
TARGETS = [
    ("hist_gbm", "general_ev"),  # v3.5 +14% headline
    ("lightgbm", "general_ev"),  # alternative boosting library
    ("random_forest", "general_ev"),
    ("hist_gbm", "top5pct_edge"),
]


def make_strategy(p_hat, edge, cost, name, time_to_deadline_sec):
    if name == "general_ev":
        return general_ev_rule(edge)
    if name == "top5pct_edge":
        return top_k_mask(edge, 0.05)
    if name == "top1pct_phat":
        return top_k_mask(p_hat, 0.01)
    if name == "home_run":
        return home_run_rule(edge, cost, time_to_deadline_sec)
    raise ValueError(name)


def main():
    print("=" * 60)
    print("sensitivity sweep — cost_floor × copycats")
    print(
        f"  bankroll=${INITIAL_CAPITAL:,}, bet_pct={MAX_BET_PCT * 100:.0f}%, "
        f"{len(COST_FLOORS)}×{len(COPYCAT_NS)}={len(COST_FLOORS) * len(COPYCAT_NS)} cells × "
        f"{len(TARGETS)} (model,strategy) = {len(COST_FLOORS) * len(COPYCAT_NS) * len(TARGETS)} runs"
    )
    print("=" * 60)

    # --- v4 data guard ------------------------------------------------------
    test_path = DATA / TEST_PARQUET
    if not test_path.exists():
        raise SystemExit(
            f"v4 parquet missing: {test_path}. Pontus has not delivered, or "
            f"Stage 0 pre-flight was skipped. Run 01_validate_schema.py first."
        )
    fcols = json.loads((DATA / "feature_cols.json").read_text())
    if len(fcols) != EXPECTED_N_FEATURES:
        raise SystemExit(
            f"feature_cols.json has {len(fcols)} features, expected "
            f"{EXPECTED_N_FEATURES}. Run 01_validate_schema.py to update it."
        )

    test = pd.read_parquet(test_path)
    test_raw = pd.read_parquet(DATA / "test.parquet")
    markets = pd.read_parquet(DATA / "markets_subset.parquet")
    res_times = market_resolution_time(markets)

    test_raw["market_id"] = test_raw["market_id"].astype(str)
    test["market_id"] = test["market_id"].astype(str)
    test_raw = test_raw.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    test["_orig_idx"] = np.arange(len(test))
    sort_key = test.sort_values(["market_id", "timestamp"])["_orig_idx"].values
    test = test.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    test["usd_amount"] = test_raw["usd_amount"].values
    test["pre_yes_price_corrected"] = compute_pre_yes_price_corrected(test_raw)

    bet_correct = test["bet_correct"].astype(int).values
    timestamps = test["timestamp"].values
    market_ids = test["market_id"].values
    usd_amount = test["usd_amount"].values
    time_to_deadline_sec = (
        np.exp(test["log_time_to_deadline_hours"].values) - 1
    ) * 3600

    # Load cached preds (deterministic; from 10_backtest.py worker run).
    # Preload every model referenced by TARGETS so missing preds fail loudly,
    # not silently (KeyError on the first lookup).
    n_test = len(test)
    preds_by_model = {}
    for m in sorted({mod for mod, _ in TARGETS}):
        path = SCRATCH / f"preds_{m}.npz"
        if not path.exists():
            print(
                f"  ✗ {m}: no preds at {path} — skipping (TARGETS using it will be dropped)"
            )
            continue
        d = np.load(path)
        cal_orig = d["cal"]
        if len(cal_orig) != n_test:
            raise SystemExit(
                f"[{m}] worker preds length {len(cal_orig)} != n_test {n_test}. "
                f"Sort-key reorder would silently misalign predictions. "
                f"Re-run with RETRAIN=1 (worker may have read a stale parquet)."
            )
        preds_by_model[m] = cal_orig[sort_key]
        print(f"  loaded {m} preds (mean p_hat {preds_by_model[m].mean():.3f})")

    rows = []
    for model_name, strat_name in TARGETS:
        if model_name not in preds_by_model:
            print(f"  · skip ({model_name}, {strat_name}): preds not loaded")
            continue
        p_hat = preds_by_model[model_name]
        for cf in COST_FLOORS:
            cost, edge, _ = compute_cost_and_edge(test, p_hat, cost_floor=cf)
            mask = make_strategy(p_hat, edge, cost, strat_name, time_to_deadline_sec)
            for n_copy in COPYCAT_NS:
                liq = 1.0 / n_copy
                r = realistic_backtest(
                    signal_mask=mask,
                    cost=cost,
                    bet_correct=bet_correct,
                    timestamps=timestamps,
                    market_ids=market_ids,
                    usd_amount=usd_amount,
                    market_res_times=res_times,
                    initial_capital=INITIAL_CAPITAL,
                    max_bet_pct_capital=MAX_BET_PCT,
                    liquidity_scaler=liq,
                )
                rows.append(
                    {
                        "model": model_name,
                        "strategy": strat_name,
                        "cost_floor": cf,
                        "copycats": n_copy,
                        "n_signals": r["n_signals"],
                        "n_executed": r["n_executed"],
                        "final_capital": r["final_capital"],
                        "roi": r["roi"],
                        "max_drawdown": r["max_drawdown"],
                    }
                )
        print(
            f"  ✓ {model_name} / {strat_name}: {len(COST_FLOORS) * len(COPYCAT_NS)} cells"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep.csv", index=False)
    print(f"\n  -> {OUT / 'sweep.csv'}")

    # ---- Heatmaps ---------------------------------------------------------
    print("\n=== ROI (%) heatmaps — rows=cost_floor, cols=copycats ===")
    for (m, s), sub in df.groupby(["model", "strategy"]):
        pivot = sub.pivot_table(
            index="cost_floor", columns="copycats", values="roi", aggfunc="first"
        )
        n_pivot = sub.pivot_table(
            index="cost_floor", columns="copycats", values="n_executed", aggfunc="first"
        )
        print(f"\n--- {m} / {s} — ROI fraction ---")
        print((pivot * 100).round(2).to_string())
        print(f"--- {m} / {s} — n_executed ---")
        print(n_pivot.to_string())

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        roi_pct = pivot.values * 100
        im = ax.imshow(roi_pct, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("copycats sharing fill (N)")
        ax.set_ylabel("cost_floor")
        ax.set_title(f"{m} / {s} — ROI (%) on $10K, 5% max bet")
        for i in range(roi_pct.shape[0]):
            for j in range(roi_pct.shape[1]):
                v = roi_pct[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:+.1f}",
                    ha="center",
                    va="center",
                    color="black" if abs(v) < 30 else "white",
                    fontsize=9,
                )
        plt.colorbar(im, ax=ax, label="ROI %")
        fig.tight_layout()
        path = OUT / f"heatmap_{m}_{s}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"  -> {path.name}")

    # ---- Summary ----------------------------------------------------------
    summary = {}
    for (m, s), sub in df.groupby(["model", "strategy"]):
        sub2 = sub.copy()
        best = sub2.loc[sub2.roi.idxmax()]
        worst = sub2.loc[sub2.roi.idxmin()]
        # Headline cell: cost_floor=0.05, copycats=10 (matches 11_realistic_backtest defaults)
        head = sub2[(sub2.cost_floor == 0.05) & (sub2.copycats == 10)].iloc[0]
        summary[f"{m}/{s}"] = {
            "best": {
                k: float(best[k])
                if k in ("cost_floor", "roi")
                else int(best[k])
                if k in ("copycats", "n_executed", "n_signals")
                else None
                for k in ("cost_floor", "copycats", "roi", "n_executed")
            },
            "worst": {
                k: float(worst[k])
                if k in ("cost_floor", "roi")
                else int(worst[k])
                if k in ("copycats", "n_executed", "n_signals")
                else None
                for k in ("cost_floor", "copycats", "roi", "n_executed")
            },
            "headline_floor005_copy10": {
                "roi": float(head["roi"]),
                "n_executed": int(head["n_executed"]),
                "final_capital": float(head["final_capital"]),
            },
            "roi_range_pct": [float(sub2.roi.min() * 100), float(sub2.roi.max() * 100)],
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  -> {OUT / 'summary.json'}")
    print("\nDONE")


if __name__ == "__main__":
    main()
