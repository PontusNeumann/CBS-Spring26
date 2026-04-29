"""
phase2_falsification.py — Phase 2 of pressure-test plan.

THE high-stakes phase. B1a is most likely to kill the headline.

T2.1  B1a — consensus-vs-contrarian decomposition of top-1% picks
T2.2  B1b — naive consensus baseline
T2.3  A1  — SELL semantics (closing vs open-short)

Outputs:
  alex/.scratch/pressure_tests/phase2_results.json
  alex/.scratch/pressure_tests/plots/b1a_*.png, b1b_*.png
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
from sklearn.metrics import roc_auc_score

from _common import (
    DATA,
    ROOT,
    SCRATCH as SCRATCH_BASE,
    compute_cost_and_edge as compute_cost_edge,
    compute_pre_yes_price_corrected,
)

warnings.filterwarnings("ignore")

SCRATCH_OUT = SCRATCH_BASE / "pressure_tests"
PLOTS = SCRATCH_OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
PRED = SCRATCH_BASE / "backtest"

results: dict = {}


def _section(name):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)


def _record(test_id, status, detail):
    results[test_id] = {"status": status, **detail}
    icon = {"PASS": "✓", "FAIL": "✗", "NOTE": "ℹ"}.get(status, "?")
    print(f"  {icon} [{test_id}] {status}")


def load_test_with_preds():
    """Load test + cached predictions, with consistent sort + corrected pre_yes."""
    test = pd.read_parquet(DATA / "test_features.parquet")
    test["market_id"] = test["market_id"].astype(str)
    test["_orig_idx"] = np.arange(len(test))
    test_sorted = test.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    sort_key = test.sort_values(["market_id", "timestamp"])["_orig_idx"].values

    # Attach corrected pre_yes_price (per-token-price bug fix)
    raw = pd.read_parquet(DATA / "test.parquet")
    test_sorted["pre_yes_price_corrected"] = compute_pre_yes_price_corrected(raw)
    print(
        f"[fix] corrected pre_yes mean {test_sorted['pre_yes_price_corrected'].mean():.3f} "
        f"vs old pre_trade_price mean {test_sorted['pre_trade_price'].mean():.3f}"
    )

    preds = {}
    for m in ["logreg_l2", "random_forest", "hist_gbm"]:
        d = np.load(PRED / f"preds_{m}.npz")
        # Predictions are saved in original (un-sorted) order
        preds[m] = {
            "raw": d["raw"][sort_key],
            "cal": d["cal"][sort_key],
        }
    return test_sorted, preds


# compute_cost_edge is imported from _common as compute_cost_and_edge.

# ---------------------------------------------------------------------------
# T2.1 — B1a — consensus vs contrarian decomposition
# ---------------------------------------------------------------------------


def t2_1_decomposition(test, preds):
    _section("T2.1  B1a — consensus-vs-contrarian decomposition of top-1% picks")
    bet_correct = test["bet_correct"].astype(int).values
    # Use corrected YES probability so consensus_yes = pre_price > 0.5 is meaningful
    pre_price = test["pre_yes_price_corrected"].values
    side_buy = test["side_buy"].values
    outcome_yes = test["outcome_yes"].values
    n = len(test)

    findings = {}
    for model_name, p in preds.items():
        p_hat = p["cal"]
        cost, edge, trader_side_wins_yes = compute_cost_edge(test, p_hat)

        # Top 1% by p_hat
        k = int(n * 0.01)
        top_idx = np.argsort(p_hat)[-k:]

        # Consensus side at trade time = side market thinks will win
        consensus_yes = pre_price > 0.5
        is_with_consensus = trader_side_wins_yes == consensus_yes.astype(int)

        with_pct = is_with_consensus[top_idx].mean()
        against_pct = 1 - with_pct

        # Hit rate breakdown
        with_mask = top_idx[is_with_consensus[top_idx] == 1]
        against_mask = top_idx[is_with_consensus[top_idx] == 0]
        with_hit = bet_correct[with_mask].mean() if len(with_mask) > 0 else None
        against_hit = (
            bet_correct[against_mask].mean() if len(against_mask) > 0 else None
        )

        # Distribution of pre_trade_price for picks
        pp = pre_price[top_idx]

        # Market concentration
        markets_in_top = test.iloc[top_idx]["market_id"].value_counts()

        findings[model_name] = {
            "n_picks": int(k),
            "pct_with_consensus": float(with_pct),
            "pct_against_consensus": float(against_pct),
            "with_consensus_hit_rate": float(with_hit)
            if with_hit is not None
            else None,
            "against_consensus_hit_rate": float(against_hit)
            if against_hit is not None
            else None,
            "n_unique_markets": int(markets_in_top.nunique()),
            "top_market_concentration": float(markets_in_top.iloc[0] / k)
            if len(markets_in_top) > 0
            else 0,
            "pre_price_mean": float(pp.mean()),
            "pre_price_median": float(np.median(pp)),
        }

        print(f"\n  [{model_name}]")
        print(f"    top-1% n_picks: {k}")
        print(f"    WITH consensus: {with_pct * 100:.1f}% (hit rate: {with_hit:.3f})")
        print(
            f"    AGAINST consensus: {against_pct * 100:.1f}% (hit rate: {against_hit if against_hit is not None else 0:.3f})"
        )
        print(
            f"    unique markets: {markets_in_top.nunique()}, top market: {markets_in_top.iloc[0]}/{k} ({markets_in_top.iloc[0] / k * 100:.0f}%)"
        )
        print(
            f"    pre_price distribution: mean={pp.mean():.3f}, median={np.median(pp):.3f}"
        )

    # Plot for RF
    p_hat_rf = preds["random_forest"]["cal"]
    cost_rf, edge_rf, trader_yes_rf = compute_cost_edge(test, p_hat_rf)
    k = int(n * 0.01)
    top_idx_rf = np.argsort(p_hat_rf)[-k:]
    pp_top = pre_price[top_idx_rf]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(pre_price, bins=50, alpha=0.5, label="all test trades", color="grey")
    axes[0].hist(pp_top, bins=50, alpha=0.7, label="top 1% (RF)", color="steelblue")
    axes[0].axvline(0.5, color="red", ls="--", lw=0.7)
    axes[0].set_xlabel("pre_yes_price (corrected — YES probability)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Pre-trade price distribution: all vs top-1%")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Consensus alignment as a stacked bar per model
    models_order = ["logreg_l2", "random_forest", "hist_gbm"]
    with_pcts = [findings[m]["pct_with_consensus"] * 100 for m in models_order]
    against_pcts = [findings[m]["pct_against_consensus"] * 100 for m in models_order]
    x = np.arange(len(models_order))
    axes[1].bar(x, with_pcts, label="WITH consensus", color="seagreen")
    axes[1].bar(
        x, against_pcts, bottom=with_pcts, label="AGAINST consensus", color="firebrick"
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models_order, rotation=15)
    axes[1].set_ylabel("% of top-1% picks")
    axes[1].set_title("Consensus alignment of top-1% picks")
    for i, w in enumerate(with_pcts):
        axes[1].text(i, w / 2, f"{w:.0f}%", ha="center", color="white", weight="bold")
        axes[1].text(
            i,
            w + (against_pcts[i]) / 2,
            f"{against_pcts[i]:.0f}%",
            ha="center",
            color="white",
            weight="bold",
        )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "b1a_decomposition.png", dpi=140)
    plt.close(fig)
    print(f"\n  -> {PLOTS / 'b1a_decomposition.png'}")

    # Verdict: if >90% with-consensus across all models, claim is "consensus detection" not "asymmetric info"
    all_high_consensus = all(f["pct_with_consensus"] > 0.90 for f in findings.values())
    status = "FAIL" if all_high_consensus else "PASS"
    detail = {"per_model": findings, "all_high_consensus": all_high_consensus}
    _record("B1a", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T2.2 — B1b — naive consensus baseline
# ---------------------------------------------------------------------------


def t2_2_naive_baseline(test, preds):
    _section("T2.2  B1b — naive consensus baseline vs model")
    bet_correct = test["bet_correct"].astype(int).values
    pre_price = test["pre_yes_price_corrected"].values
    side_buy = test["side_buy"].values
    outcome_yes = test["outcome_yes"].values
    n = len(test)

    # Naive predictor 1: score = how aligned trade is with market consensus
    # if trader bets on dominant side, score = max(p, 1-p), else 1 - that
    trader_side_wins_yes = side_buy * outcome_yes + (1 - side_buy) * (1 - outcome_yes)
    consensus_yes = pre_price > 0.5
    is_with = trader_side_wins_yes == consensus_yes.astype(int)
    consensus_strength = np.maximum(pre_price, 1 - pre_price)
    naive_score = np.where(is_with, consensus_strength, 1 - consensus_strength)

    # Naive predictor 2: extreme-confidence consensus only
    # score = consensus_strength if trader is on dominant side, else 0
    naive_score_extreme = np.where(is_with, consensus_strength, 0.0)

    # Compare top-1% precision and AUC for naive vs models
    k = int(n * 0.01)
    table = []
    for label, score in [
        ("naive_consensus_aligned", naive_score),
        ("naive_extreme_confidence", naive_score_extreme),
        ("logreg_l2", preds["logreg_l2"]["cal"]),
        ("random_forest", preds["random_forest"]["cal"]),
        ("hist_gbm", preds["hist_gbm"]["cal"]),
    ]:
        try:
            auc = float(roc_auc_score(bet_correct, score))
        except Exception:
            auc = None
        top_idx = np.argsort(score)[-k:]
        top_hit = float(bet_correct[top_idx].mean())
        table.append({"strategy": label, "auc": auc, "top_1pct_precision": top_hit})

    print(f"\n  Comparison ({k} trades = top 1%):")
    print(f"  {'strategy':<28} {'AUC':>7} {'top-1%':>9}")
    print(f"  {'-' * 28:<28} {'-' * 7:>7} {'-' * 9:>9}")
    for r in table:
        auc_str = f"{r['auc']:.3f}" if r["auc"] is not None else "—"
        print(f"  {r['strategy']:<28} {auc_str:>7} {r['top_1pct_precision']:>9.3f}")

    # Verdict: if naive matches or beats model, model adds no signal
    model_aucs = [
        r["auc"]
        for r in table
        if r["strategy"] in ("random_forest", "hist_gbm") and r["auc"] is not None
    ]
    naive_auc = next(
        (r["auc"] for r in table if r["strategy"] == "naive_consensus_aligned"), None
    )
    naive_extreme_auc = next(
        (r["auc"] for r in table if r["strategy"] == "naive_extreme_confidence"), None
    )
    model_top = max(
        r["top_1pct_precision"]
        for r in table
        if r["strategy"] in ("random_forest", "hist_gbm")
    )
    naive_top = next(
        (
            r["top_1pct_precision"]
            for r in table
            if r["strategy"] == "naive_consensus_aligned"
        ),
        0,
    )

    naive_matches = (naive_auc is not None and max(model_aucs) - naive_auc < 0.02) or (
        naive_top >= 0.95
    )

    status = "FAIL" if naive_matches else "PASS"
    detail = {"comparison": table, "naive_matches_model": naive_matches}
    _record("B1b", status, detail)
    return status == "PASS"


# ---------------------------------------------------------------------------
# T2.3 — A1 — SELL semantics
# ---------------------------------------------------------------------------


def t2_3_sell_semantics(test, preds):
    _section("T2.3  A1 — SELL semantics (closing vs open-short)")

    raw_test = pd.read_parquet(DATA / "test.parquet")
    raw_test["market_id"] = raw_test["market_id"].astype(str)
    raw_test = raw_test.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    # For each SELL trade by taker T in market M, did T have a prior BUY of the same nonusdc_side?
    raw_test["taker_str"] = raw_test["taker"].astype(str)
    raw_test["is_sell"] = raw_test["taker_direction"].astype(str).str.upper().eq("SELL")
    raw_test["is_buy"] = raw_test["taker_direction"].astype(str).str.upper().eq("BUY")
    raw_test["nonusdc_str"] = raw_test["nonusdc_side"].astype(str)

    # For each row, count prior BUYs by the same (market, taker, nonusdc_side)
    grp = raw_test.groupby(["market_id", "taker_str", "nonusdc_str"])
    raw_test["prior_buys_same_side"] = grp["is_buy"].cumsum().shift(1).fillna(0)
    # Reset shift carryover at group boundaries
    first_idx = grp.head(1).index
    raw_test.loc[first_idx, "prior_buys_same_side"] = 0

    sell_mask = raw_test["is_sell"]
    n_sells = int(sell_mask.sum())
    sells = raw_test[sell_mask]
    n_closing = int((sells["prior_buys_same_side"] >= 1).sum())
    n_open_short = n_sells - n_closing
    pct_closing = n_closing / max(n_sells, 1)

    print(f"  total trades:      {len(raw_test):,}")
    print(f"  total SELLs:       {n_sells:,}")
    print(
        f"    closing (prior BUY same side):  {n_closing:,} ({pct_closing * 100:.1f}%)"
    )
    print(
        f"    open-short (no prior BUY):      {n_open_short:,} ({(1 - pct_closing) * 100:.1f}%)"
    )

    # How many SELLs in the top-1% picks (by RF p_hat)?
    p_hat_rf = preds["random_forest"]["cal"]
    n = len(test)
    k = int(n * 0.01)
    top_idx = np.argsort(p_hat_rf)[-k:]
    # We need to map test idx -> raw_test idx. They share sort by (market_id, timestamp).
    # Both should have same length and ordering after sort.
    assert len(test) == len(raw_test), "test and raw_test row counts differ"

    top_taker_dir = raw_test.iloc[top_idx]["taker_direction"].astype(str).str.upper()
    top_n_sell = int((top_taker_dir == "SELL").sum())
    top_n_buy = int((top_taker_dir == "BUY").sum())
    top_sells_closing = int(
        (
            (top_taker_dir == "SELL")
            & (raw_test.iloc[top_idx]["prior_buys_same_side"] >= 1)
        ).sum()
    )

    print(f"\n  Top-1% picks (n={k}, RF):")
    print(f"    BUYs:                         {top_n_buy} ({top_n_buy / k * 100:.1f}%)")
    print(
        f"    SELLs:                        {top_n_sell} ({top_n_sell / k * 100:.1f}%)"
    )
    print(
        f"    SELLs that are closing trades: {top_sells_closing} ({top_sells_closing / max(top_n_sell, 1) * 100:.1f}% of SELLs)"
    )

    detail = {
        "n_sells": n_sells,
        "n_sells_closing": n_closing,
        "pct_closing": float(pct_closing),
        "top_1pct_n_buys": top_n_buy,
        "top_1pct_n_sells": top_n_sell,
        "top_1pct_sells_closing": top_sells_closing,
    }
    # Verdict: if >50% of SELLs are closing, treating them as fresh directional bets is wrong
    status = "FAIL" if pct_closing > 0.50 else "NOTE" if pct_closing > 0.25 else "PASS"
    _record("A1", status, detail)
    return True  # not stopping on this — informational


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Phase 2 — claim falsification")
    print("=" * 60)
    test, preds = load_test_with_preds()
    print(f"test rows: {len(test):,}, models loaded: {list(preds.keys())}")

    t2_1_decomposition(test, preds)
    t2_2_naive_baseline(test, preds)
    t2_3_sell_semantics(test, preds)

    out_path = SCRATCH_OUT / "phase2_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    for tid, r in results.items():
        print(f"  {tid:5} {r['status']}")
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
