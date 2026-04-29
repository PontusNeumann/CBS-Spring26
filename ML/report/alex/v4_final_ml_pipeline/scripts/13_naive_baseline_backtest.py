"""
13_naive_baseline_backtest.py — realistic backtest on naive consensus baseline.

Answers: under identical capital / fee / fill-share constraints, how does the
naive 'follow the consensus' rule (Phase 2 B1b, AUC 0.844 on the ceasefire test
set) compare to the supervised models?

Predictor: naive_consensus_aligned p_hat.
  consensus_yes      = pre_yes_price_corrected > 0.5
  trader_side_wins_yes = side_buy * outcome_yes + (1-side_buy) * (1-outcome_yes)
  is_with_consensus  = trader_side_wins_yes == consensus_yes
  consensus_strength = max(p_yes, 1 - p_yes)
  p_hat_naive        = where(is_with, consensus_strength, 1 - consensus_strength)

Mirrors 11_realistic_backtest.py structure: same strategy masks, same realistic
execution engine, same (capital × bet_pct × fill_share) grid. Drop-in compare.

Outputs:
  alex/outputs/backtest/naive_baseline/
    summary.json
    sensitivity.csv
    capital_curves.png
"""

from __future__ import annotations

import json
import warnings
from importlib import import_module as _imp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    DATA,
    LIQUIDITY_SCALER_DEFAULT as LIQUIDITY_SCALER,
    ROOT,
    compute_cost_and_edge,
    compute_pre_yes_price_corrected,
    general_ev_rule,
    home_run_rule,
    market_resolution_time,
    top_k_mask,
)

# Reuse the realistic execution engine from 11.
_rb11 = _imp("11_realistic_backtest")
realistic_backtest = _rb11.realistic_backtest

warnings.filterwarnings("ignore")

OUT = ROOT / "outputs" / "backtest" / "naive_baseline"
OUT.mkdir(parents=True, exist_ok=True)

TEST_PARQUET = "test_features_v4.parquet"
EXPECTED_N_FEATURES = 64


def naive_consensus_aligned(test: pd.DataFrame) -> np.ndarray:
    """Return p_hat per trade from the naive consensus rule (B1b)."""
    p_yes = test["pre_yes_price_corrected"].values
    side_buy = test["side_buy"].values
    outcome_yes = test["outcome_yes"].values
    trader_side_wins_yes = side_buy * outcome_yes + (1 - side_buy) * (1 - outcome_yes)
    consensus_yes = (p_yes > 0.5).astype(int)
    is_with = (trader_side_wins_yes == consensus_yes).astype(int)
    consensus_strength = np.maximum(p_yes, 1 - p_yes)
    return np.where(is_with == 1, consensus_strength, 1 - consensus_strength)


def main():
    print("=" * 60)
    print("naive consensus baseline — realistic backtest")
    print("=" * 60)

    test_path = DATA / TEST_PARQUET
    if not test_path.exists():
        raise SystemExit(f"v4 parquet missing: {test_path}")
    fcols = json.loads((DATA / "feature_cols.json").read_text())
    if len(fcols) != EXPECTED_N_FEATURES:
        raise SystemExit(
            f"feature_cols.json has {len(fcols)} features, expected "
            f"{EXPECTED_N_FEATURES}"
        )

    test = pd.read_parquet(test_path)
    test_raw = pd.read_parquet(DATA / "test.parquet")
    markets = pd.read_parquet(DATA / "markets_subset.parquet")
    res_times = market_resolution_time(markets)

    test_raw["market_id"] = test_raw["market_id"].astype(str)
    test["market_id"] = test["market_id"].astype(str)
    test_raw = test_raw.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    test = test.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    assert len(test) == len(test_raw)
    test["usd_amount"] = test_raw["usd_amount"].values
    test["pre_yes_price_corrected"] = compute_pre_yes_price_corrected(test_raw)
    print(f"[fix] corrected pre_yes mean {test['pre_yes_price_corrected'].mean():.3f}")

    bet_correct = test["bet_correct"].astype(int).values
    timestamps = test["timestamp"].values
    market_ids = test["market_id"].values
    usd_amount = test["usd_amount"].values
    n_test = len(test)
    print(f"test trades: {n_test:,}")

    time_to_deadline_sec = (
        np.exp(test["log_time_to_deadline_hours"].values) - 1
    ) * 3600

    # Naive p_hat → cost / edge
    p_hat = naive_consensus_aligned(test)
    cost, edge, _ = compute_cost_and_edge(test, p_hat)
    print(
        f"naive p_hat: mean {p_hat.mean():.3f}, "
        f"acc@0.5 {((p_hat > 0.5) == bet_correct).mean():.3f}"
    )
    from sklearn.metrics import roc_auc_score

    print(f"naive AUC on test: {roc_auc_score(bet_correct, p_hat):.3f}")

    strategies = {
        "general_ev": general_ev_rule(edge),
        "home_run": home_run_rule(edge, cost, time_to_deadline_sec),
        "top1pct_phat": top_k_mask(p_hat, 0.01),
        "top1pct_edge": top_k_mask(edge, 0.01),
        "top5pct_edge": top_k_mask(edge, 0.05),
        "phat_gt_0.9": p_hat > 0.9,
        "phat_gt_0.95": p_hat > 0.95,
        "phat_gt_0.99": p_hat > 0.99,
        "general_ev_cheap": general_ev_rule(edge) & (cost < 0.30),
        "general_ev_late": general_ev_rule(edge) & (time_to_deadline_sec < 86400),
    }
    for s_name, mask in strategies.items():
        print(
            f"  {s_name:<14}: {mask.sum():>6} signals, "
            f"hit {bet_correct[mask].mean() if mask.sum() else 0:.3f}"
        )

    capital_grid = [1_000, 10_000, 100_000, 1_000_000]
    bet_pct_grid = [0.01, 0.05, 0.10]
    fill_share_grid = [1.0, 0.10]

    summary = {"strategies": {}}
    sensitivity_rows = []
    curves_for_plot = {}

    for s_name, mask in strategies.items():
        scenarios = {}
        for cap in capital_grid:
            for bet_pct in bet_pct_grid:
                for ls in fill_share_grid:
                    r = realistic_backtest(
                        signal_mask=mask,
                        cost=cost,
                        bet_correct=bet_correct,
                        timestamps=timestamps,
                        market_ids=market_ids,
                        usd_amount=usd_amount,
                        market_res_times=res_times,
                        initial_capital=cap,
                        max_bet_pct_capital=bet_pct,
                        liquidity_scaler=ls,
                    )
                    ls_label = "no_fill_share" if ls >= 1.0 else f"ls{ls:g}"
                    scenarios[f"cap{cap}_bet{int(bet_pct * 100)}pct_{ls_label}"] = {
                        k: v
                        for k, v in r.items()
                        if k not in ("capital_curve", "trades_executed")
                    }
                    sensitivity_rows.append(
                        {
                            "model": "naive_consensus",
                            "strategy": s_name,
                            "initial_capital": cap,
                            "max_bet_pct": bet_pct,
                            "liquidity_scaler": ls,
                            "n_signals": r["n_signals"],
                            "n_executed": r["n_executed"],
                            "final_capital": r["final_capital"],
                            "roi": r["roi"],
                            "max_drawdown": r["max_drawdown"],
                            "skipped_capital": r["skipped_capital"],
                            "skipped_concentration": r["skipped_concentration"],
                        }
                    )
                    if (
                        cap == 10_000
                        and bet_pct == 0.05
                        and ls >= 1.0
                        and s_name in ("general_ev", "top1pct_edge", "top5pct_edge")
                    ):
                        curves_for_plot[f"naive_{s_name}_no_fill_share"] = r[
                            "capital_curve"
                        ]
        summary["strategies"][s_name] = scenarios

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df.to_csv(OUT / "sensitivity.csv", index=False)

    for ls_val, label in [(1.0, "default no copycats"), (0.10, "10× copycats stress")]:
        print("\n" + "=" * 80)
        print(f"NAIVE — $10K, 5% bet — {label} (ls={ls_val})")
        print("=" * 80)
        h = sens_df[
            (sens_df.initial_capital == 10_000)
            & (sens_df.max_bet_pct == 0.05)
            & (sens_df.liquidity_scaler == ls_val)
        ]
        print(
            h[["strategy", "n_executed", "final_capital", "roi", "max_drawdown"]]
            .round(2)
            .to_string(index=False)
        )

    if curves_for_plot:
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, curve in curves_for_plot.items():
            ts = [pd.to_datetime(t, unit="s") for t, _ in curve]
            v = [v for _, v in curve]
            ax.plot(ts, v, label=label, lw=1.0)
        ax.axhline(10_000, color="black", ls="--", lw=0.5, label="initial $10K")
        ax.set_xlabel("trade timestamp")
        ax.set_ylabel("capital ($)")
        ax.set_title("Naive consensus baseline — $10K, 5% bet, no fill share")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUT / "capital_curves.png", dpi=140)
        plt.close(fig)
        print(f"\n  -> {OUT / 'capital_curves.png'}")

    print(f"\nDONE — outputs in {OUT}")


if __name__ == "__main__":
    main()
