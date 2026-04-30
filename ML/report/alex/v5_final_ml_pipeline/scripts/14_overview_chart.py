"""
14_overview_chart.py — single shareable PNG of the realistic backtest results.

Combines:
  - 5 supervised models from outputs/backtest/realistic/sensitivity.csv
  - naive consensus baseline from outputs/backtest/naive_baseline/sensitivity.csv

Filter: $10K bankroll, 5% max bet, no copycats (ls=1.0).

Output: alex/outputs/backtest/overview.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "outputs" / "v5" / "backtest" / "realistic" / "sensitivity.csv"
NAIVE = ROOT / "outputs" / "v5" / "backtest" / "naive_baseline" / "sensitivity.csv"
OUT_PNG = ROOT / "outputs" / "v5" / "backtest" / "overview.png"


def load(scenario_filter: dict) -> pd.DataFrame:
    frames = []
    for path in (ML, NAIVE):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for k, v in scenario_filter.items():
            df = df[df[k] == v]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    headline = {"initial_capital": 10000, "max_bet_pct": 0.05, "liquidity_scaler": 1.0}
    df = load(headline)
    if df.empty:
        raise SystemExit("no rows match headline scenario; check sensitivity.csv files")

    pivot = df.pivot_table(
        index="strategy", columns="model", values="roi", aggfunc="first"
    )

    model_order = [
        "logreg_l2",
        "random_forest",
        "hist_gbm",
        "lightgbm",
        "xgboost",
        "mlp_sklearn",
        "naive_consensus",
    ]
    model_order = [m for m in model_order if m in pivot.columns]
    strategy_order = [
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
    ]
    strategy_order = [s for s in strategy_order if s in pivot.index]
    pivot = pivot.loc[strategy_order, model_order]

    counts = (
        df.pivot_table(
            index="strategy", columns="model", values="n_executed", aggfunc="first"
        )
        .loc[strategy_order, model_order]
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(11, 7.5))

    cmap = LinearSegmentedColormap.from_list(
        "rg",
        [
            (0.0, "#7a1717"),
            (0.4, "#d96a6a"),
            (0.5, "#f0f0f0"),
            (0.6, "#7fcf86"),
            (1.0, "#15703a"),
        ],
    )
    vmin, vmax = -1.0, 0.30
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    img = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")

    for i, _ in enumerate(strategy_order):
        for j, _ in enumerate(model_order):
            val = pivot.values[i, j]
            n = counts.values[i, j]
            if pd.isna(val) or n == 0:
                ax.text(j, i, "—", ha="center", va="center", color="#999", fontsize=9)
                continue
            color = "white" if abs(val) > 0.30 else "#222"
            ax.text(
                j,
                i,
                f"{val * 100:+.0f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )
            ax.text(
                j,
                i + 0.32,
                f"n={n:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                alpha=0.8,
            )

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, fontsize=10)
    ax.set_yticks(range(len(strategy_order)))
    ax.set_yticklabels(strategy_order, fontsize=10)
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, top=True, bottom=False)

    for label in ax.get_xticklabels():
        if label.get_text() == "naive_consensus":
            label.set_fontweight("bold")
            label.set_color("#1565c0")

    cbar = fig.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("ROI (return on $10K)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Realistic backtest — does the model beat a free heuristic?",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    ax.set_title(
        "$10K bankroll · 5% max bet · 20% concentration cap · gas + slippage applied · no copycats\n"
        "Train: 65 Iran-strike markets   →   Test: 10 Iran-ceasefire markets   (cross-regime transfer)",
        fontsize=10,
        color="#555",
        pad=16,
    )

    note = (
        "naive_consensus = free heuristic (bet aligned with market favorite, weighted by consensus strength).\n"
        "phat_gt_0.X = bet whenever model predicts P(win) > X. only MLP and naive routinely produce p_hat > 0.9."
    )
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="#666", style="italic")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"DONE → {OUT_PNG}")


if __name__ == "__main__":
    main()
