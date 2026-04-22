"""
04_naive_market_investigation.py

Answers: is the naive-market baseline (p_hat = market_implied_prob)'s
test ROC 0.63 a real signal, or a structural artifact of the joint
(side, outcomeIndex, market_implied_prob, resolution) distribution?

Hypothesis to falsify: within each single market × (side, outcomeIndex)
subgroup, `bet_correct` is constant (all 1s or all 0s) because resolution
is fixed per market and winning-side is determined by (side, outcomeIndex).
So market_implied_prob CANNOT predict bet_correct within any subgroup.
The 0.63 aggregate test ROC must therefore come from RANKING BETWEEN
subgroups — i.e., bet_correct=1 subgroups happen to have systematically
different market_implied_prob than bet_correct=0 subgroups.

That's Simpson's paradox territory: an aggregate correlation driven by
group-level means, not within-group predictive value. If confirmed,
naive_market's 0.63 isn't a benchmark our trained models should be
compared to head-on.

Produces a markdown report + calibration plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "experiments"
OUT_DIR = ROOT / "alex" / "outputs" / "investigations" / "naive_market"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_roc(y, p):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def describe_subgroup(df, name, lines):
    df = df.copy()
    df["side_bool"] = df["side"].astype(str).str.upper().eq("BUY").astype(int)
    df["subgroup"] = df.apply(
        lambda r: ("BUY" if r["side_bool"] else "SELL")
        + "_idx"
        + str(int(r["outcomeIndex"])),
        axis=1,
    )

    lines.append(f"\n## {name} — side × outcomeIndex subgroup breakdown\n")
    lines.append(
        "| Subgroup | Interpretation | n | bc_rate | p_hat mean | p_hat std | ROC (naive) |\n"
        "|---|---|---:|---:|---:|---:|---:|"
    )
    interp = {
        "BUY_idx0": "BUY YES",
        "BUY_idx1": "BUY NO",
        "SELL_idx0": "SELL YES",
        "SELL_idx1": "SELL NO",
    }
    for sg, g in df.groupby("subgroup"):
        p = g["market_implied_prob"].to_numpy(dtype=float)
        y = g["bet_correct"].astype(int).to_numpy()
        mask = ~np.isnan(p)
        p, y = p[mask], y[mask]
        if len(y) == 0:
            continue
        roc = safe_roc(y, p)
        roc_str = f"{roc:.4f}" if roc is not None else "—"
        lines.append(
            f"| `{sg}` | {interp.get(sg, '?')} | {len(y):,} | {y.mean():.3f} | "
            f"{p.mean():.3f} | {p.std():.3f} | {roc_str} |"
        )


def per_market_naive(df, lines, split_name):
    lines.append(f"\n## {split_name} — per-market naive ROC\n")
    lines.append(
        "| Market | n | bc_rate | p_hat mean | ROC (naive) |\n|---|---:|---:|---:|---:|"
    )
    rows = []
    for q, g in df.groupby("question"):
        p = g["market_implied_prob"].to_numpy(dtype=float)
        y = g["bet_correct"].astype(int).to_numpy()
        mask = ~np.isnan(p)
        p, y = p[mask], y[mask]
        roc = safe_roc(y, p)
        rows.append((q, len(y), y.mean(), p.mean(), roc))
    rows.sort(key=lambda r: -r[1])
    for q, n, bc, pm, roc in rows:
        roc_str = f"{roc:.4f}" if roc is not None else "—"
        lines.append(f"| {q[:70]} | {n:,} | {bc:.3f} | {pm:.3f} | {roc_str} |")


def simpson_check(df, lines, split_name):
    """For each market × (side, outcomeIndex) subgroup, show bet_correct variance
    and p_hat range. If bet_correct is constant within subgroup, ROC is undefined
    — the aggregate ROC must come from ranking BETWEEN subgroups."""
    lines.append(
        f"\n## {split_name} — within (market × subgroup) bet_correct variance check\n"
    )
    lines.append(
        "If bc_std ≈ 0 for nearly every cell, aggregate ROC can only come from "
        "between-cell ranking. That confirms Simpson's-paradox artifact.\n"
    )
    df = df.copy()
    df["side_bool"] = df["side"].astype(str).str.upper().eq("BUY").astype(int)

    cells = []
    for (q, side_b, oi), g in df.groupby(["question", "side_bool", "outcomeIndex"]):
        y = g["bet_correct"].astype(int).to_numpy()
        p = g["market_implied_prob"].to_numpy(dtype=float)
        mask = ~np.isnan(p)
        p, y = p[mask], y[mask]
        if len(y) < 5:
            continue
        cells.append(
            {
                "market": q,
                "side": "BUY" if side_b else "SELL",
                "outcomeIndex": int(oi),
                "n": int(len(y)),
                "bc_mean": float(y.mean()),
                "bc_std": float(y.std()),
                "p_mean": float(p.mean()),
                "p_std": float(p.std()),
                "within_roc": safe_roc(y, p),
            }
        )
    zero_var = sum(1 for c in cells if c["bc_std"] < 1e-9)
    lines.append(
        f"Cells audited: {len(cells)}. Cells with `bc_std = 0` "
        f"(constant bet_correct within cell): **{zero_var}**.\n"
    )
    lines.append(
        "| Market | Side | idx | n | bc_mean | bc_std | p_mean | Within-cell ROC |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )
    for c in sorted(cells, key=lambda r: -r["n"])[:25]:
        roc_str = f"{c['within_roc']:.4f}" if c["within_roc"] is not None else "—"
        lines.append(
            f"| {c['market'][:50]} | {c['side']} | {c['outcomeIndex']} | {c['n']:,} | "
            f"{c['bc_mean']:.3f} | {c['bc_std']:.3f} | {c['p_mean']:.3f} | {roc_str} |"
        )


def plot_p_by_subgroup(df, path, title):
    df = df.copy()
    df["side_bool"] = df["side"].astype(str).str.upper().eq("BUY").astype(int)
    df["subgroup"] = df.apply(
        lambda r: ("BUY" if r["side_bool"] else "SELL")
        + "_idx"
        + str(int(r["outcomeIndex"])),
        axis=1,
    )
    df["winner"] = df["bet_correct"].astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in [
        ("BUY_idx0", "#3b82f6"),
        ("BUY_idx1", "#10b981"),
        ("SELL_idx0", "#f59e0b"),
        ("SELL_idx1", "#ef4444"),
    ]:
        g = df[df["subgroup"] == name]
        if g.empty:
            continue
        winners = g[g["winner"] == 1]["market_implied_prob"].dropna()
        losers = g[g["winner"] == 0]["market_implied_prob"].dropna()
        bins = np.linspace(0, 1, 31)
        if len(winners) > 0:
            ax.hist(
                winners,
                bins=bins,
                color=color,
                alpha=0.5,
                label=f"{name} winners (n={len(winners):,})",
                density=True,
            )
        if len(losers) > 0:
            ax.hist(
                losers,
                bins=bins,
                color=color,
                alpha=0.15,
                label=f"{name} losers (n={len(losers):,})",
                density=True,
                histtype="step",
                linewidth=1.5,
            )
    ax.set_xlabel("market_implied_prob at trade time")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def calibration_plot(df, path, title):
    df = df.dropna(subset=["market_implied_prob"]).copy()
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(df["market_implied_prob"], bins[1:-1], right=False)
    xs, ys, ns = [], [], []
    for b in range(10):
        mask = idx == b
        if not mask.any():
            continue
        xs.append(df.loc[mask, "market_implied_prob"].mean())
        ys.append(df.loc[mask, "bet_correct"].astype(int).mean())
        ns.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4, label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#3b82f6", label="observed")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(
            f"n={n:,}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7
        )
    ax.set_xlabel("market_implied_prob bin mean")
    ax.set_ylabel("observed bet_correct rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    lines = ["# Naive-market investigation", ""]
    lines.append(
        "_Script: `alex/scripts/04_naive_market_investigation.py`. Purpose: "
        "determine whether the naive-market baseline's test ROC 0.63 is a "
        "real signal or a Simpson's-paradox artifact of the subgroup structure._\n"
    )

    # Top-level ROC on each split
    lines.append("## Top-level naive-market ROC (recap)\n")
    lines.append(
        "| Split | n | bc_rate | p_hat mean | ROC |\n|---|---:|---:|---:|---:|"
    )
    for name, df in [("train", train), ("val", val), ("test", test)]:
        p = df["market_implied_prob"].to_numpy(dtype=float)
        y = df["bet_correct"].astype(int).to_numpy()
        mask = ~np.isnan(p)
        p, y = p[mask], y[mask]
        roc = safe_roc(y, p)
        roc_str = f"{roc:.4f}" if roc is not None else "—"
        lines.append(
            f"| {name} | {len(y):,} | {y.mean():.3f} | {p.mean():.3f} | {roc_str} |"
        )

    # Side × outcomeIndex subgroup breakdown per split
    describe_subgroup(train, "train", lines)
    describe_subgroup(val, "val", lines)
    describe_subgroup(test, "test", lines)

    # Per-market
    per_market_naive(test, lines, "test")

    # Simpson's check: within-cell bet_correct variance
    simpson_check(test, lines, "test")

    # Plots
    plot_p_by_subgroup(
        test,
        OUT_DIR / "p_by_subgroup_test.png",
        "Test — market_implied_prob distribution per subgroup (winners filled, losers outline)",
    )
    plot_p_by_subgroup(
        train,
        OUT_DIR / "p_by_subgroup_train.png",
        "Train — market_implied_prob distribution per subgroup",
    )
    calibration_plot(
        test,
        OUT_DIR / "calibration_test.png",
        "Test — naive-market calibration (market_implied_prob vs realised bet_correct)",
    )
    calibration_plot(
        train, OUT_DIR / "calibration_train.png", "Train — naive-market calibration"
    )

    lines.append("\n## Plots\n")
    lines.append(
        "- `p_by_subgroup_test.png` — p_hat distribution by (side, outcomeIndex) × winner/loser, test."
    )
    lines.append("- `p_by_subgroup_train.png` — same for train.")
    lines.append(
        "- `calibration_test.png` — does p_hat predict bet_correct monotonically?"
    )
    lines.append("- `calibration_train.png` — same for train.")

    (OUT_DIR / "report.md").write_text("\n".join(lines))
    print(f"[done] {OUT_DIR / 'report.md'}")
    print(f"       plots in {OUT_DIR}/")

    # Stdout summary
    print("\n" + "=" * 80)
    print("NAIVE-MARKET INVESTIGATION — KEY NUMBERS")
    print("=" * 80)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        p = df["market_implied_prob"].to_numpy(dtype=float)
        y = df["bet_correct"].astype(int).to_numpy()
        mask = ~np.isnan(p)
        print(
            f"\n{name}: naive_market ROC = {safe_roc(y[mask], p[mask]):.4f}  "
            f"(n={mask.sum():,}, bc={y[mask].mean():.3f})"
        )
        df2 = df.copy()
        df2["side_bool"] = df2["side"].astype(str).str.upper().eq("BUY").astype(int)
        df2["subgroup"] = df2.apply(
            lambda r: ("BUY" if r["side_bool"] else "SELL")
            + "_idx"
            + str(int(r["outcomeIndex"])),
            axis=1,
        )
        for sg, g in df2.groupby("subgroup"):
            p = g["market_implied_prob"].to_numpy(dtype=float)
            y = g["bet_correct"].astype(int).to_numpy()
            mask = ~np.isnan(p)
            p, y = p[mask], y[mask]
            if len(y) == 0:
                continue
            roc = safe_roc(y, p)
            roc_str = f"{roc:.4f}" if roc is not None else "constant bc; ROC undef"
            print(
                f"  {sg:<12}  n={len(y):>6,}  bc={y.mean():.3f}  p̂={p.mean():.3f}  ROC={roc_str}"
            )


if __name__ == "__main__":
    main()
