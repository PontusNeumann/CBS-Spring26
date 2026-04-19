"""Comprehensive EDA script for the Iran-strike labeled dataset.

Run this on `data/iran_strike_labeled.parquet` to produce all the plots and
diagnostics the team needs to decide on final modelling choices. Saves
figures to `data/eda_outputs/` and prints numeric summaries to stdout.

Covers every item in the EDA test plan from PR #1:
  1. Shape + null check
  2. Class balance per bucket + per market
  3. Feature distributions + skewness (confirms log1p choices)
  4. Correlation heatmap — identify near-duplicates
  5. PCA projection — the behavioural-taxonomy four-quadrant plot
  6. Per-market price trajectory + volume timing
  7. Wallet-level aggregates: who are the whales, the spread-builders, the
     brand-new wallets? (distribution summaries)
  8. Magamyman-window inspection — what's happening in Feb 27-28 trades?

Usage:
  python scripts/eda.py
    [--labeled data/iran_strike_labeled.parquet]
    [--out data/eda_outputs/]

Outputs:
  data/eda_outputs/
    01_class_balance.png
    02_feature_distributions.png
    03_skewness_table.csv
    04_correlation_heatmap.png
    05_pca_wallets.png
    06_price_trajectories.png
    07_feb28_final_days_volume.png
    08_wallet_type_distributions.png
    summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9})

# A restrained palette that reads well in print (for the report)
PALETTE = {
    "train": "#4C72B0",
    "val": "#DD8452",
    "test": "#55A868",
    "correct": "#2E8B57",
    "incorrect": "#C44E52",
    "accent": "#7B2CBF",
}


# -----------------------------------------------------------------------------
# 1. Shape + sanity
# -----------------------------------------------------------------------------
def basic_diagnostics(df: pd.DataFrame, out_txt_path: Path) -> None:
    lines = []
    lines.append(f"dataset shape: {df.shape}")
    lines.append(f"total unique wallets (taker): {df['wallet'].nunique():,}")
    lines.append(f"total unique markets: {df['condition_id'].nunique()}")
    lines.append(f"total nulls: {int(df.isna().sum().sum()):,}")
    lines.append("\nbucket sizes:")
    lines.append(str(df["bucket"].value_counts().to_frame("n")))
    lines.append("\nbet_correct mean per bucket:")
    lines.append(str(df.groupby("bucket")["bet_correct"].mean().round(3)))
    lines.append("\nbet_correct mean per market (resolved + trades):")
    per_market = (
        df.groupby(["question", "resolved"])
        .agg(n_trades=("bet_correct", "size"), correct_rate=("bet_correct", "mean"))
        .round(3)
    )
    lines.append(str(per_market))
    out_txt_path.write_text("\n".join(lines))
    print("\n".join(lines))


# -----------------------------------------------------------------------------
# 2. Class balance
# -----------------------------------------------------------------------------
def plot_class_balance(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Per bucket
    ct = pd.crosstab(df["bucket"], df["bet_correct"], normalize="index") * 100
    ct = ct.reindex(["train", "val", "test"])
    ct.plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
        color=[PALETTE["incorrect"], PALETTE["correct"]],
        edgecolor="white",
        width=0.7,
    )
    axes[0].set_title("bet_correct rate per bucket")
    axes[0].set_ylabel("% of trades")
    axes[0].set_xlabel("")
    axes[0].legend(["incorrect", "correct"], frameon=False)
    axes[0].tick_params(axis="x", rotation=0)

    # Per market
    mkt = df.groupby("question")["bet_correct"].mean().sort_values()
    colors = [
        PALETTE["correct"] if v >= 0.5 else PALETTE["incorrect"] for v in mkt.values
    ]
    axes[1].barh(
        [
            q.replace("US strikes Iran by ", "").replace(", 2026?", "")
            for q in mkt.index
        ],
        mkt.values,
        color=colors,
        edgecolor="white",
    )
    axes[1].axvline(0.5, color="black", linestyle="--", lw=0.8, alpha=0.5)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("bet_correct rate per market")
    axes[1].set_xlabel("mean bet_correct")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


# -----------------------------------------------------------------------------
# 3. Feature distributions + skewness
# -----------------------------------------------------------------------------
def plot_feature_distributions(
    df: pd.DataFrame, out_path: Path, skew_out: Path
) -> None:
    feat_cols = [
        "log_size",
        "log_time_to_settlement",
        "pct_time_elapsed",
        "market_cumvol_log",
        "market_price_std_1h",
        "wallet_polymarket_age_days",
        "wallet_prior_trades_log",
        "wallet_directional_purity_in_market",
        "wallet_spread_ratio",
        "wallet_trades_in_market_last_10min",
        "trade_size_vs_position_pct",
        "size_x_log_time",
    ]
    present = [c for c in feat_cols if c in df.columns]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)
    for ax, col in zip(axes.flatten(), present):
        # separate by bet_correct
        for lbl, cc in [(0, PALETTE["incorrect"]), (1, PALETTE["correct"])]:
            vals = (
                df.loc[df["bet_correct"] == lbl, col]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(vals) > 0:
                # clip for plotting readability
                lo, hi = np.percentile(vals, [1, 99])
                vals = vals.clip(lo, hi)
                ax.hist(
                    vals,
                    bins=40,
                    alpha=0.55,
                    color=cc,
                    label=f"correct={lbl}",
                    density=True,
                )
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Feature distributions split by bet_correct", y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")

    # Skewness table
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skew_vals = df[numeric_cols].skew().sort_values(key=abs, ascending=False)
    skew_vals.to_csv(skew_out, header=["skewness"])
    print(f"saved {skew_out.name} — top 10 most skewed:")
    print(skew_vals.head(10).round(2))


# -----------------------------------------------------------------------------
# 4. Correlation heatmap
# -----------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    feat_cols = [c for c in df.columns if df[c].dtype.kind in "bif"]
    drop_meta = {
        "timestamp",
        "block_number",
        "settlement_ts",
        "polygon_first_tx_ts",
        "first_usdc_inbound_ts",
    }
    feat_cols = [c for c in feat_cols if c not in drop_meta]
    # subsample rows if very big for speed
    sample = df[feat_cols].sample(min(50_000, len(df)), random_state=42)
    corr = sample.corr(numeric_only=True).fillna(0)
    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        ax=ax,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    )
    ax.set_title("Feature correlation heatmap (50k-row sample)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")

    # Report highly-correlated feature pairs
    corr_abs = corr.abs()
    # Mask the diagonal and lower triangle (avoid duplicates)
    upper = corr_abs.where(np.triu(np.ones_like(corr_abs, dtype=bool), k=1))
    pairs = upper.stack().sort_values(ascending=False).head(20).round(3)
    print("\ntop 20 feature-pair correlations (|r| desc):")
    print(pairs.to_string())


# -----------------------------------------------------------------------------
# 5. PCA projection — wallet-level taxonomy
# -----------------------------------------------------------------------------
def plot_pca_wallets(df: pd.DataFrame, out_path: Path) -> None:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Aggregate per wallet: behavioural summary
    agg = (
        df.groupby("wallet")
        .agg(
            trades=("bet_correct", "size"),
            correct_rate=("bet_correct", "mean"),
            log_size_mean=("log_size", "mean"),
            age_days_at_last=("wallet_polymarket_age_days", "max"),
            purity=("wallet_directional_purity_in_market", "mean"),
            burst_rate=("wallet_is_burst", "mean"),
            whale=("wallet_is_whale_in_market", "max"),
            spread_ratio=("wallet_spread_ratio", "mean"),
            n_markets=("wallet_prior_markets", "max"),
        )
        .query("trades >= 5")  # drop noisy single-trade wallets
    )

    features = [
        "log_size_mean",
        "age_days_at_last",
        "purity",
        "burst_rate",
        "spread_ratio",
        "n_markets",
    ]
    X = agg[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    Xs = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2).fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sc = ax.scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=agg["correct_rate"],
        cmap="RdYlGn",
        s=8,
        alpha=0.6,
        vmin=0,
        vmax=1,
        edgecolors="none",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"Wallet-level PCA (n={len(agg):,} wallets with ≥5 trades) — colour = mean correctness"
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.75)
    cb.set_label("wallet mean bet_correct")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


# -----------------------------------------------------------------------------
# 6. Per-market price trajectory
# -----------------------------------------------------------------------------
def plot_price_trajectories(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    order = df.groupby("question")["settlement_ts"].first().sort_values().index.tolist()
    cmap = plt.get_cmap("viridis", len(order))
    for i, q in enumerate(order):
        sub = df[df["question"] == q].sort_values("timestamp")
        # downsample for legibility
        sub = sub.iloc[:: max(1, len(sub) // 2000)]
        ax.plot(
            pd.to_datetime(sub["timestamp"], unit="s", utc=True),
            sub["price"],
            lw=0.5,
            alpha=0.7,
            color=cmap(i),
            label=q.replace("US strikes Iran by ", "").replace(", 2026?", ""),
        )
    ax.set_xlabel("Trade timestamp")
    ax.set_ylabel("Trade price (token1 probability)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-market price trajectory across 7 sub-markets")
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


# -----------------------------------------------------------------------------
# 7. Feb-27/28 volume spike
# -----------------------------------------------------------------------------
def plot_feb28_spike(df: pd.DataFrame, out_path: Path) -> None:
    feb28 = df[df["question"] == "US strikes Iran by February 28, 2026?"].copy()
    feb28["hour"] = pd.to_datetime(feb28["timestamp"], unit="s", utc=True).dt.floor("h")
    hourly = feb28.groupby("hour").agg(
        trades=("usd_amount", "size"),
        volume=("usd_amount", "sum"),
        correct_rate=("bet_correct", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax1.plot(
        hourly.index,
        hourly["volume"],
        color=PALETTE["accent"],
        lw=1.5,
        label="hourly USD volume",
    )
    ax1.set_ylabel("hourly USD volume", color=PALETTE["accent"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["accent"])
    ax2 = ax1.twinx()
    ax2.plot(
        hourly.index,
        hourly["correct_rate"],
        color=PALETTE["train"],
        lw=1,
        alpha=0.6,
        label="hourly correct rate",
    )
    ax2.set_ylabel("hourly correct rate", color=PALETTE["train"])
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor=PALETTE["train"])
    ax1.set_title(
        "Feb 28 market — hourly volume and correctness (Magamyman's window is Feb 27-28)"
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


# -----------------------------------------------------------------------------
# 8. Wallet-type distributions
# -----------------------------------------------------------------------------
def plot_wallet_types(df: pd.DataFrame, out_path: Path) -> None:
    agg = df.groupby("wallet").agg(
        trades=("bet_correct", "size"),
        mean_purity=("wallet_directional_purity_in_market", "mean"),
        burst_rate=("wallet_is_burst", "mean"),
        whale_any=("wallet_is_whale_in_market", "max"),
        new=("wallet_is_new_to_polymarket", "max"),
        correct_rate=("bet_correct", "mean"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    axes[0, 0].hist(
        agg["mean_purity"].dropna(), bins=40, color=PALETTE["accent"], alpha=0.8
    )
    axes[0, 0].set_title("directional purity — wallet mean")
    axes[0, 0].set_xlabel("purity")
    axes[0, 0].set_ylabel("n wallets")

    axes[0, 1].hist(
        agg["burst_rate"].dropna(), bins=40, color=PALETTE["train"], alpha=0.8
    )
    axes[0, 1].set_title("burst rate — share of wallet's trades that are bursty")
    axes[0, 1].set_xlabel("burst rate")
    axes[0, 1].set_ylabel("n wallets")

    axes[1, 0].hist(
        np.log1p(agg["trades"]), bins=40, color=PALETTE["correct"], alpha=0.8
    )
    axes[1, 0].set_title("wallet trade count (log1p) — the Pareto tail")
    axes[1, 0].set_xlabel("log1p(trades)")
    axes[1, 0].set_ylabel("n wallets")

    # correctness by wallet-type quadrant
    q_high_purity = agg["mean_purity"] > 0.7
    q_bursty = agg["burst_rate"] > 0.2
    quadrants = pd.Series(index=agg.index, dtype=object)
    quadrants[q_high_purity & q_bursty] = "pure + bursty\n(informed?)"
    quadrants[q_high_purity & ~q_bursty] = "pure, not bursty\n(retail)"
    quadrants[~q_high_purity & q_bursty] = "not pure, bursty\n(MM/vol)"
    quadrants[~q_high_purity & ~q_bursty] = "not pure, not bursty\n(rebalancer)"
    by_q = agg.groupby(quadrants)["correct_rate"].mean()
    by_q.plot(
        kind="bar",
        ax=axes[1, 1],
        color=[
            PALETTE["accent"],
            PALETTE["train"],
            PALETTE["val"],
            PALETTE["incorrect"],
        ],
    )
    axes[1, 1].axhline(0.5, color="black", linestyle="--", lw=0.8, alpha=0.5)
    axes[1, 1].set_title("mean correct rate per behavioural quadrant")
    axes[1, 1].set_ylabel("mean bet_correct")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")
    print("\nquadrant correctness means:")
    print(by_q.round(3))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labeled", default=str(ROOT / "data" / "iran_strike_labeled.parquet")
    )
    ap.add_argument("--out", default=str(ROOT / "data" / "eda_outputs"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.labeled}...")
    df = pd.read_parquet(args.labeled)
    print(f"loaded {len(df):,} rows, {len(df.columns)} columns")

    # Attach question text from markets.parquet if missing
    if "question" not in df.columns:
        mkts = pd.read_parquet(
            ROOT / "data" / "iran_strike_markets.parquet",
            columns=["condition_id", "question"],
        )
        df = df.merge(mkts, on="condition_id", how="left")
        print(f"joined question text from markets.parquet")

    basic_diagnostics(df, out_dir / "summary.txt")
    plot_class_balance(df, out_dir / "01_class_balance.png")
    plot_feature_distributions(
        df, out_dir / "02_feature_distributions.png", out_dir / "03_skewness_table.csv"
    )
    plot_correlation_heatmap(df, out_dir / "04_correlation_heatmap.png")
    plot_pca_wallets(df, out_dir / "05_pca_wallets.png")
    plot_price_trajectories(df, out_dir / "06_price_trajectories.png")
    plot_feb28_spike(df, out_dir / "07_feb28_final_days_volume.png")
    plot_wallet_types(df, out_dir / "08_wallet_type_distributions.png")

    print(f"\nall EDA outputs in {out_dir}/")


if __name__ == "__main__":
    main()
