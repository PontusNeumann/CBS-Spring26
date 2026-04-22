"""EDA for the Iran-markets enriched dataset.

Reads `data/03_consolidated_dataset.csv` (the consolidated dataset produced by
`02_build_dataset.py`) and writes figures plus a numeric summary to
`outputs/eda/`. All figures follow `report/Design.md` conventions so they drop
straight into the Word document with no post-processing.

EDA stages (aligned to lectures 2, 5, 7):
    01  shape + dtypes + missingness            (L2 preprocessing)
    02  class balance per split + per market    (target)
    03  feature distributions split by target + skewness table
    04  outlier boxplots on top-skew features   (L7)
    05  correlation heatmap + redundant pairs
    06  PCA on wallet behavioural aggregates    (L5)
    07  per-market price trajectory
    08  wallet behavioural quadrants

Usage:
    python scripts/04_eda.py [--csv data/03_consolidated_dataset.csv] [--out outputs/eda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR_DEFAULT = ROOT / "outputs" / "eda"


# ---------------------------------------------------------------------------
# Design.md theme
# ---------------------------------------------------------------------------
sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})

C_MAP = sns.color_palette("rocket_r", as_cmap=True)
PAL_10 = sns.color_palette("rocket_r", 10)
PAL_K = [PAL_10[i] for i in (1, 3, 6, 8, 9)]
INK = PAL_10[8]
COL_DARK = "0.15"
COL_CORRECT = PAL_10[6]
COL_INCORRECT = PAL_10[2]

FIG_W = 6.3
FIG_W_HALF = 3.1
FIG_W_WIDE = 7.8


def clean_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.name}")


# ---------------------------------------------------------------------------
# Feature taxonomies — grouped by the six layers in project_plan.md §4
# ---------------------------------------------------------------------------
TARGET_COL = "bet_correct"
SPLIT_COL = "split"
WALLET_COL = "proxyWallet"
MARKET_COL = "condition_id"
PRICE_COL = "price"
TS_COL = "timestamp"

# Numeric features used in distributions / correlation / outlier panels. All
# strictly present in `03_consolidated_dataset.csv`; anything gated on Polygonscan or
# GDELT is deliberately absent.
NUM_FEATURES = [
    "log_size",
    "time_to_settlement_s",
    "log_time_to_settlement",
    "pct_time_elapsed",
    "trade_value_usd",
    "market_trade_count_so_far",
    "market_volume_so_far_usd",
    "market_price_vol_last_1h",
    "market_vol_1h_log",
    "market_vol_24h_log",
    "wallet_prior_trades",
    "wallet_prior_volume_usd",
    "wallet_prior_win_rate",
    "wallet_first_minus_trade_sec",
    "wallet_trades_in_market_last_1min",
    "wallet_trades_in_market_last_10min",
    "wallet_trades_in_market_last_60min",
    "wallet_directional_purity_in_market",
    "wallet_spread_ratio",
    "wallet_position_size_before_trade",
    "trade_size_vs_position_pct",
    "wallet_prior_trades_in_market",
    "wallet_cumvol_same_side_last_10min",
    "size_vs_wallet_avg",
    "size_x_time_to_settlement",
    "size_vs_market_cumvol_pct",
]

# Distributions panel is a 3x4 grid, so take the first 12 that exist.
DIST_PRIORITY = [
    "log_size",
    "log_time_to_settlement",
    "pct_time_elapsed",
    "market_price_vol_last_1h",
    "market_volume_so_far_usd",
    "wallet_prior_trades",
    "wallet_prior_win_rate",
    "wallet_directional_purity_in_market",
    "wallet_spread_ratio",
    "wallet_trades_in_market_last_10min",
    "trade_size_vs_position_pct",
    "size_x_time_to_settlement",
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_dataset(csv_path: Path) -> pd.DataFrame:
    print(f"loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"loaded {len(df):,} rows x {len(df.columns)} cols")
    if "question" not in df.columns:
        mkts = pd.read_csv(DATA_DIR / "01_markets_meta.csv",
                           usecols=[MARKET_COL, "question"])
        df = df.merge(mkts, on=MARKET_COL, how="left")
        print("joined question text from 01_markets_meta.csv")
    return df


# ---------------------------------------------------------------------------
# 01. Shape, dtypes, missingness
# ---------------------------------------------------------------------------
def panel_missingness(df: pd.DataFrame, out_path: Path) -> pd.Series:
    nulls = df.isna().sum()
    present = nulls[nulls > 0].sort_values(ascending=True)
    if present.empty:
        print("no nulls present; skipping missingness figure")
        return nulls

    fig, ax = plt.subplots(figsize=(FIG_W, max(2.5, 0.22 * len(present))),
                           constrained_layout=True)
    pct = present / len(df) * 100
    ax.barh(pct.index, pct.values, color=INK, edgecolor="white", height=0.7)
    for y, (name, v) in enumerate(pct.items()):
        ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=7, color=COL_DARK)
    ax.set_xlabel("missing (%)")
    ax.set_xlim(0, max(pct.max() * 1.15, 5))
    clean_ax(ax)
    save_fig(fig, out_path)
    return nulls


# ---------------------------------------------------------------------------
# 02. Class balance
# ---------------------------------------------------------------------------
def panel_class_balance(df: pd.DataFrame, out_path: Path) -> None:
    mkt = df.groupby("question")[TARGET_COL].agg(["mean", "size"]).dropna()
    mkt = mkt.sort_values("mean")
    n_mkt = len(mkt)

    # Tall figure so 74-market bars don't collide; left panel centred via
    # a gridspec with spacer rows.
    from matplotlib.gridspec import GridSpec
    height = max(4.0, 0.18 * n_mkt + 1.2)
    fig = plt.figure(figsize=(FIG_W_WIDE, height), constrained_layout=True)
    gs = GridSpec(3, 2, width_ratios=[1, 1.6], height_ratios=[1, 2.5, 1],
                  figure=fig)
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[:, 1])

    ct = (pd.crosstab(df[SPLIT_COL], df[TARGET_COL], normalize="index") * 100)
    ct = ct.reindex(["train", "val", "test"]).fillna(0)
    ct.columns = ["incorrect", "correct"] if set(ct.columns) == {0, 1} else ct.columns
    ax_left.bar(ct.index, ct["correct"], color=COL_CORRECT, label="correct",
                edgecolor="white", width=0.6)
    ax_left.bar(ct.index, ct["incorrect"], bottom=ct["correct"],
                color=COL_INCORRECT, label="incorrect", edgecolor="white", width=0.6)
    ax_left.axhline(50, color=COL_DARK, lw=0.8, ls="--", alpha=0.5)
    ax_left.set_ylabel("share of trades (%)")
    ax_left.set_ylim(0, 100)
    ax_left.legend(frameon=False, loc="lower right", fontsize=7)
    clean_ax(ax_left)

    labels = [q[:55] + ("..." if len(q) > 55 else "") for q in mkt.index]
    colors = [COL_CORRECT if v >= 0.5 else COL_INCORRECT for v in mkt["mean"]]
    ax_right.barh(labels, mkt["mean"], color=colors, edgecolor="white", height=0.7)
    ax_right.axvline(0.5, color=COL_DARK, lw=0.8, ls="--", alpha=0.5)
    ax_right.set_xlim(0, 1)
    ax_right.set_xlabel("mean bet_correct")
    ax_right.tick_params(axis="y", labelsize=6)
    ax_right.margins(y=0.005)
    clean_ax(ax_right)

    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# 03. Feature distributions split by target + skewness table
# ---------------------------------------------------------------------------
def panel_distributions(df: pd.DataFrame, fig_path: Path, skew_path: Path) -> None:
    present = [c for c in DIST_PRIORITY if c in df.columns]
    if len(present) < 12:
        extra = [c for c in NUM_FEATURES if c in df.columns and c not in present]
        present = (present + extra)[:12]

    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W_WIDE, 6.3),
                             constrained_layout=True)
    for ax, col in zip(axes.flatten(), present):
        for lbl, color in [(0, COL_INCORRECT), (1, COL_CORRECT)]:
            vals = pd.to_numeric(
                df.loc[df[TARGET_COL] == lbl, col], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                continue
            lo, hi = np.percentile(vals, [1, 99])
            if lo == hi:
                continue
            vals = vals.clip(lo, hi)
            ax.hist(vals, bins=40, alpha=0.55, color=color,
                    label="correct" if lbl == 1 else "incorrect",
                    density=True)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)
        clean_ax(ax)
    axes[0, 0].legend(frameon=False, fontsize=7)
    save_fig(fig, fig_path)

    numeric_cols = [c for c in NUM_FEATURES if c in df.columns]
    skew = (df[numeric_cols].apply(pd.to_numeric, errors="coerce")
                             .replace([np.inf, -np.inf], np.nan)
                             .skew()
                             .sort_values(key=abs, ascending=False))
    skew.to_csv(skew_path, header=["skewness"])
    print(f"saved {skew_path.name} — top 8 most skewed:")
    print(skew.head(8).round(2).to_string())


# ---------------------------------------------------------------------------
# 04. Outlier boxplots on top-skew features
# ---------------------------------------------------------------------------
def panel_outliers(df: pd.DataFrame, out_path: Path, top_k: int = 8) -> None:
    numeric_cols = [c for c in NUM_FEATURES if c in df.columns]
    skew = (df[numeric_cols].apply(pd.to_numeric, errors="coerce")
                             .replace([np.inf, -np.inf], np.nan)
                             .skew()
                             .sort_values(key=abs, ascending=False))
    cols = skew.head(top_k).index.tolist()

    data, labels = [], []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        lo, hi = np.percentile(s, [1, 99])
        data.append(s.clip(lo, hi).values)
        labels.append(c)

    # One subplot per feature with independent y-axes - features have wildly
    # different scales, so a shared axis squashes most boxes to invisible.
    n = len(data)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W_WIDE, 2.7 * n_rows),
                             constrained_layout=True)
    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    for i, (ax, d, lbl) in enumerate(zip(axes_flat, data, labels)):
        bp = ax.boxplot([d], tick_labels=[""], vert=True, showfliers=False,
                        patch_artist=True, widths=0.55,
                        medianprops=dict(color=COL_DARK, lw=1.0),
                        whiskerprops=dict(color=COL_DARK, lw=0.8),
                        capprops=dict(color=COL_DARK, lw=0.8))
        bp["boxes"][0].set_facecolor(PAL_10[min(1 + i, 9)])
        bp["boxes"][0].set_edgecolor(COL_DARK)
        bp["boxes"][0].set_alpha(0.85)
        ax.set_title(lbl, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        clean_ax(ax)
    for ax in axes_flat[len(data):]:
        ax.set_visible(False)
    fig.supylabel("value (1–99th percentile clipped)", fontsize=9)
    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# 05. Correlation heatmap
# ---------------------------------------------------------------------------
def panel_correlation(df: pd.DataFrame, out_path: Path, txt_out: Path) -> None:
    feat = [c for c in NUM_FEATURES if c in df.columns]
    sample = df[feat].apply(pd.to_numeric, errors="coerce").sample(
        min(100_000, len(df)), random_state=42
    )
    corr = sample.corr().fillna(0)

    # Colorbar height tied to heatmap via make_axes_locatable so the bar
    # spans the same vertical extent as the (square) matrix.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    n = len(feat)
    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 0.45 * n + 1.2))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    sns.heatmap(
        corr, cmap=C_MAP, vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.0,
        annot=False, ax=ax, cbar_ax=cax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.tight_layout()
    save_fig(fig, out_path)

    upper = corr.abs().where(np.triu(np.ones_like(corr, dtype=bool), k=1))
    pairs = (upper.stack().sort_values(ascending=False).head(20).round(3))
    with txt_out.open("w") as f:
        f.write("top 20 |Pearson r| feature pairs\n")
        f.write(pairs.to_string())
    print(f"saved {txt_out.name} — top 20 pairs written")


# ---------------------------------------------------------------------------
# 06. Wallet PCA
# ---------------------------------------------------------------------------
def panel_pca_wallets(df: pd.DataFrame, out_path: Path) -> None:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    agg = (df.groupby(WALLET_COL)
             .agg(trades=(TARGET_COL, "size"),
                  correct_rate=(TARGET_COL, "mean"),
                  log_size_mean=("log_size", "mean"),
                  purity=("wallet_directional_purity_in_market", "mean"),
                  burst_rate=("wallet_is_burst", "mean"),
                  whale=("wallet_is_whale_in_market", "max"),
                  spread_ratio=("wallet_spread_ratio", "mean"),
                  prior_vol=("wallet_prior_volume_usd", "max"))
             .query("trades >= 5"))
    if agg.empty:
        print("pca skipped — no wallets with >=5 trades")
        return

    features = ["log_size_mean", "purity", "burst_rate",
                "spread_ratio", "prior_vol"]
    X = (agg[features].replace([np.inf, -np.inf], np.nan)
                      .fillna(agg[features].median(numeric_only=True))
                      .to_numpy())
    Xs = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2).fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(FIG_W, 4.4), constrained_layout=True)
    sc = ax.scatter(pcs[:, 0], pcs[:, 1],
                    c=agg["correct_rate"], cmap=C_MAP,
                    s=6, alpha=0.6, vmin=0.3, vmax=0.7,
                    edgecolors="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # Clip display range to focus on the dense core; extreme-wallet outliers
    # sit off-plot but add no readable signal.
    ax.set_xlim(left=-15)
    ax.set_ylim(top=25)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=30)
    cb.set_label("wallet mean bet_correct")
    clean_ax(ax)
    save_fig(fig, out_path)
    print(f"  pca on {len(agg):,} wallets with >=5 trades")


# ---------------------------------------------------------------------------
# 07. Per-market price trajectory
# ---------------------------------------------------------------------------
def panel_price_trajectories(df: pd.DataFrame, out_path: Path) -> None:
    ts = pd.to_datetime(df[TS_COL], format="mixed", utc=True, errors="coerce")
    use = df.assign(_ts=ts).dropna(subset=["_ts", PRICE_COL, "question"])
    order = (use.groupby("question")["_ts"].min().sort_values().index.tolist())
    n = len(order)
    colors = sns.color_palette("rocket_r", max(n, 3))

    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 4.2), constrained_layout=True)
    for i, q in enumerate(order):
        sub = use[use["question"] == q].sort_values("_ts")
        if len(sub) > 2000:
            sub = sub.iloc[:: max(1, len(sub) // 2000)]
        short = q[:50] + ("..." if len(q) > 50 else "")
        ax.plot(sub["_ts"], sub[PRICE_COL], lw=0.4, alpha=0.7,
                color=colors[i], label=short)
    ax.set_xlabel("trade time")
    ax.set_ylabel("trade price")
    ax.set_ylim(0, 1)
    if n <= 20:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                  fontsize=6, frameon=False, ncol=1)
    clean_ax(ax)
    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# 08. Event timing — volume and correctness by time-to-settlement bucket
# ---------------------------------------------------------------------------
def panel_event_timing(df: pd.DataFrame, fig_path: Path, txt_path: Path) -> None:
    """Aggregate USD volume and mean bet_correct by time-to-settlement bucket
    across all resolved markets. Empirical justification for the home-run
    gating rule (`time_to_settlement < 6h`) in §5.2.
    """
    tts = pd.to_numeric(df["time_to_settlement_s"], errors="coerce")
    tv = pd.to_numeric(df["trade_value_usd"], errors="coerce")
    bc = pd.to_numeric(df[TARGET_COL], errors="coerce")

    mask = tts.notna() & (tts > 0)  # drop post-resolution close-outs
    sub = pd.DataFrame({
        "tts": tts[mask], "tv": tv[mask].fillna(0.0),
        "bc": bc[mask],
    })

    bins = [0, 3600, 6 * 3600, 24 * 3600, 7 * 86400, 30 * 86400, np.inf]
    labels = ["<1h", "1-6h", "6-24h", "1-7d", "7-30d", ">30d"]
    sub["bucket"] = pd.cut(sub["tts"], bins=bins, labels=labels, right=False)

    agg = sub.groupby("bucket", observed=True).agg(
        trades=("bc", "size"),
        volume_usd=("tv", "sum"),
        mean_correct=("bc", "mean"),
    ).reindex(labels)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_WIDE, 3.6),
                             constrained_layout=True)
    colors = [PAL_10[i] for i in (1, 3, 5, 6, 7, 8)][: len(labels)]
    axes[0].bar(agg.index.astype(str), agg["volume_usd"] / 1e6,
                color=colors, edgecolor="white")
    axes[0].set_ylabel("total volume (million USD)")
    axes[0].set_xlabel("time to settlement")
    clean_ax(axes[0])

    axes[1].bar(agg.index.astype(str), agg["mean_correct"],
                color=colors, edgecolor="white")
    axes[1].axhline(0.5, color=COL_DARK, lw=0.8, ls="--", alpha=0.5)
    axes[1].set_ylim(0.3, 0.7)
    axes[1].set_ylabel("mean bet_correct")
    axes[1].set_xlabel("time to settlement")
    clean_ax(axes[1])
    save_fig(fig, fig_path)

    with txt_path.open("w") as f:
        f.write("volume and mean bet_correct by time-to-settlement bucket\n\n")
        f.write(agg.round(4).to_string())
    print(f"saved {txt_path.name}")


# ---------------------------------------------------------------------------
# 09. Wallet behavioural quadrants
# ---------------------------------------------------------------------------
def panel_wallet_quadrants(df: pd.DataFrame, out_path: Path, txt_out: Path) -> None:
    agg = df.groupby(WALLET_COL).agg(
        trades=(TARGET_COL, "size"),
        purity=("wallet_directional_purity_in_market", "mean"),
        burst_rate=("wallet_is_burst", "mean"),
        correct_rate=(TARGET_COL, "mean"),
    ).query("trades >= 5")
    if agg.empty:
        print("wallet quadrants skipped — no wallets with >=5 trades")
        return

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_WIDE, 5.6),
                             constrained_layout=True)

    axes[0, 0].hist(agg["purity"].dropna(), bins=40, color=PAL_10[3],
                    alpha=0.85, edgecolor="white")
    axes[0, 0].set_xlabel("wallet mean directional purity")
    axes[0, 0].set_ylabel("n wallets")
    clean_ax(axes[0, 0])

    axes[0, 1].hist(agg["burst_rate"].dropna(), bins=40, color=PAL_10[5],
                    alpha=0.85, edgecolor="white")
    axes[0, 1].set_xlabel("wallet burst rate")
    axes[0, 1].set_ylabel("n wallets")
    clean_ax(axes[0, 1])

    axes[1, 0].hist(np.log1p(agg["trades"]), bins=40, color=PAL_10[7],
                    alpha=0.85, edgecolor="white")
    axes[1, 0].set_xlabel("log1p(wallet trades)")
    axes[1, 0].set_ylabel("n wallets")
    clean_ax(axes[1, 0])

    high_p = agg["purity"] > 0.7
    hi_b = agg["burst_rate"] > 0.2
    labels = pd.Series(index=agg.index, dtype=object)
    labels[high_p & hi_b] = "pure + bursty"
    labels[high_p & ~hi_b] = "pure, not bursty"
    labels[~high_p & hi_b] = "mixed + bursty"
    labels[~high_p & ~hi_b] = "mixed, not bursty"
    order = ["pure + bursty", "pure, not bursty",
             "mixed + bursty", "mixed, not bursty"]
    summary = agg.groupby(labels).agg(
        n_wallets=("correct_rate", "size"),
        mean_correct=("correct_rate", "mean"),
    ).reindex(order)

    colors = [PAL_10[i] for i in (2, 4, 6, 8)]
    axes[1, 1].bar(summary.index, summary["mean_correct"], color=colors,
                   edgecolor="white")
    axes[1, 1].axhline(0.5, color=COL_DARK, lw=0.8, ls="--", alpha=0.5)
    axes[1, 1].set_ylim(0.3, 0.7)
    axes[1, 1].set_ylabel("mean bet_correct")
    axes[1, 1].tick_params(axis="x", rotation=18, labelsize=7)
    clean_ax(axes[1, 1])

    save_fig(fig, out_path)

    with txt_out.open("w") as f:
        f.write("wallet behavioural quadrants (wallets with >=5 trades)\n\n")
        f.write(summary.round(3).to_string())
    print(f"saved {txt_out.name}")


# ---------------------------------------------------------------------------
# summary.txt
# ---------------------------------------------------------------------------
def write_summary(df: pd.DataFrame, nulls: pd.Series, out_path: Path) -> None:
    lines = []
    lines.append("=" * 60)
    lines.append(f"shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    lines.append(f"wallets ({WALLET_COL}): {df[WALLET_COL].nunique():,}")
    lines.append(f"markets ({MARKET_COL}): {df[MARKET_COL].nunique()}")
    ts = pd.to_datetime(df[TS_COL], format="mixed", utc=True, errors="coerce")
    lines.append(f"timespan: {ts.min()} -> {ts.max()}")
    lines.append(f"{TARGET_COL} mean: {df[TARGET_COL].mean():.3f}")

    lines.append("")
    lines.append("split sizes and target mean:")
    lines.append(df.groupby(SPLIT_COL)[TARGET_COL]
                  .agg(["size", "mean"]).round(3).to_string())

    lines.append("")
    lines.append("missing cells (columns with >0 nulls):")
    present = nulls[nulls > 0].sort_values(ascending=False)
    if present.empty:
        lines.append("  none")
    else:
        pct = (present / len(df) * 100).round(2)
        lines.append(pd.DataFrame({"n": present, "pct": pct}).to_string())

    lines.append("")
    lines.append("per-market correctness (top 5 + bottom 5):")
    per_market = (df.groupby("question")[TARGET_COL]
                    .agg(["size", "mean"]).round(3)
                    .sort_values("mean"))
    lines.append("bottom 5:")
    lines.append(per_market.head(5).to_string())
    lines.append("top 5:")
    lines.append(per_market.tail(5).to_string())

    out_path.write_text("\n".join(str(x) for x in lines))
    print(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DATA_DIR / "03_consolidated_dataset.csv"))
    ap.add_argument("--out", default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(args.csv))

    nulls = panel_missingness(df, out_dir / "01_missingness.png")
    panel_class_balance(df, out_dir / "02_class_balance.png")
    panel_distributions(df,
                        out_dir / "03_feature_distributions.png",
                        out_dir / "03_skewness_table.csv")
    panel_outliers(df, out_dir / "04_outlier_boxplots.png")
    panel_correlation(df,
                      out_dir / "05_correlation_heatmap.png",
                      out_dir / "05_top_correlations.txt")
    panel_pca_wallets(df, out_dir / "06_pca_wallets.png")
    panel_price_trajectories(df, out_dir / "07_price_trajectories.png")
    panel_event_timing(df,
                       out_dir / "08_event_timing.png",
                       out_dir / "08_event_timing.txt")
    panel_wallet_quadrants(df,
                           out_dir / "09_wallet_quadrants.png",
                           out_dir / "09_wallet_quadrants.txt")
    write_summary(df, nulls, out_dir / "summary.txt")

    print(f"\nall EDA outputs in {out_dir}/")


if __name__ == "__main__":
    main()
