"""EDA for the wallet-joined Alex cohort.

Reads `data/consolidated_modeling_data.parquet` (Alex's 70 engineered
features + 12 Layer-6 wallet features + meta cols, with `split = {train,
test}` already in the frame) and writes the report-style panels to
`outputs/eda/`.

Panels that fit the joined schema:
    01  shape + dtypes + missingness                 (incl. wallet-coverage breakdown)
    02  class balance per split + per market         (target = bet_correct)
    03  feature distributions split by target + skewness table
    04  outlier boxplots on top-skew features
    05  correlation heatmap (lower triangle) + redundant pairs
    06  per-market trade count + base-rate spread
    07  train-vs-test distribution shift on top features

Panels skipped (joined parquet is feature-only, no wallet ID, no price
trajectory series, no deadline_ts column at the level the original EDA
needed):
    - wallet PCA, wallet quadrants, price trajectories, event timing.

Re-run after the wallet-extras enrichment finishes; the wallet-coverage
panel will jump from 59.8% on test to ~99%+.

Usage:
    python scripts/eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "submission" / "data"
OUT_DIR = ROOT / "submission" / "report_assets" / "figures" / "appendix" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match the report Design.md theme (mirror of scripts/04_eda.py)
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
COL_DARK = "0.15"
COL_CORRECT = PAL_10[6]
COL_INCORRECT = PAL_10[2]
COL_TRAIN = PAL_10[3]
COL_TEST = PAL_10[7]

FIG_W = 6.3
FIG_W_WIDE = 7.8

TARGET = "bet_correct"
SPLIT = "split"
MARKET = "market_id"
WALLET_FLAG = "wallet_enriched"
LAYER6_NUMERIC = [
    "wallet_polygon_age_at_t_days",
    "wallet_polygon_nonce_at_t",
    "wallet_log_polygon_nonce_at_t",
    "wallet_n_inbound_at_t",
    "wallet_log_n_inbound_at_t",
    "wallet_n_cex_deposits_at_t",
    "wallet_cex_usdc_cumulative_at_t",
    "wallet_log_cex_usdc_cum",
    "days_from_first_usdc_to_t",
]
NON_FEATURE = {
    TARGET, SPLIT, MARKET, "ts_dt", "timestamp",
    WALLET_FLAG, "wallet_funded_by_cex", "wallet_funded_by_cex_scoped",
}


def clean_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.name}")


def load_joined() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "consolidated_modeling_data.parquet")
    # Wallet ID was dropped during Alex's feature-engineering, but the raw
    # cohort parquets in `data/archive/alex/` still carry it (column `taker`).
    # Lift it over by row order: the consolidated splits preserve the same
    # ordering as the raw archives, with row counts matching exactly.
    cols = ["taker", "usd_amount"]
    train_raw = pd.read_parquet(
        DATA_DIR / "archive" / "alex" / "train.parquet", columns=cols
    )
    test_raw = pd.read_parquet(
        DATA_DIR / "archive" / "alex" / "test.parquet", columns=cols
    )
    n_train = (df[SPLIT] == "train").sum()
    n_test = (df[SPLIT] == "test").sum()
    if len(train_raw) == n_train and len(test_raw) == n_test:
        lifted = pd.concat(
            [train_raw.reset_index(drop=True), test_raw.reset_index(drop=True)],
            ignore_index=True,
        )
        # Sort df so train rows precede test rows, then attach lifted columns.
        df = pd.concat(
            [df[df[SPLIT] == "train"].reset_index(drop=True),
             df[df[SPLIT] == "test"].reset_index(drop=True)],
            ignore_index=True,
        )
        df["taker"] = lifted["taker"].values
        df["usd_amount"] = lifted["usd_amount"].values
        print(f"attached taker + usd_amount from archive: {df['taker'].nunique():,} unique wallets")
    else:
        print(f"WARN: row counts mismatch ({len(train_raw)} vs {n_train}, "
              f"{len(test_raw)} vs {n_test}); taker/usd_amount not attached")
    print(f"loaded joined dataset: {len(df):,} rows × {len(df.columns)} cols")
    return df


def panel_zero_density(df: pd.DataFrame) -> pd.Series:
    """Share of rows where each numeric feature equals exactly zero.

    The dataset has zero NaN: Alex's feature engineering imputes structural
    "no prior history" cases with 0 (or 0.5 for `taker_yes_share_global`)
    at compute time. This panel surfaces which features carry a heavy zero
    mass, so the reader can tell whether a model is effectively seeing
    "no information" rather than a meaningful zero.
    """
    META = {SPLIT, "market_id", "ts_dt", "timestamp"}
    num_cols = [
        c for c in df.select_dtypes("number").columns
        if c != TARGET and c not in META
    ]
    zd_overall = (df[num_cols].abs() < 1e-12).mean().sort_values(ascending=False)
    keep = zd_overall.head(20)

    train_zd = (df.loc[df[SPLIT] == "train", keep.index].abs() < 1e-12).mean()
    test_zd = (df.loc[df[SPLIT] == "test", keep.index].abs() < 1e-12).mean()

    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 0.32 * len(keep) + 1.0))
    width = 0.4
    y = np.arange(len(keep))
    ax.barh(y - width / 2, train_zd.reindex(keep.index).values * 100,
            width, color=COL_TRAIN, label="train")
    ax.barh(y + width / 2, test_zd.reindex(keep.index).values * 100,
            width, color=COL_TEST, label="test")
    ax.set_yticks(y)
    ax.set_yticklabels(keep.index, fontsize=8)
    ax.set_xlabel("% rows where value = 0")
    ax.set_title("Zero-density by feature (top 20). Imputation-as-zero is the dataset's NaN proxy.")
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    clean_ax(ax)
    save_fig(fig, OUT_DIR / "01_zero_density.png")

    table = (
        zd_overall.head(30)
            .to_frame("overall")
            .assign(train=train_zd.reindex(zd_overall.head(30).index),
                    test=test_zd.reindex(zd_overall.head(30).index))
            .round(4)
    )
    csv_path = OUT_DIR / "01_zero_density_table.csv"
    table.to_csv(csv_path, index_label="feature")
    print(f"saved {csv_path.name}")
    return zd_overall


def panel_wallet_coverage(df: pd.DataFrame) -> dict:
    cov = (
        df.groupby(SPLIT)[WALLET_FLAG]
        .agg(["mean", "size"])
        .rename(columns={"mean": "pct_enriched", "size": "n_rows"})
    )
    cov["pct_enriched"] *= 100

    fig, ax = plt.subplots(figsize=(FIG_W, 2.6))
    bars = ax.bar(
        cov.index, cov["pct_enriched"],
        color=[COL_TRAIN if s == "train" else COL_TEST for s in cov.index],
    )
    ax.set_ylabel("% trades with enriched wallet")
    ax.set_ylim(0, 100)
    ax.set_title("Wallet-feature coverage by split", pad=10)
    for bar, pct, n in zip(bars, cov["pct_enriched"], cov["n_rows"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 4,
            f"{pct:.1f}%\n({n:,} trades)",
            ha="center", va="top", fontsize=9, color="white",
        )
    clean_ax(ax)
    save_fig(fig, OUT_DIR / "02_wallet_coverage.png")

    csv_path = OUT_DIR / "02_wallet_coverage_table.csv"
    cov.round(2).to_csv(csv_path, index_label="split")
    print(f"saved {csv_path.name}")
    return cov.to_dict("index")


def panel_class_balance(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_WIDE, 3.0))

    rates = df.groupby(SPLIT)[TARGET].mean()
    counts = df.groupby(SPLIT).size()
    ax = axes[0]
    bars = ax.bar(
        rates.index, rates.values,
        color=[COL_TRAIN if s == "train" else COL_TEST for s in rates.index],
    )
    ax.axhline(0.5, color=COL_DARK, ls="--", lw=0.8)
    ax.set_ylabel("base rate (bet_correct)")
    ax.set_ylim(0, 0.7)
    ax.set_title("Class balance by split")
    for bar, p, n in zip(bars, rates.values, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{p:.3f}\n(n={n:,})",
            ha="center", va="bottom", fontsize=9,
        )
    clean_ax(ax)

    per_market = (
        df.groupby([SPLIT, MARKET])[TARGET].mean().reset_index()
    )
    ax = axes[1]
    sns.boxplot(
        data=per_market, x=SPLIT, y=TARGET, hue=SPLIT,
        ax=ax, palette={"train": COL_TRAIN, "test": COL_TEST},
        legend=False, width=0.45, fliersize=2,
    )
    ax.axhline(0.5, color=COL_DARK, ls="--", lw=0.8)
    ax.set_ylabel("per-market base rate")
    ax.set_title("Per-market base-rate spread")
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, OUT_DIR / "03_class_balance.png")

    summary = pd.DataFrame({
        "n_trades": counts,
        "base_rate": rates.round(4),
    })
    csv_path = OUT_DIR / "03_class_balance_table.csv"
    summary.to_csv(csv_path, index_label="split")
    print(f"saved {csv_path.name}")


def _numeric_features(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in NON_FEATURE
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def panel_distributions_and_skew(df: pd.DataFrame) -> pd.DataFrame:
    feats = _numeric_features(df)
    sample = df[feats + [TARGET]].sample(min(200_000, len(df)), random_state=42)
    # Skewness AND excess kurtosis (Lecture 5 / MA2 convention). A feature
    # can be symmetric but fat-tailed; |excess_kurtosis| > 3 flags fat-tail
    # behaviour the model may need to be robust to.
    skew = sample[feats].skew(numeric_only=True).sort_values(key=np.abs, ascending=False)
    kurt = sample[feats].kurt(numeric_only=True).reindex(skew.index)

    out_skew = OUT_DIR / "04_skewness_table.csv"
    pd.DataFrame({"skew": skew, "excess_kurtosis": kurt}).round(3).to_csv(out_skew)
    print(f"saved {out_skew.name}")

    # Pick the top-12 by absolute skew, but require enough spread to actually
    # render a histogram (drop near-constant features whose 1-99 pct range is
    # degenerate, those produce empty subplots).
    plottable = []
    for feat in skew.index:
        v = sample[feat].dropna()
        if v.empty:
            continue
        p1, p99 = np.percentile(v, [1, 99])
        if p1 == p99:
            continue
        plottable.append(feat)
        if len(plottable) == 12:
            break
    top12 = plottable

    fig, axes = plt.subplots(3, 4, figsize=(FIG_W_WIDE, 6.6),
                             constrained_layout=True)
    for ax, feat in zip(axes.flat, top12):
        for outcome, colour, label in (
            (0, COL_INCORRECT, "incorrect"),
            (1, COL_CORRECT, "correct"),
        ):
            vals = sample.loc[sample[TARGET] == outcome, feat].dropna()
            if vals.empty:
                continue
            lo, hi = np.percentile(vals, [1, 99])
            vals = vals.clip(lo, hi)
            ax.hist(vals, bins=40, alpha=0.45, color=colour, density=True,
                    label=label, edgecolor="none")
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=7)
        clean_ax(ax)
    axes.flat[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Top-12 skewed features: distribution by bet_correct (1 to 99 pct clipped)",
                 y=1.02)
    save_fig(fig, OUT_DIR / "04_distributions.png")
    return skew


def panel_outliers(df: pd.DataFrame, skew: pd.DataFrame, top_k: int = 8) -> None:
    """One boxplot per feature, independent y-axes (features have wildly
    different scales so a shared axis squashes most boxes flat).
    """
    sample = df[skew.index.tolist()].sample(min(150_000, len(df)), random_state=42)
    data, labels = [], []
    for feat in skew.index:
        s = sample[feat].dropna()
        if s.empty:
            continue
        lo, hi = np.percentile(s, [1, 99])
        if lo == hi:
            continue
        data.append(s.clip(lo, hi).values)
        labels.append(feat)
        if len(data) == top_k:
            break

    n = len(data)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W_WIDE, 2.7 * n_rows),
                             constrained_layout=True)
    axes_flat = axes.flatten() if n > 1 else [axes]
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
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Top-{n} skewed features: value spread (boxplots)", fontsize=10)
    fig.supylabel("value (1–99th percentile clipped)", fontsize=9)
    save_fig(fig, OUT_DIR / "05_outlier_boxplots.png")


def panel_correlation(df: pd.DataFrame) -> None:
    feats = _numeric_features(df)
    sample = df[feats].sample(min(150_000, len(df)), random_state=42)
    corr_full = sample.corr(numeric_only=True).fillna(0)

    # Restrict to top-40 features by centrality, defined as the mean
    # absolute Pearson correlation with all OTHER features. This picks
    # the most entangled features, which is what a correlation matrix is
    # designed to surface. Diagonal is excluded so self-correlation does
    # not dominate the ranking.
    if len(corr_full) > 40:
        without_self = corr_full.copy()
        np.fill_diagonal(without_self.values, 0.0)
        centrality = without_self.abs().mean().sort_values(ascending=False)
        keep = centrality.head(40).index.tolist()
        corr = corr_full.loc[keep, keep]
    else:
        corr = corr_full

    # Upper-triangle only (mask the lower, keep the diagonal).
    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    n = len(corr)
    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 0.42 * n + 1.5))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    sns.heatmap(
        corr, mask=mask, cmap=C_MAP, vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.0, annot=False, ax=ax, cbar_ax=cax,
        cbar_kws={"label": "Pearson r"},
    )
    # X-axis labels along the TOP edge, tilted up-and-left, no tick marks.
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="both", length=0, labelsize=9, pad=3)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(-45)
        lbl.set_ha("right")
        lbl.set_rotation_mode("anchor")
        lbl.set_color(COL_DARK)
        lbl.set_fontweight("medium")
    # Y-axis labels staircased along the diagonal (next to the triangle
    # border) instead of pinned to the far left. Hide default y-ticks.
    ax.set_yticks([])
    ax.tick_params(axis="y", which="both", length=0, labelleft=False)
    for i, name in enumerate(corr.index):
        ax.text(i, i + 0.5, name + "  ", ha="right", va="center", fontsize=8,
                color=COL_DARK)
    ax.set_title(
        "Top-40 features by correlation centrality (mean |Pearson r| with the others)",
        fontsize=11, pad=20,
    )
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "06_correlation_heatmap.png")

    upper = corr.abs().where(np.triu(np.ones_like(corr, dtype=bool), k=1))
    pairs = upper.stack().sort_values(ascending=False).head(25).round(3)
    out_path = OUT_DIR / "06_top_correlations.txt"
    with out_path.open("w") as f:
        f.write("top 25 |Pearson r| feature pairs (lower-triangle scan)\n\n")
        f.write(pairs.to_string())
    print(f"saved {out_path.name}")


def panel_market_volume(df: pd.DataFrame) -> None:
    per = df.groupby([SPLIT, MARKET]).agg(
        n=(TARGET, "size"),
        base_rate=(TARGET, "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
    for split_name, colour in (("train", COL_TRAIN), ("test", COL_TEST)):
        sub = per[per[SPLIT] == split_name]
        ax.scatter(
            sub["n"], sub["base_rate"], alpha=0.7, s=24,
            color=colour, label=split_name, edgecolors="white", linewidths=0.5,
        )
    ax.set_xscale("log")
    ax.axhline(0.5, color=COL_DARK, ls="--", lw=0.8)
    ax.set_xlabel("# trades in market (log)")
    ax.set_ylabel("base rate")
    ax.set_title("Per-market size vs base rate")
    ax.legend(frameon=False, loc="upper right")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "07_market_volume.png")

    # Cohort sizing summary as a report-ready CSV table.
    summary = per.groupby(SPLIT)["n"].describe()[["count", "min", "50%", "max"]]
    summary.columns = ["n_markets", "min_trades", "median_trades", "max_trades"]
    summary = summary.round(0).astype(int)
    sizing_path = OUT_DIR / "07_cohort_sizing_table.csv"
    summary.to_csv(sizing_path)
    print(f"saved {sizing_path.name}")

    # Per-market detail as a report-ready CSV table.
    detail = per.sort_values(["split", "n"], ascending=[True, False]).round({"base_rate": 4})
    detail_path = OUT_DIR / "07_per_market_table.csv"
    detail.to_csv(detail_path, index=False)
    print(f"saved {detail_path.name}")


def panel_train_test_shift(df: pd.DataFrame) -> None:
    feats = _numeric_features(df)
    train = df[df[SPLIT] == "train"][feats]
    test = df[df[SPLIT] == "test"][feats]
    if train.empty or test.empty:
        print("skipping shift panel, one split missing")
        return

    # Standardised mean shift per feature (Cohen's d-like).
    pooled_std = pd.concat([train, test]).std(numeric_only=True).replace(0, np.nan)
    shift = ((test.mean(numeric_only=True) - train.mean(numeric_only=True)) / pooled_std).abs()
    shift = shift.dropna().sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 0.34 * len(shift) + 1.0))
    ax.barh(shift.index[::-1], shift.values[::-1], color=PAL_10[6])
    ax.set_xlabel("|standardised mean shift| (test − train)")
    ax.set_title("Top 15 features with largest train→test mean shift")
    clean_ax(ax)
    save_fig(fig, OUT_DIR / "08_train_test_shift.png")

    csv_path = OUT_DIR / "08_train_test_shift_table.csv"
    shift.round(3).to_frame("abs_std_mean_shift").to_csv(csv_path, index_label="feature")
    print(f"saved {csv_path.name}")


# ---------------------------------------------------------------------------
# 09. Late-flow signature, base rate vs time-to-deadline buckets.
# Replicates the Mitts & Ofir 2026 setup that motivated the cohort design.
# ---------------------------------------------------------------------------
def panel_late_flow(df: pd.DataFrame) -> None:
    if "log_time_to_deadline_hours" not in df.columns:
        print("skipping late-flow panel, log_time_to_deadline_hours missing")
        return

    # log_time_to_deadline_hours = log1p(hours) ⇒ recover hours then bucket
    hrs = np.expm1(df["log_time_to_deadline_hours"].clip(lower=0))
    df = df.assign(_hrs_to_deadline=hrs)
    bins = [-0.01, 1 / 12, 1, 6, 24, 24 * 3, 24 * 7, 24 * 14, np.inf]
    labels = ["≤5min", "5min–1h", "1h–6h", "6h–24h", "1d–3d", "3d–7d", "7d–14d", ">14d"]
    df["_bucket"] = pd.cut(df["_hrs_to_deadline"], bins=bins, labels=labels)

    agg = (
        df.groupby([SPLIT, "_bucket"], observed=True)[TARGET]
        .agg(["mean", "size"])
        .rename(columns={"mean": "base_rate", "size": "n"})
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_WIDE, 3.4))
    ax = axes[0]
    width = 0.4
    x = np.arange(len(labels))
    for i, (split_name, colour) in enumerate((("train", COL_TRAIN), ("test", COL_TEST))):
        sub = agg[agg[SPLIT] == split_name].set_index("_bucket").reindex(labels)
        ax.bar(
            x + (i - 0.5) * width, sub["base_rate"].fillna(0), width,
            color=colour, edgecolor="white", label=split_name,
        )
    ax.axhline(0.5, color=COL_DARK, ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("base rate (bet_correct)")
    ax.set_title("Hit rate vs time-to-deadline (Mitts & Ofir-style buckets)")
    ax.legend(frameon=False, loc="upper left")
    clean_ax(ax)

    ax = axes[1]
    for split_name, colour in (("train", COL_TRAIN), ("test", COL_TEST)):
        sub = agg[agg[SPLIT] == split_name].set_index("_bucket").reindex(labels)
        share = sub["n"].fillna(0) / sub["n"].fillna(0).sum()
        ax.bar(
            x + (0 if split_name == "train" else width / 2) - width / 4,
            share * 100, width / 2,
            color=colour, edgecolor="white", label=split_name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% of trades in bucket")
    ax.set_title("Trade-count share by time-to-deadline bucket")
    ax.legend(frameon=False, loc="upper left")
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, OUT_DIR / "09_late_flow.png")

    csv_path = OUT_DIR / "09_late_flow_table.csv"
    agg.pivot(index="_bucket", columns=SPLIT, values=["base_rate", "n"]).round(4).to_csv(csv_path)
    print(f"saved {csv_path.name}")


# ---------------------------------------------------------------------------
# 10. Wallet-stratum base rates, motivates Layer-6 enrichment.
# ---------------------------------------------------------------------------
def panel_wallet_strata(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_WIDE, 3.4))

    # 10a. age decile
    ax = axes[0]
    if "wallet_polygon_age_at_t_days" in df.columns:
        sub = df.dropna(subset=["wallet_polygon_age_at_t_days"])
        deciles = pd.qcut(sub["wallet_polygon_age_at_t_days"], 10, duplicates="drop")
        rate = sub.groupby(deciles, observed=True)[TARGET].agg(["mean", "size"])
        x = np.arange(len(rate))
        ax.bar(x, rate["mean"], color=PAL_10[5], edgecolor="white")
        ax.axhline(0.5, color=COL_DARK, ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(len(rate))], fontsize=8)
        ax.set_xlabel("wallet age decile (1=youngest, 10=oldest)")
        ax.set_ylabel("base rate")
        ax.set_title("Hit rate by wallet polygon-age decile")
        clean_ax(ax)

    # 10b. funded-by-cex split
    ax = axes[1]
    if "wallet_funded_by_cex_scoped" in df.columns:
        rate = df.groupby("wallet_funded_by_cex_scoped")[TARGET].agg(["mean", "size"])
        rate.index = ["not CEX-funded", "CEX-funded (causal)"]
        ax.bar(rate.index, rate["mean"], color=[PAL_10[3], PAL_10[7]], edgecolor="white")
        ax.axhline(0.5, color=COL_DARK, ls="--", lw=0.8)
        ax.set_ylabel("base rate")
        ax.set_title("Hit rate by CEX-funding (scoped, causal)")
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.005, f"{m:.3f}\n(n={n:,})", ha="center", fontsize=8)
        clean_ax(ax)

    # 10c. nonce decile (overall trader experience)
    ax = axes[2]
    if "wallet_polygon_nonce_at_t" in df.columns:
        sub = df.dropna(subset=["wallet_polygon_nonce_at_t"])
        deciles = pd.qcut(sub["wallet_polygon_nonce_at_t"], 10, duplicates="drop")
        rate = sub.groupby(deciles, observed=True)[TARGET].agg(["mean", "size"])
        x = np.arange(len(rate))
        ax.bar(x, rate["mean"], color=PAL_10[5], edgecolor="white")
        ax.axhline(0.5, color=COL_DARK, ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(len(rate))], fontsize=8)
        ax.set_xlabel("polygon-nonce decile (1=fewest tx, 10=most)")
        ax.set_ylabel("base rate")
        ax.set_title("Hit rate by trader-experience decile")
        clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, OUT_DIR / "10_wallet_strata.png")


# ---------------------------------------------------------------------------
# 11. Per-market base rate, the single-event-resolution bimodality.
# ---------------------------------------------------------------------------
def panel_per_market_bimodality(df: pd.DataFrame) -> None:
    per = df.groupby([SPLIT, MARKET]).agg(
        n=(TARGET, "size"),
        base_rate=(TARGET, "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_WIDE, 3.0))
    for ax, split_name, colour in (
        (axes[0], "train", COL_TRAIN),
        (axes[1], "test", COL_TEST),
    ):
        sub = per[per[SPLIT] == split_name]
        ax.hist(sub["base_rate"], bins=20, color=colour, edgecolor="white")
        ax.axvline(0.5, color=COL_DARK, ls="--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("per-market base rate")
        ax.set_ylabel("# markets")
        ax.set_title(f"{split_name}: {len(sub)} markets")
        clean_ax(ax)

    fig.suptitle("Per-market bet_correct base rate (single-event-resolution bimodality)", y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "11_per_market_bimodality.png")


# ---------------------------------------------------------------------------
# 12. Single-feature ROC per market, feature-stability heatmap.
# ---------------------------------------------------------------------------
def panel_feature_stability(df: pd.DataFrame, top_k: int = 8) -> None:
    from sklearn.metrics import roc_auc_score

    feats = _numeric_features(df)
    # Pick top-k features by absolute Pearson with target on a 200k sample.
    sample = df.sample(min(200_000, len(df)), random_state=42)
    rho = sample[feats].apply(
        lambda s: pd.to_numeric(s, errors="coerce").corr(sample[TARGET])
    )
    top = rho.abs().sort_values(ascending=False).head(top_k).index.tolist()

    rows = []
    for mkt, g in df.groupby(MARKET):
        if g[TARGET].nunique() < 2 or len(g) < 200:
            continue
        for feat in top:
            x = g[feat]
            y = g[TARGET]
            mask = x.notna() & y.notna()
            if mask.sum() < 200 or y[mask].nunique() < 2:
                continue
            try:
                auc = roc_auc_score(y[mask], x[mask])
            except ValueError:
                continue
            rows.append({"market": mkt, "feature": feat, "auc": float(auc)})

    if not rows:
        print("skipping feature-stability panel, no markets met thresholds")
        return

    pm = pd.DataFrame(rows).pivot(index="feature", columns="market", values="auc")
    pm = pm.reindex(top)

    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 0.45 * len(top) + 1.5))
    sns.heatmap(
        pm, cmap="RdBu_r", center=0.5, vmin=0.2, vmax=0.8,
        ax=ax, cbar_kws={"label": "single-feature ROC-AUC"},
        xticklabels=False, linewidths=0,
    )
    ax.set_xlabel(f"{pm.shape[1]} markets (sorted by ID)")
    ax.set_ylabel("")
    ax.set_title(f"Single-feature ROC-AUC per market (top {top_k} by |Pearson|)")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "12_feature_stability.png")

    out_path = OUT_DIR / "12_feature_stability.txt"
    with out_path.open("w") as f:
        f.write("Per-feature, per-market single-feature ROC-AUC summary\n\n")
        summary = pd.DataFrame({
            "median": pm.median(axis=1),
            "p5": pm.quantile(0.05, axis=1),
            "p95": pm.quantile(0.95, axis=1),
            "n_markets_below_0_5": (pm < 0.5).sum(axis=1),
            "n_markets_above_0_55": (pm > 0.55).sum(axis=1),
        }).round(3)
        f.write(summary.to_string())
    print(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# 13. Mutual-information feature ranking, non-linear signal counterpart
# to the Pearson skew table.
# ---------------------------------------------------------------------------
def panel_mutual_information(df: pd.DataFrame) -> None:
    from sklearn.feature_selection import mutual_info_classif

    feats = _numeric_features(df)
    sample = df.sample(min(150_000, len(df)), random_state=42)
    X = sample[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    y = sample[TARGET].astype(int).to_numpy()

    print(f"  computing MI on {len(sample):,} rows × {len(feats)} features (~30s)...")
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    mi_df = pd.Series(mi, index=feats).sort_values(ascending=False)

    top20 = mi_df.head(20)
    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 0.34 * len(top20) + 1.0))
    ax.barh(top20.index[::-1], top20.values[::-1], color=PAL_10[6])
    ax.set_xlabel("mutual information with bet_correct (nats)")
    ax.set_title("Top-20 features by mutual information with target")
    clean_ax(ax)
    save_fig(fig, OUT_DIR / "13_mutual_information.png")

    out_path = OUT_DIR / "13_mutual_information.csv"
    mi_df.round(5).to_frame("mi").to_csv(out_path)
    print(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# 14. Feature-group taxonomy, how the 79 features split across layers.
# ---------------------------------------------------------------------------
FEATURE_GROUPS: list[tuple[str, list[str]]] = [
    ("Trade-level", [
        "log_size", "side_buy", "outcome_yes", "trade_size_to_recent_volume_ratio",
        "trade_size_vs_recent_avg", "avg_trade_size_recent_1h", "log_size_vs_taker_avg",
        "log_same_block_trade_count",
    ]),
    ("Time-to-deadline", [
        "log_time_to_deadline_hours", "pct_time_elapsed", "log_time_since_last_trade",
        "is_within_24h_of_deadline", "is_within_1h_of_deadline", "is_within_5min_of_deadline",
        "hour_of_day_sin", "day_of_week_sin", "day_of_week_cos",
    ]),
    ("Market-state rolling", [
        "log_n_trades_to_date", "market_buy_share_running", "log_recent_volume_5min",
        "log_recent_volume_1h", "log_recent_volume_24h", "log_trade_count_5min",
        "log_trade_count_1h", "log_trade_count_24h", "market_price_vol_last_5min",
        "market_price_vol_last_1h", "market_price_vol_last_24h",
        "order_flow_imbalance_5min", "order_flow_imbalance_1h", "order_flow_imbalance_24h",
        "yes_volume_share_recent_5min", "yes_volume_share_recent_1h",
        "yes_buy_pressure_5min", "token_side_skew_5min",
    ]),
    ("Price / vol microstructure", [
        "pre_trade_price", "recent_price_mean_5min", "recent_price_mean_1h",
        "recent_price_mean_24h", "recent_price_high_1h", "recent_price_low_1h",
        "recent_price_range_1h", "pre_trade_price_change_5min", "pre_trade_price_change_1h",
        "pre_trade_price_change_24h", "implied_variance", "kyle_lambda_market_static",
        "realized_vol_1h", "jump_component_1h", "signed_oi_autocorr_1h",
        "distance_from_boundary",
    ]),
    ("Edge / payoff", [
        "consensus_strength", "contrarian_score", "is_long_shot_buy",
        "contrarian_strength", "log_payoff_if_correct", "risk_reward_ratio_pre",
    ]),
    ("Taker (HF within-cohort)", [
        "log_taker_prior_trades_in_market", "taker_first_trade_in_market",
        "log_taker_cumvol_in_market", "taker_position_size_before_trade",
        "log_taker_prior_trades_total", "log_taker_prior_volume_total_usd",
        "log_taker_unique_markets_traded", "taker_yes_share_global",
        "taker_directional_purity_in_market", "taker_traded_in_event_id_before",
        "log_taker_burst_5min", "log_taker_first_minutes_ago_in_market",
        "log_maker_prior_trades_in_market",
    ]),
    ("Wallet on-chain (Layer 6)", LAYER6_NUMERIC + ["wallet_funded_by_cex", "wallet_funded_by_cex_scoped"]),
]


def panel_feature_taxonomy(df: pd.DataFrame) -> None:
    feats = set(_numeric_features(df))
    rows = []
    accounted: set[str] = set()
    for group, members in FEATURE_GROUPS:
        present = [f for f in members if f in feats]
        rows.append({"group": group, "n": len(present)})
        accounted |= set(present)
    rest = sorted(feats - accounted)
    if rest:
        rows.append({"group": "Other / unclassified", "n": len(rest)})

    tax = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 0.5 * len(tax) + 1.2))
    bars = ax.barh(tax["group"][::-1], tax["n"][::-1], color=PAL_10[6])
    for bar, n in zip(bars, tax["n"][::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(n), va="center", fontsize=9)
    ax.set_xlabel("# numeric features")
    ax.set_title(f"Feature-group taxonomy (total numeric features: {len(feats)})")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "14_feature_taxonomy.png")

    rows = []
    for group, members in FEATURE_GROUPS:
        present = [m for m in members if m in feats]
        for m in present:
            rows.append({"group": group, "feature": m})
    for m in rest:
        rows.append({"group": "Other / unclassified", "feature": m})
    csv_path = OUT_DIR / "14_feature_taxonomy_table.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"saved {csv_path.name}")

    counts_path = OUT_DIR / "14_feature_taxonomy_counts.csv"
    counts = pd.DataFrame(rows).groupby("group").size().rename("n_features").to_frame()
    counts.to_csv(counts_path, index_label="group")
    print(f"saved {counts_path.name}")


# ---------------------------------------------------------------------------
# 15. Distribution-tail diagnostics, fat-tail screen.
# Pairs with skew + excess kurtosis from panel 04. For each top-10 fat-tail
# feature, show the 1/5/95/99 percentiles + tail-conditional mean (mean
# beyond p95 / below p5). Cohen's-d style robustness diagnostic from MA2.
# ---------------------------------------------------------------------------
def panel_tail_diagnostics(df: pd.DataFrame) -> None:
    feats = _numeric_features(df)
    sample = df[feats].sample(min(200_000, len(df)), random_state=42)
    kurt = sample.kurt(numeric_only=True).abs().sort_values(ascending=False)
    top = kurt.head(15).index.tolist()

    rows = []
    for f in top:
        s = pd.to_numeric(sample[f], errors="coerce").dropna()
        if s.empty:
            continue
        p1, p5, p95, p99 = s.quantile([0.01, 0.05, 0.95, 0.99]).values
        upper_tail = s[s >= p95].mean()
        lower_tail = s[s <= p5].mean()
        rows.append({
            "feature": f,
            "excess_kurtosis": float(s.kurt()),
            "skew": float(s.skew()),
            "p1": p1, "p5": p5, "p95": p95, "p99": p99,
            "tail_mean_below_p5": lower_tail,
            "tail_mean_above_p95": upper_tail,
        })
    tail_df = pd.DataFrame(rows).set_index("feature")

    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 0.34 * len(tail_df) + 1.0))
    y = np.arange(len(tail_df))
    ax.barh(y, tail_df["excess_kurtosis"], color=PAL_10[6])
    ax.set_yticks(y)
    ax.set_yticklabels(tail_df.index, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(3, color=COL_DARK, ls="--", lw=0.8, label="|kurt|=3 (fat-tail threshold)")
    ax.set_xlabel("excess kurtosis")
    ax.set_title("Top-15 fat-tailed features (|excess kurtosis|)")
    ax.legend(frameon=False, loc="lower right")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "15_tail_diagnostics.png")

    out_path = OUT_DIR / "15_tail_diagnostics.csv"
    tail_df.round(3).to_csv(out_path)
    print(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# 16. Temporal base-rate drift, rolling base rate vs trade time.
# MA2-style rolling-window drift detection. Tests whether bet_correct base
# rate is stable across the strike-countdown / ceasefire-countdown windows.
# ---------------------------------------------------------------------------
def panel_temporal_drift(df: pd.DataFrame) -> None:
    if "timestamp" not in df.columns:
        print("skipping temporal-drift panel, timestamp missing")
        return

    times = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df2 = df.assign(_t=times).copy()

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_WIDE, 3.6), sharey=True)
    for ax, split_name, colour in (
        (axes[0], "train", COL_TRAIN),
        (axes[1], "test", COL_TEST),
    ):
        sub = df2[df2[SPLIT] == split_name].sort_values("_t")
        if sub.empty:
            continue
        # Daily bin so the curve isn't too noisy.
        daily = (
            sub.set_index("_t")[TARGET]
            .resample("1D")
            .agg(["mean", "size"])
            .rename(columns={"mean": "rate", "size": "n"})
        )
        # Suppress days with too few trades to avoid spikes.
        daily = daily[daily["n"] >= 50]
        if daily.empty:
            continue
        # 7-day rolling mean for trend.
        rolling = daily["rate"].rolling(7, min_periods=2).mean()
        ax.plot(daily.index, daily["rate"], lw=0.5, color=colour, alpha=0.4, label="daily")
        ax.plot(rolling.index, rolling.values, lw=1.6, color=colour, label="7-day rolling")
        ax.axhline(0.5, color=COL_DARK, ls="--", lw=0.8)
        ax.set_ylim(0.30, 0.70)
        ax.set_ylabel("base rate")
        ax.set_title(f"{split_name}: {len(sub):,} trades, {sub['_t'].min().date()} → {sub['_t'].max().date()}")
        ax.legend(frameon=False, loc="lower left", fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        clean_ax(ax)

    fig.suptitle("Daily bet_correct base rate (≥50 trades/day) with 7-day rolling mean", y=1.03)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "16_temporal_drift.png")


# ---------------------------------------------------------------------------
# 17. Wallet behavioural archetypes via PCA.
# Per-wallet aggregates of the trade-level features, then 2-D PCA, scatter
# coloured by per-wallet hit rate. Brings back the panel from the legacy EDA
# (originally panel 06) using the current feature names.
# ---------------------------------------------------------------------------
def panel_pca_wallets(df: pd.DataFrame) -> None:
    if "taker" not in df.columns:
        print("skipping pca-wallets panel, taker column not attached")
        return
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    agg = (
        df.groupby("taker")
        .agg(
            trades=(TARGET, "size"),
            correct_rate=(TARGET, "mean"),
            log_size_mean=("log_size", "mean"),
            purity_mean=("taker_directional_purity_in_market", "mean"),
            yes_share_mean=("taker_yes_share_global", "mean"),
            markets_seen_max=("log_taker_unique_markets_traded", "max"),
            prior_vol_max=("log_taker_prior_volume_total_usd", "max"),
            polygon_age_mean=("wallet_polygon_age_at_t_days", "mean"),
        )
        .query("trades >= 5")
    )
    if agg.empty:
        print("pca skipped, no wallets with >=5 trades")
        return

    features = [
        "log_size_mean", "purity_mean", "yes_share_mean",
        "markets_seen_max", "prior_vol_max", "polygon_age_mean",
    ]
    X = (
        agg[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(agg[features].median(numeric_only=True))
        .to_numpy()
    )
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit(Xs)
    pcs = pca.transform(Xs)
    var_pct = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(FIG_W, 4.4), constrained_layout=True)
    sc = ax.scatter(
        pcs[:, 0], pcs[:, 1],
        c=agg["correct_rate"], cmap=C_MAP,
        s=6, alpha=0.6, vmin=0.3, vmax=0.7,
        edgecolors="none",
    )
    ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% var)")
    ax.set_title(f"Wallet behavioural archetypes via PCA ({len(agg):,} wallets, ≥5 trades)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=30)
    cb.set_label("wallet mean bet_correct")
    clean_ax(ax)
    save_fig(fig, OUT_DIR / "17_pca_wallets.png")

    out_path = OUT_DIR / "17_pca_wallets.txt"
    with out_path.open("w") as f:
        f.write(f"PCA on {len(agg):,} wallets with >=5 trades\n\n")
        f.write(f"Features used: {features}\n")
        f.write(f"PC1 explained variance: {var_pct[0]:.2f}%\n")
        f.write(f"PC2 explained variance: {var_pct[1]:.2f}%\n")
        f.write(f"Cumulative: {var_pct.sum():.2f}%\n\n")
        loadings = pd.DataFrame(
            pca.components_.T,
            index=features,
            columns=["PC1", "PC2"],
        ).round(3)
        f.write("Loadings:\n")
        f.write(loadings.to_string())
    print(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# 18. Per-market price trajectories.
# `pre_trade_price` is the canonical market-implied probability per trade.
# Plot one thin line per market, coloured by split, so the cohort design is
# visible alongside price evolution.
# ---------------------------------------------------------------------------
def panel_price_trajectories(df: pd.DataFrame) -> None:
    if "pre_trade_price" not in df.columns or "timestamp" not in df.columns:
        print("skipping price-trajectories panel, pre_trade_price or timestamp missing")
        return
    times = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df2 = df.assign(_t=times)[["_t", MARKET, SPLIT, "pre_trade_price"]].dropna()

    market_order = (
        df2.groupby(MARKET)["_t"].min().sort_values().index.tolist()
    )
    palette = sns.color_palette("husl", len(market_order))
    market_colour = dict(zip(market_order, palette))

    train_max = df2.loc[df2[SPLIT] == "train", "_t"].max()

    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 3.6), constrained_layout=True)
    for mid, sub in df2.groupby(MARKET):
        sub = sub.sort_values("_t")
        if len(sub) > 1500:
            sub = sub.iloc[:: max(1, len(sub) // 1500)]
        ax.plot(
            sub["_t"], sub["pre_trade_price"],
            lw=0.5, alpha=0.55, color=market_colour[mid],
        )
    if pd.notna(train_max):
        ax.axvline(train_max, color=COL_DARK, ls="--", lw=0.8, alpha=0.7)
        ax.text(
            train_max, 1.01, "  train → test",
            ha="left", va="bottom", fontsize=8, color=COL_DARK,
        )
    ax.set_xlabel("trade time (UTC)")
    ax.set_ylabel("pre-trade price (market-implied probability)")
    ax.set_title(
        f"Per-market price trajectories, one colour per market "
        f"(n={len(market_order)})"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="x", rotation=20)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    clean_ax(ax)
    save_fig(fig, OUT_DIR / "18_price_trajectories.png")


# ---------------------------------------------------------------------------
# 19. Event timing.
# Daily trade volume with the strike and ceasefire announcement marked. Shows
# the cohort design in calendar time and the relative trading intensity
# around each event.
# ---------------------------------------------------------------------------
def panel_event_timing(df: pd.DataFrame) -> None:
    if "timestamp" not in df.columns:
        print("skipping event-timing panel, timestamp missing")
        return
    times = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df2 = df.assign(_t=times)
    daily = (
        df2.groupby([SPLIT, df2["_t"].dt.floor("D")])
        .size()
        .rename("n_trades")
        .reset_index()
    )

    STRIKE = pd.Timestamp("2026-02-28T06:35:00", tz="UTC")
    CEASEFIRE = pd.Timestamp("2026-04-07T23:59:00", tz="UTC")

    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 3.4), constrained_layout=True)
    for split_name, colour in (("train", COL_TRAIN), ("test", COL_TEST)):
        sub = daily[daily[SPLIT] == split_name].sort_values("_t")
        if sub.empty:
            continue
        ax.bar(sub["_t"], sub["n_trades"], width=0.9, color=colour,
               label=split_name, alpha=0.85, edgecolor="none")
    ax.axvline(STRIKE, color=COL_DARK, ls="--", lw=0.9)
    ax.text(STRIKE, ax.get_ylim()[1] * 0.95, "  strike (28 Feb)",
            ha="left", va="top", fontsize=8, color=COL_DARK)
    ax.axvline(CEASEFIRE, color=COL_DARK, ls="--", lw=0.9)
    ax.text(CEASEFIRE, ax.get_ylim()[1] * 0.95, "  ceasefire announcement (7 Apr)",
            ha="left", va="top", fontsize=8, color=COL_DARK)
    ax.set_xlabel("trade date (UTC)")
    ax.set_ylabel("trades per day")
    ax.set_title("Trade volume over calendar time, with key events")
    ax.legend(frameon=False, loc="upper right")
    ax.tick_params(axis="x", rotation=20)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    clean_ax(ax)
    save_fig(fig, OUT_DIR / "19_event_timing.png")

    csv_path = OUT_DIR / "19_event_timing_table.csv"
    daily.rename(columns={"_t": "date"}).to_csv(csv_path, index=False)
    print(f"saved {csv_path.name}")


# ---------------------------------------------------------------------------
# 19b. Event-zoom panels: ±48h around strike and ceasefire, hourly trade
# counts. Surfaces the intraday reaction that the daily bars in panel 19
# smooth away.
# ---------------------------------------------------------------------------
STRIKE = pd.Timestamp("2026-02-28T06:35:00", tz="UTC")
CEASEFIRE = pd.Timestamp("2026-04-07T23:59:00", tz="UTC")


def panel_event_zoom(df: pd.DataFrame) -> None:
    if "timestamp" not in df.columns:
        print("skipping event-zoom panel, timestamp missing")
        return
    times = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df2 = df.assign(_t=times)

    # Only draw events that fall inside the data range. Ceasefire (7 Apr)
    # is past the dataset's end (31 Mar) so it produces an empty axis.
    t_min, t_max = df2["_t"].min(), df2["_t"].max()
    events = [
        ("strike (28 Feb 06:35 UTC)", STRIKE),
        ("ceasefire (7 Apr 23:59 UTC)", CEASEFIRE),
    ]
    events = [(lbl, t) for lbl, t in events if t_min <= t <= t_max]
    if not events:
        print("skipping event-zoom panel, no events inside data range")
        return

    fig, axes = plt.subplots(
        1, len(events), figsize=(FIG_W_WIDE / 2 * len(events) + 0.2, 3.4),
        constrained_layout=True, squeeze=False,
    )
    for ax, (label, event_t) in zip(axes[0], events):
        window_start = event_t - pd.Timedelta(hours=48)
        window_end = event_t + pd.Timedelta(hours=48)
        sub = df2[(df2["_t"] >= window_start) & (df2["_t"] <= window_end)]
        hourly = (
            sub.set_index("_t")
            .assign(_n=1)["_n"]
            .resample("1h")
            .sum()
        )
        ax.bar(hourly.index, hourly.values, width=1 / 24,
               color=COL_TRAIN, alpha=0.85, edgecolor="none")
        ax.axvline(event_t, color=COL_DARK, ls="--", lw=0.9)
        ax.set_xlim(window_start, window_end)
        ax.set_xlabel("trade time (UTC)")
        ax.set_ylabel("trades per hour")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        clean_ax(ax)

    fig.suptitle("Hourly trade counts in a ±48h window around each event", y=1.04)
    save_fig(fig, OUT_DIR / "19b_event_zoom.png")


# ---------------------------------------------------------------------------
# 19c. Calendar-time $-volume bars. Same axis as panel 19, but `usd_amount`
# summed instead of trade counts: shows whether reaction is real $-flow or
# just a count of small trades.
# ---------------------------------------------------------------------------
def panel_event_volume(df: pd.DataFrame) -> None:
    if "timestamp" not in df.columns or "usd_amount" not in df.columns:
        print("skipping event-volume panel, timestamp or usd_amount missing")
        return
    times = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df2 = df.assign(_t=times)
    daily = (
        df2.set_index("_t")["usd_amount"]
        .resample("1D")
        .sum()
        .rename("usd_total")
        .reset_index()
    )

    t_min, t_max = df2["_t"].min(), df2["_t"].max()
    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 3.4), constrained_layout=True)
    ax.bar(daily["_t"], daily["usd_total"] / 1e6,
           width=0.9, color=COL_TRAIN, alpha=0.85, edgecolor="none")
    if t_min <= STRIKE <= t_max:
        ax.axvline(STRIKE, color=COL_DARK, ls="--", lw=0.9)
        ax.text(STRIKE, ax.get_ylim()[1] * 0.95, "  strike (28 Feb)",
                ha="left", va="top", fontsize=8, color=COL_DARK)
    if t_min <= CEASEFIRE <= t_max:
        ax.axvline(CEASEFIRE, color=COL_DARK, ls="--", lw=0.9)
        ax.text(CEASEFIRE, ax.get_ylim()[1] * 0.95, "  ceasefire (7 Apr)",
                ha="right", va="top", fontsize=8, color=COL_DARK)
    ax.set_xlabel("trade date (UTC)")
    ax.set_ylabel(r"daily \$-volume (\$M)")
    ax.set_title(r"Daily \$-volume over calendar time, with key events")
    ax.tick_params(axis="x", rotation=20)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    clean_ax(ax)
    save_fig(fig, OUT_DIR / "19c_event_volume.png")


def write_index(out_dir: Path) -> None:
    """Cheat-sheet that maps each generated panel to what it shows."""
    items = [
        ("01_zero_density.png", "Top-20 features by share of rows where value = 0 (imputation-as-zero proxy for missingness; dataset has no NaN)."),
        ("01_zero_density_table.csv", "Per-feature overall/train/test zero-density, top 30. Report-ready CSV."),
        ("02_wallet_coverage.png", "% of trades with enriched wallet, by split."),
        ("02_wallet_coverage_table.csv", "Wallet-feature coverage by split (pct_enriched, n_rows). Report-ready CSV."),
        ("03_class_balance.png", "Base rate per split + per-market base-rate spread."),
        ("03_class_balance_table.csv", "Trade count and base rate per split. Report-ready CSV."),
        ("04_distributions.png", "Top-12 skewed features, histogram by bet_correct (1-99 pct clipped)."),
        ("04_skewness_table.csv", "All numeric features ranked by absolute skew. Report-ready CSV."),
        ("05_outlier_boxplots.png", "Box plots for the 8 most-skewed features."),
        ("06_correlation_heatmap.png", "Upper-triangle Pearson heatmap, top-40 features by centrality."),
        ("06_top_correlations.txt", "Top 25 |Pearson r| feature pairs."),
        ("07_market_volume.png", "Per-market trade count vs base rate."),
        ("07_cohort_sizing_table.csv", "Cohort sizing summary (n_markets, min, median, max trades) per split. Report-ready CSV."),
        ("07_per_market_table.csv", "Per-market detail (split, market_id, n trades, base rate). Report-ready CSV."),
        ("08_train_test_shift.png", "Top-15 features by |standardised mean shift| between train and test."),
        ("08_train_test_shift_table.csv", "Per-feature |std mean shift| (Cohen's d). Report-ready CSV."),
        ("09_late_flow.png", "Hit rate vs time-to-deadline (Mitts & Ofir-style buckets)."),
        ("09_late_flow_table.csv", "Per-bucket base rate and trade count by split. Report-ready CSV."),
        ("10_wallet_strata.png", "Hit rate by wallet age decile, CEX-funding, polygon nonce decile."),
        ("11_per_market_bimodality.png", "Per-market base-rate histogram (single-event resolution)."),
        ("12_feature_stability.png", "Single-feature ROC-AUC heatmap per market for top-8 features."),
        ("13_mutual_information.png", "Top-20 features by mutual information with bet_correct."),
        ("13_mutual_information.csv", "Per-feature MI to bet_correct. Report-ready CSV."),
        ("14_feature_taxonomy.png", "How the numeric features split across feature-engineering layers."),
        ("14_feature_taxonomy_table.csv", "Group + feature membership for each modelling feature. Report-ready CSV."),
        ("14_feature_taxonomy_counts.csv", "Feature count per group. Report-ready CSV."),
        ("15_tail_diagnostics.png", "Top-15 fat-tailed features by |excess kurtosis|."),
        ("15_tail_diagnostics.csv", "p1/p5/p95/p99 + tail-conditional means. Report-ready CSV."),
        ("16_temporal_drift.png", "Daily bet_correct base rate with 7-day rolling mean, per split."),
        ("17_pca_wallets.png", "Wallet behavioural archetypes via 2-D PCA on per-wallet aggregates, coloured by hit rate."),
        ("17_pca_wallets.txt", "PC1/PC2 explained variance + loadings for the panel-17 PCA."),
        ("18_price_trajectories.png", "Per-market `pre_trade_price` over calendar time, one colour per market, with a dashed line at the train→test boundary."),
        ("19_event_timing.png", "Trades per day with the strike and ceasefire announcement marked. Cohort design visible in calendar time."),
        ("19_event_timing_table.csv", "Trades per day per split. Report-ready CSV."),
        ("19b_event_zoom.png", "Hourly trade counts in a ±48h window around the strike and ceasefire announcements."),
        ("19c_event_volume.png", "Daily $-volume (`usd_amount`) over calendar time, with the strike and ceasefire marked."),
        ("summary.txt", "Plain-text summary of dataset shape, base rates, missingness."),
    ]
    lines = ["# EDA index, wallet-joined Alex cohort", ""]
    for name, blurb in items:
        path = out_dir / name
        present = "✓" if path.exists() else "x"
        lines.append(f"- [{present}] **{name}**: {blurb}")
    (out_dir / "index.md").write_text("\n".join(lines) + "\n")
    print(f"saved index.md")


def write_summary(df: pd.DataFrame, nulls: pd.Series, cov: dict) -> None:
    feats = _numeric_features(df)
    lines = [
        "EDA summary, wallet-joined Alex cohort",
        f"  rows: {len(df):,}   cols: {len(df.columns):,}   numeric features: {len(feats):,}",
        "",
        "split sizes:",
        df.groupby(SPLIT).size().to_string(),
        "",
        "base rate by split:",
        df.groupby(SPLIT)[TARGET].mean().round(4).to_string(),
        "",
        "wallet-feature coverage:",
    ]
    for split_name, info in cov.items():
        lines.append(
            f"  {split_name}: {info['pct_enriched']:.1f}% of {info['n_rows']:,} trades enriched"
        )
    lines += [
        "",
        f"non-zero-null columns: {(nulls > 0).sum()} of {len(df.columns)}",
        "top 10 missing columns:",
        nulls[nulls > 0].head(10).round(4).to_string(),
    ]
    out_path = OUT_DIR / "summary.txt"
    out_path.write_text("\n".join(lines))
    print(f"saved {out_path.name}")


def main() -> None:
    df = load_joined()
    nulls = panel_zero_density(df)
    cov = panel_wallet_coverage(df)
    panel_class_balance(df)
    skew = panel_distributions_and_skew(df)
    panel_outliers(df, skew)
    panel_correlation(df)
    panel_market_volume(df)
    panel_train_test_shift(df)
    panel_late_flow(df)
    panel_wallet_strata(df)
    panel_per_market_bimodality(df)
    panel_feature_stability(df)
    panel_mutual_information(df)
    panel_feature_taxonomy(df)
    panel_tail_diagnostics(df)
    panel_temporal_drift(df)
    panel_pca_wallets(df)
    panel_price_trajectories(df)
    panel_event_timing(df)
    panel_event_zoom(df)
    panel_event_volume(df)
    write_summary(df, nulls, cov)
    write_index(OUT_DIR)
    print(f"\nEDA done, outputs in {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
