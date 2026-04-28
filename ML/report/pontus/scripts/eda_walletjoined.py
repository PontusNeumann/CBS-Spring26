"""EDA for the wallet-joined Alex cohort.

Reads `pontus/data/{train,test}_features_walletjoined.parquet` (Alex's 70
engineered features + 12 Layer-6 wallet features + meta cols), tags rows
with split = {train, test}, and writes the report-style panels to
`pontus/outputs/eda_walletjoined/`.

Panels that fit the joined schema:
    01  shape + dtypes + missingness                 (incl. wallet-coverage breakdown)
    02  class balance per split + per market         (target = bet_correct)
    03  feature distributions split by target + skewness table
    04  outlier boxplots on top-skew features
    05  correlation heatmap (lower triangle) + redundant pairs
    06  per-market trade count + base-rate spread
    07  train-vs-test distribution shift on top features

Panels skipped (joined parquet is feature-only — no wallet ID, no price
trajectory series, no deadline_ts column at the level the original EDA
needed):
    - wallet PCA, wallet quadrants, price trajectories, event timing.

Re-run after the wallet-extras enrichment finishes; the wallet-coverage
panel will jump from 59.8% on test to ~99%+.

Usage:
    python pontus/scripts/eda_walletjoined.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "pontus" / "data"
OUT_DIR = ROOT / "pontus" / "outputs" / "eda_walletjoined"
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
    train = pd.read_parquet(DATA_DIR / "train_features_walletjoined.parquet")
    test = pd.read_parquet(DATA_DIR / "test_features_walletjoined.parquet")
    train[SPLIT] = "train"
    test[SPLIT] = "test"
    df = pd.concat([train, test], ignore_index=True)
    print(f"loaded joined dataset: {len(df):,} rows × {len(df.columns)} cols")
    return df


def panel_missingness(df: pd.DataFrame) -> pd.Series:
    nulls_overall = df.isna().mean().sort_values(ascending=False)
    keep = nulls_overall[nulls_overall > 0].head(30)
    if keep.empty:
        print("no missing values found — skipping missingness panel")
        return nulls_overall

    nulls_by_split = (
        df.groupby(SPLIT)[keep.index.tolist()].apply(lambda g: g.isna().mean())
    )

    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 0.32 * len(keep) + 1.0))
    width = 0.4
    y = np.arange(len(keep))
    train_pct = nulls_by_split.loc["train"].reindex(keep.index).values * 100
    test_pct = nulls_by_split.loc["test"].reindex(keep.index).values * 100
    ax.barh(y - width / 2, train_pct, width, color=COL_TRAIN, label="train")
    ax.barh(y + width / 2, test_pct, width, color=COL_TEST, label="test")
    ax.set_yticks(y)
    ax.set_yticklabels(keep.index, fontsize=8)
    ax.set_xlabel("% missing")
    ax.set_title("Missingness by feature, split (top 30 with any missing)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    clean_ax(ax)
    save_fig(fig, OUT_DIR / "01_missingness.png")
    return nulls_overall


def panel_wallet_coverage(df: pd.DataFrame) -> dict:
    cov = (
        df.groupby(SPLIT)[WALLET_FLAG]
        .agg(["mean", "size"])
        .rename(columns={"mean": "pct_enriched", "size": "n_rows"})
    )
    cov["pct_enriched"] *= 100

    fig, ax = plt.subplots(figsize=(FIG_W, 2.4))
    bars = ax.bar(
        cov.index, cov["pct_enriched"],
        color=[COL_TRAIN if s == "train" else COL_TEST for s in cov.index],
    )
    ax.set_ylabel("% trades with enriched wallet")
    ax.set_ylim(0, 105)
    ax.set_title("Wallet-feature coverage by split")
    for bar, pct, n in zip(bars, cov["pct_enriched"], cov["n_rows"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{pct:.1f}%\n({n:,} trades)",
            ha="center", va="bottom", fontsize=9,
        )
    clean_ax(ax)
    save_fig(fig, OUT_DIR / "02_wallet_coverage.png")

    out_path = OUT_DIR / "02_wallet_coverage.txt"
    with out_path.open("w") as f:
        f.write("wallet feature coverage by split\n\n")
        f.write(cov.round(2).to_string())
    print(f"saved {out_path.name}")
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
        data=per_market, x=SPLIT, y=TARGET,
        ax=ax, palette={"train": COL_TRAIN, "test": COL_TEST},
        width=0.45, fliersize=2,
    )
    ax.axhline(0.5, color=COL_DARK, ls="--", lw=0.8)
    ax.set_ylabel("per-market base rate")
    ax.set_title("Per-market base-rate spread")
    clean_ax(ax)

    fig.tight_layout()
    save_fig(fig, OUT_DIR / "03_class_balance.png")


def _numeric_features(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in NON_FEATURE
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def panel_distributions_and_skew(df: pd.DataFrame) -> pd.DataFrame:
    feats = _numeric_features(df)
    sample = df[feats + [TARGET]].sample(min(200_000, len(df)), random_state=42)
    skew = sample[feats].skew(numeric_only=True).sort_values(key=np.abs, ascending=False)

    out_skew = OUT_DIR / "04_skewness_table.csv"
    skew.to_frame("skew").round(3).to_csv(out_skew)
    print(f"saved {out_skew.name}")

    top12 = skew.index[:12].tolist()
    fig, axes = plt.subplots(3, 4, figsize=(FIG_W_WIDE, 6.6))
    for ax, feat in zip(axes.flat, top12):
        for outcome, colour in ((1, COL_CORRECT), (0, COL_INCORRECT)):
            data = sample.loc[sample[TARGET] == outcome, feat].dropna()
            if data.empty:
                continue
            sns.kdeplot(
                data, ax=ax, fill=True, alpha=0.35, color=colour, lw=1,
                common_norm=False,
            )
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        clean_ax(ax)
    fig.suptitle("Top-12 skewed features — KDE by bet_correct", y=1.01)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "04_distributions.png")
    return skew


def panel_outliers(df: pd.DataFrame, skew: pd.DataFrame, top_k: int = 8) -> None:
    feats = skew.index[:top_k].tolist()
    sample = df[feats].sample(min(50_000, len(df)), random_state=42)
    long = sample.melt(var_name="feature", value_name="value").dropna()
    fig, ax = plt.subplots(figsize=(FIG_W_WIDE, 3.2))
    sns.boxplot(
        data=long, x="feature", y="value",
        ax=ax, color=PAL_10[5], width=0.5, fliersize=2,
    )
    ax.set_title(f"Top-{top_k} skewed features — value spread")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    clean_ax(ax)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "05_outlier_boxplots.png")


def panel_correlation(df: pd.DataFrame) -> None:
    feats = _numeric_features(df)
    sample = df[feats].sample(min(150_000, len(df)), random_state=42)
    corr = sample.corr(numeric_only=True).fillna(0)

    # Restrict to top-40 features by std to keep the heatmap readable.
    if len(corr) > 40:
        std_rank = sample.std(numeric_only=True).sort_values(ascending=False)
        keep = std_rank.head(40).index.tolist()
        corr = corr.loc[keep, keep]

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
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
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

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_WIDE, 3.0))
    ax = axes[0]
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
    ax.legend(frameon=False, loc="lower right")
    clean_ax(ax)

    ax = axes[1]
    summary = per.groupby(SPLIT)["n"].describe()[["count", "min", "50%", "max"]]
    summary.columns = ["n_markets", "min_trades", "median_trades", "max_trades"]
    ax.axis("off")
    table = ax.table(
        cellText=summary.round(0).astype(int).values,
        rowLabels=summary.index, colLabels=summary.columns,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.05, 1.4)
    ax.set_title("Cohort sizing summary")

    fig.tight_layout()
    save_fig(fig, OUT_DIR / "07_market_volume.png")

    out_path = OUT_DIR / "07_market_volume.txt"
    with out_path.open("w") as f:
        f.write("per-market trade counts and base rates\n\n")
        f.write(per.sort_values(["split", "n"], ascending=[True, False]).to_string(index=False))
    print(f"saved {out_path.name}")


def panel_train_test_shift(df: pd.DataFrame) -> None:
    feats = _numeric_features(df)
    train = df[df[SPLIT] == "train"][feats]
    test = df[df[SPLIT] == "test"][feats]
    if train.empty or test.empty:
        print("skipping shift panel — one split missing")
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

    out_path = OUT_DIR / "08_train_test_shift.txt"
    with out_path.open("w") as f:
        f.write("|standardised mean shift| (test − train) on numeric features\n")
        f.write("interpretation: >0.2 small, >0.5 medium, >0.8 large (Cohen's d convention)\n\n")
        f.write(shift.round(3).to_string())
    print(f"saved {out_path.name}")


def write_summary(df: pd.DataFrame, nulls: pd.Series, cov: dict) -> None:
    feats = _numeric_features(df)
    lines = [
        "EDA summary — wallet-joined Alex cohort",
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
    nulls = panel_missingness(df)
    cov = panel_wallet_coverage(df)
    panel_class_balance(df)
    skew = panel_distributions_and_skew(df)
    panel_outliers(df, skew)
    panel_correlation(df)
    panel_market_volume(df)
    panel_train_test_shift(df)
    write_summary(df, nulls, cov)
    print(f"\nEDA done — outputs in {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
