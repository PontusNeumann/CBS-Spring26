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


# ---------------------------------------------------------------------------
# 09. Late-flow signature — base rate vs time-to-deadline buckets.
# Replicates the Mitts & Ofir 2026 setup that motivated the cohort design.
# ---------------------------------------------------------------------------
def panel_late_flow(df: pd.DataFrame) -> None:
    if "log_time_to_deadline_hours" not in df.columns:
        print("skipping late-flow panel — log_time_to_deadline_hours missing")
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

    out_path = OUT_DIR / "09_late_flow.txt"
    with out_path.open("w") as f:
        f.write("Base rate and trade share by time-to-deadline bucket\n\n")
        f.write(
            agg.pivot(index="_bucket", columns=SPLIT, values=["base_rate", "n"]).round(4).to_string()
        )
    print(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# 10. Wallet-stratum base rates — motivates Layer-6 enrichment.
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
# 11. Per-market base rate — the single-event-resolution bimodality.
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
# 12. Single-feature ROC per market — feature-stability heatmap.
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
        print("skipping feature-stability panel — no markets met thresholds")
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
# 13. Mutual-information feature ranking — non-linear signal counterpart
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
# 14. Feature-group taxonomy — how the 79 features split across layers.
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

    out_path = OUT_DIR / "14_feature_taxonomy.txt"
    with out_path.open("w") as f:
        f.write("Feature-group membership\n\n")
        for group, members in FEATURE_GROUPS:
            present = [m for m in members if m in feats]
            f.write(f"== {group} ({len(present)}) ==\n")
            for m in present:
                f.write(f"  {m}\n")
            f.write("\n")
        if rest:
            f.write("== Other / unclassified ==\n")
            for m in rest:
                f.write(f"  {m}\n")
    print(f"saved {out_path.name}")


# ---------------------------------------------------------------------------
# 15. Distribution-tail diagnostics — fat-tail screen.
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
# 16. Temporal base-rate drift — rolling base rate vs trade time.
# MA2-style rolling-window drift detection. Tests whether bet_correct base
# rate is stable across the strike-countdown / ceasefire-countdown windows.
# ---------------------------------------------------------------------------
def panel_temporal_drift(df: pd.DataFrame) -> None:
    if "timestamp" not in df.columns:
        print("skipping temporal-drift panel — timestamp missing")
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


def write_index(out_dir: Path) -> None:
    """Cheat-sheet that maps each generated panel to what it shows."""
    items = [
        ("01_missingness.png", "Missing-value share per column, split by train/test."),
        ("02_wallet_coverage.png", "% of trades with enriched wallet, by split."),
        ("03_class_balance.png", "Base rate per split + per-market base-rate spread."),
        ("04_distributions.png", "Top-12 skewed features, KDE by bet_correct."),
        ("04_skewness_table.csv", "All numeric features ranked by absolute skew."),
        ("05_outlier_boxplots.png", "Box plots for the 8 most-skewed features."),
        ("06_correlation_heatmap.png", "Upper-triangle Pearson heatmap, top-40 features by std."),
        ("06_top_correlations.txt", "Top 25 |Pearson r| feature pairs."),
        ("07_market_volume.png", "Per-market trade count vs base rate."),
        ("08_train_test_shift.png", "Top-15 features by |Cohen's d| between train and test."),
        ("09_late_flow.png", "Hit rate vs time-to-deadline (Mitts & Ofir-style buckets)."),
        ("10_wallet_strata.png", "Hit rate by wallet age decile, CEX-funding, polygon nonce decile."),
        ("11_per_market_bimodality.png", "Per-market base-rate histogram (single-event resolution)."),
        ("12_feature_stability.png", "Single-feature ROC-AUC heatmap per market for top-8 features."),
        ("13_mutual_information.png", "Top-20 features by mutual information with bet_correct."),
        ("14_feature_taxonomy.png", "How the numeric features split across feature-engineering layers."),
        ("15_tail_diagnostics.png", "Top-15 fat-tailed features by |excess kurtosis|. CSV gives p1/p5/p95/p99 + tail-conditional means."),
        ("16_temporal_drift.png", "Daily bet_correct base rate with 7-day rolling mean, per split."),
        ("summary.txt", "Plain-text summary of dataset shape, base rates, missingness."),
    ]
    lines = ["# EDA index — wallet-joined Alex cohort", ""]
    for name, blurb in items:
        path = out_dir / name
        present = "✓" if path.exists() else "—"
        lines.append(f"- [{present}] **{name}** — {blurb}")
    (out_dir / "index.md").write_text("\n".join(lines) + "\n")
    print(f"saved index.md")


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
    panel_late_flow(df)
    panel_wallet_strata(df)
    panel_per_market_bimodality(df)
    panel_feature_stability(df)
    panel_mutual_information(df)
    panel_feature_taxonomy(df)
    panel_tail_diagnostics(df)
    panel_temporal_drift(df)
    write_summary(df, nulls, cov)
    write_index(OUT_DIR)
    print(f"\nEDA done — outputs in {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
