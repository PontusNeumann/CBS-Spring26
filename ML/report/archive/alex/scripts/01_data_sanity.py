"""
01_data_sanity.py

Quick data sanity pass on the train/val/test cohorts before modelling.
Goal: catch surprises (pathological NaN columns, target skew, feature
distributions that would crash a model, etc.) before any training loop runs.

Outputs: `alex/outputs/data_sanity/` — Markdown summary + a few PNG plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # .../ML/report
DATA_DIR = ROOT / "data" / "experiments"
OUT_DIR = ROOT / "alex" / "outputs" / "data_sanity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature exclusion list — mirrors alex/notes/feature-exclusion-list.md
NON_FEATURE_COLS = {
    # identifiers / metadata
    "proxyWallet",
    "asset",
    "transactionHash",
    "condition_id",
    "conditionId",
    "source",
    "title",
    "slug_x",
    "slug_y",
    "icon",
    "eventSlug",
    "outcome",
    "name",
    "pseudonym",
    "bio",
    "profileImage",
    "profileImageOptimized",
    "question",
    "end_date",
    "winning_outcome_index",
    "resolved",
    "resolution_ts",
    "outcomes",
    "is_yes",
    # raw cols superseded
    "size",
    "price",
    "timestamp",
    # filter / label / benchmark / split
    "settlement_minus_trade_sec",
    "bet_correct",
    "market_implied_prob",
    "split",
    # leakage mitigations (P0-1, P0-2)
    "wallet_is_whale_in_market",
    "is_position_exit",
    # market-identifying absolute-scale (P0-8)
    "time_to_settlement_s",
    "log_time_to_settlement",
    "market_volume_so_far_usd",
    "market_vol_1h_log",
    "market_vol_24h_log",
    "market_trade_count_so_far",
    "size_x_time_to_settlement",
}


def log(msg: str, *, lines: list[str] | None = None) -> None:
    print(msg)
    if lines is not None:
        lines.append(msg)


def describe_cohort(
    name: str, df: pd.DataFrame, features: list[str], lines: list[str]
) -> None:
    lines.append(f"\n## {name}\n")
    log(f"\n=== {name} ===", lines=lines)

    # Shape + target rate
    n, c = df.shape
    bc = df["bet_correct"].astype(float)
    log(f"- shape: {n:,} rows × {c} cols", lines=lines)
    log(
        f"- bet_correct rate: {bc.mean():.4f} ({int(bc.sum()):,} positives)",
        lines=lines,
    )
    log(f"- feature cols after drops: {len(features)}", lines=lines)

    # Per-market breakdown
    lines.append("\n**Per-market breakdown**\n")
    lines.append("| Trades | is_yes | bc_rate | Question |\n|---:|:---:|---:|---|")
    for q, g in df.groupby("question"):
        is_yes = int(g["is_yes"].iloc[0])
        row = f"| {len(g):,} | {'YES' if is_yes else 'NO'} | {g['bet_correct'].mean():.3f} | {q} |"
        lines.append(row)
        log(
            f"  [{len(g):>7,}]  {'YES' if is_yes else 'NO '}  bc={g['bet_correct'].mean():.3f}  {q}",
            lines=None,
        )

    # NaN counts per feature
    lines.append("\n**NaN counts (features with any NaN)**\n")
    lines.append("| Feature | NaN count | NaN % |\n|---|---:|---:|")
    nan_counts = df[features].isna().sum().sort_values(ascending=False)
    any_nan = nan_counts[nan_counts > 0]
    if any_nan.empty:
        lines.append("| *none* | — | — |")
        log("  NaN: none in features", lines=None)
    else:
        for col, cnt in any_nan.items():
            pct = 100 * cnt / len(df)
            lines.append(f"| `{col}` | {int(cnt):,} | {pct:.1f}% |")
        log(f"  NaN: {len(any_nan)} feature cols have NaNs (see summary)", lines=None)

    # Feature dtype summary
    dtypes = df[features].dtypes.value_counts()
    log(f"  dtypes: {dict(dtypes)}", lines=None)
    lines.append(f"\n**Feature dtypes:** `{dict(dtypes)}`\n")


def side_encoding_report(df: pd.DataFrame, name: str, lines: list[str]) -> None:
    if "side" not in df.columns:
        return
    vc = df["side"].value_counts(dropna=False).to_dict()
    lines.append(f"**`side` encoding ({name}):** `{vc}`\n")
    log(f"  side values: {vc}", lines=None)


def feature_stats(train: pd.DataFrame, features: list[str], lines: list[str]) -> None:
    lines.append("\n## Feature summary (train only)\n")
    log("\n=== Feature summary (train) ===", lines=lines)

    tr = train[features].copy()
    if "side" in tr.columns:
        tr["side"] = (tr["side"].astype(str).str.upper() == "BUY").astype(int)
    numeric = tr.select_dtypes(include=[np.number])
    lines.append(
        "| Feature | Mean | Std | Min | 25% | 50% | 75% | Max | NaN% |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    desc = numeric.describe().T
    nan_pct = tr.isna().mean() * 100
    for col in sorted(numeric.columns):
        d = desc.loc[col]
        lines.append(
            f"| `{col}` | {d['mean']:.3g} | {d['std']:.3g} | {d['min']:.3g} | "
            f"{d['25%']:.3g} | {d['50%']:.3g} | {d['75%']:.3g} | {d['max']:.3g} | "
            f"{nan_pct[col]:.1f}% |"
        )


def correlations_with_target(
    train: pd.DataFrame, features: list[str], lines: list[str]
) -> None:
    lines.append(
        "\n## Top absolute Pearson correlations with `bet_correct` (train only)\n"
    )
    log("\n=== Feature ↔ target correlations (train) ===", lines=lines)

    tr = train[features + ["bet_correct"]].copy()
    if "side" in tr.columns:
        tr["side"] = (tr["side"].astype(str).str.upper() == "BUY").astype(int)
    tr = tr.select_dtypes(include=[np.number])

    # Compute correlations (NaN-safe, pairwise)
    corr = tr.corr(method="pearson", numeric_only=True)["bet_correct"].drop(
        "bet_correct"
    )
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(20)

    lines.append("| Feature | Pearson r |\n|---|---:|")
    for col, val in top.items():
        lines.append(f"| `{col}` | {val:+.4f} |")
    for col, val in top.head(10).items():
        log(f"  {val:+.4f}  {col}", lines=None)

    # Warn on anything suspicious (|r| > 0.3 suggests leak or trivial predictor)
    suspicious = corr[corr.abs() > 0.3]
    if not suspicious.empty:
        lines.append(
            f"\n**⚠ Features with |r| > 0.3 on target** (potential leakage or trivial predictor):\n"
        )
        for col, val in suspicious.items():
            lines.append(f"- `{col}` → r = {val:+.4f}")
            log(f"  ⚠ |r|>0.3: {val:+.4f}  {col}", lines=None)


def plot_target_rate_per_market(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    rows = []
    for name, df in [("train", train), ("val", val), ("test", test)]:
        for q, g in df.groupby("question"):
            rows.append(
                dict(
                    cohort=name,
                    market=q[:50] + ("…" if len(q) > 50 else ""),
                    n=len(g),
                    bc_rate=g["bet_correct"].mean(),
                )
            )
    tbl = pd.DataFrame(rows).sort_values(["cohort", "n"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(tbl))))
    colors = {"train": "#3b82f6", "val": "#f59e0b", "test": "#10b981"}
    for i, r in enumerate(tbl.itertuples()):
        ax.barh(i, r.bc_rate, color=colors[r.cohort], alpha=0.75)
        ax.text(
            r.bc_rate + 0.005,
            i,
            f"{r.bc_rate:.3f}  (n={r.n:,})",
            va="center",
            fontsize=8,
        )
    ax.axvline(0.5, color="black", linestyle=":", lw=0.8, alpha=0.4)
    ax.set_yticks(range(len(tbl)))
    ax.set_yticklabels(tbl["market"], fontsize=8)
    ax.set_xlabel("bet_correct rate")
    ax.set_title("Target rate per market")
    ax.set_xlim(0.35, 0.65)
    ax.invert_yaxis()

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=n) for n, c in colors.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "target_rate_per_market.png", dpi=140)
    plt.close(fig)


def plot_feature_correlations(train: pd.DataFrame, features: list[str]) -> None:
    tr = train[features + ["bet_correct"]].copy()
    if "side" in tr.columns:
        tr["side"] = (tr["side"].astype(str).str.upper() == "BUY").astype(int)
    tr = tr.select_dtypes(include=[np.number])
    corr = tr.corr(method="pearson", numeric_only=True)["bet_correct"].drop(
        "bet_correct"
    )
    ordered = corr.reindex(corr.abs().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(9, max(6, 0.22 * len(ordered))))
    colors = ["#10b981" if v > 0 else "#ef4444" for v in ordered.values]
    ax.barh(range(len(ordered)), ordered.values, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered.index, fontsize=7)
    ax.set_xlabel("Pearson r with bet_correct")
    ax.set_title("Feature ↔ target linear correlations (train)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_correlations.png", dpi=140)
    plt.close(fig)


def main() -> None:
    lines: list[str] = ["# Data sanity report", ""]
    lines.append(
        "_Generated by `alex/scripts/01_data_sanity.py`. Quick pre-modelling audit of the train/val/test cohorts._\n"
    )

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    features = [c for c in train.columns if c not in NON_FEATURE_COLS]
    log(f"features in use: {len(features)}", lines=lines)

    for nm, df in [("Train", train), ("Val", val), ("Test", test)]:
        describe_cohort(nm, df, features, lines)
        side_encoding_report(df, nm, lines)

    feature_stats(train, features, lines)
    correlations_with_target(train, features, lines)

    plot_target_rate_per_market(train, val, test)
    plot_feature_correlations(train, features)

    lines.append("\n## Plots\n")
    lines.append("- `target_rate_per_market.png`")
    lines.append("- `feature_correlations.png`")

    report = OUT_DIR / "report.md"
    report.write_text("\n".join(lines))
    log(f"\n[done] report → {report}", lines=None)


if __name__ == "__main__":
    main()
