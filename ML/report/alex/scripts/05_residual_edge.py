"""
05_residual_edge.py

The proper RQ1b analysis per plan §5.2: does our model's p_hat carry
information BEYOND what market_implied_prob already contains?

Aggregate ROC comparisons are misleading (see the naive-market Simpson's
paradox investigation in 04). What we actually want:

  1. Calibration curves — is p_hat calibrated within subgroups (the
     efficient-market null hypothesis)?
  2. Residual edge = p_hat - market_implied_prob per trade. Does this
     residual predict bet_correct AFTER controlling for
     market_implied_prob?
  3. Directional analysis — does the SIGN of edge correlate with
     bet_correct on trades where the direction matters?

Fits the LogReg baseline on train (same prep as 03_baselines_sweep.py).
Scores val and test. Analyses edge.

Outputs: alex/outputs/investigations/residual_edge/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "experiments"
OUT_DIR = ROOT / "alex" / "outputs" / "investigations" / "residual_edge"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same exclusion set as 03_baselines_sweep.py (keep synchronised with
# alex/notes/feature-exclusion-list.md)
NON_FEATURE_COLS = {
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
    "size",
    "price",
    "timestamp",
    "settlement_minus_trade_sec",
    "bet_correct",
    "market_implied_prob",
    "split",
    "wallet_is_whale_in_market",
    "is_position_exit",
    "time_to_settlement_s",
    "log_time_to_settlement",
    "market_volume_so_far_usd",
    "market_vol_1h_log",
    "market_vol_24h_log",
    "market_trade_count_so_far",
    "size_x_time_to_settlement",
    "wallet_prior_win_rate",
    "side",
    "outcomeIndex",
    "wallet_position_size_before_trade",
    "trade_size_vs_position_pct",
    "is_position_flip",
    "wallet_cumvol_same_side_last_10min",
    "wallet_directional_purity_in_market",
    "wallet_has_both_sides_in_market",
    "market_buy_share_running",
}

WINSORISE_COLS = ["trade_value_usd", "wallet_prior_volume_usd"]
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Prep (reuse the pipeline from 03)
# ---------------------------------------------------------------------------


def load_and_prep() -> dict:
    train = pd.read_parquet(DATA_DIR / "train.parquet").reset_index(drop=True)
    val = pd.read_parquet(DATA_DIR / "val.parquet").reset_index(drop=True)
    test = pd.read_parquet(DATA_DIR / "test.parquet").reset_index(drop=True)

    features = [c for c in train.columns if c not in NON_FEATURE_COLS]
    print(f"[features] using {len(features)}")

    # Keep `side`/`outcomeIndex` in the source dataframes for subgroup analysis
    # even though they're not features.

    # Winsorise with train bounds
    bounds = {}
    for c in WINSORISE_COLS:
        if c in train.columns:
            lo, hi = train[c].quantile([0.01, 0.99]).tolist()
            bounds[c] = (lo, hi)
            for df in (train, val, test):
                df[c] = df[c].clip(lo, hi)

    X_train = train[features]
    X_val = val[features]
    X_test = test[features]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xva = scaler.transform(imputer.transform(X_val))
    Xte = scaler.transform(imputer.transform(X_test))

    return {
        "features": features,
        "train": train,
        "val": val,
        "test": test,
        "Xtr": Xtr,
        "Xva": Xva,
        "Xte": Xte,
        "ytr": train["bet_correct"].astype(int).to_numpy(),
        "yva": val["bet_correct"].astype(int).to_numpy(),
        "yte": test["bet_correct"].astype(int).to_numpy(),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def calibration_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Return per-bin (mean p, mean y, count) for a calibration plot."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, edges[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "n": int(mask.sum()),
                "p_mean": float(p[mask].mean()),
                "y_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def partial_corr_edge_bc(
    edge: np.ndarray, y: np.ndarray, control: np.ndarray
) -> tuple[float, float]:
    """Partial correlation of edge and y controlling for `control`.
    Ordinary linear partialling."""
    # Regress edge on control
    a, b = np.polyfit(control, edge, 1)
    edge_resid = edge - (a * control + b)
    # Regress y on control
    a2, b2 = np.polyfit(control, y.astype(float), 1)
    y_resid = y.astype(float) - (a2 * control + b2)
    # Correlation
    r = np.corrcoef(edge_resid, y_resid)[0, 1]
    # Sample-size adjusted ROC using edge_resid as predictor
    # (ranks y on edge residual)
    try:
        resid_roc = float(roc_auc_score(y, edge_resid))
    except ValueError:
        resid_roc = float("nan")
    return float(r), resid_roc


def subgroup_label(side_val, oi) -> str:
    side = "BUY" if str(side_val).upper() == "BUY" else "SELL"
    return f"{side}_idx{int(oi)}"


def calibration_per_subgroup(
    df: pd.DataFrame, p_hat: np.ndarray, y: np.ndarray, name: str, out_dir: Path
) -> None:
    df = df.copy()
    df["p_hat_logreg"] = p_hat
    df["y"] = y
    df["subgroup"] = df.apply(
        lambda r: subgroup_label(r["side"], r["outcomeIndex"]), axis=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for title, col, ax in [
        ("LogReg p_hat", "p_hat_logreg", axes[0]),
        ("Market p_hat (naive)", "market_implied_prob", axes[1]),
    ]:
        ax.plot([0, 1], [0, 1], "k:", alpha=0.4, label="perfect calibration")
        for sg, g in df.groupby("subgroup"):
            cal = calibration_bins(
                g[col].to_numpy(dtype=float), g["y"].to_numpy(dtype=int)
            )
            if cal.empty:
                continue
            ax.plot(
                cal["p_mean"],
                cal["y_rate"],
                "o-",
                label=f"{sg} (n={len(g):,})",
                markersize=6,
                alpha=0.8,
            )
        ax.set_xlabel(f"{title}")
        ax.set_ylabel("realised bet_correct rate")
        ax.set_title(f"{name} — {title} calibration per subgroup")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"calibration_{name}.png", dpi=140)
    plt.close(fig)


def edge_analysis(
    df: pd.DataFrame,
    p_hat: np.ndarray,
    y: np.ndarray,
    name: str,
    out_dir: Path,
    lines: list,
) -> dict:
    df = df.copy()
    df["p_hat_logreg"] = p_hat
    df["edge"] = df["p_hat_logreg"] - df["market_implied_prob"]
    df["y"] = y

    # Aggregate stats
    edge = df["edge"].to_numpy(dtype=float)
    mip = df["market_implied_prob"].to_numpy(dtype=float)
    mask = ~np.isnan(edge)
    edge_f = edge[mask]
    mip_f = mip[mask]
    y_f = y[mask]

    lines.append(f"\n## {name} — edge = p_hat_logreg − market_implied_prob\n")
    lines.append(f"- n valid: {int(mask.sum()):,} / {len(df):,}")
    lines.append(f"- edge mean: {edge_f.mean():+.4f}")
    lines.append(f"- edge std:  {edge_f.std():.4f}")
    lines.append(f"- |edge| mean: {np.abs(edge_f).mean():.4f}")
    pct = np.percentile(edge_f, [1, 5, 25, 50, 75, 95, 99])
    lines.append(
        f"- edge percentiles (1/5/25/50/75/95/99): "
        + "  ".join(f"{v:+.3f}" for v in pct)
    )

    # Direct ROC of edge vs bet_correct (does edge predict bet_correct?)
    try:
        edge_roc = float(roc_auc_score(y_f, edge_f))
    except ValueError:
        edge_roc = float("nan")
    abs_edge_roc = (
        float(roc_auc_score(y_f, np.abs(edge_f))) if len(set(y_f)) > 1 else float("nan")
    )

    # Residual edge — does edge predict bet_correct AFTER market has its say?
    partial_r, resid_roc = partial_corr_edge_bc(edge_f, y_f, mip_f)

    lines.append(f"\n### Predictive tests (does edge predict bet_correct?)")
    lines.append(f"- ROC of raw edge: {edge_roc:.4f}")
    lines.append(f"- ROC of |edge|:   {abs_edge_roc:.4f}")
    lines.append(
        f"- Partial correlation (edge ⊥ market_implied_prob) with bet_correct: {partial_r:+.4f}"
    )
    lines.append(
        f"- ROC of edge RESIDUAL (after linearly subtracting market_implied_prob): {resid_roc:.4f}"
    )

    # Per-subgroup edge signal
    df["subgroup"] = df.apply(
        lambda r: subgroup_label(r["side"], r["outcomeIndex"]), axis=1
    )
    lines.append(f"\n### Per-subgroup edge direction and predictive value\n")
    lines.append(
        "| Subgroup | n | mean edge | |edge| mean | ROC (edge → bc) | bc_rate |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    for sg, g in df.groupby("subgroup"):
        g_mask = ~g["edge"].isna()
        if g_mask.sum() < 5:
            continue
        e = g.loc[g_mask, "edge"].to_numpy(dtype=float)
        yy = g.loc[g_mask, "y"].to_numpy(dtype=int)
        try:
            sg_roc = float(roc_auc_score(yy, e))
            sg_roc_str = f"{sg_roc:.4f}"
        except ValueError:
            sg_roc_str = "—"
        lines.append(
            f"| `{sg}` | {int(g_mask.sum()):,} | {e.mean():+.4f} | "
            f"{np.abs(e).mean():.4f} | {sg_roc_str} | {yy.mean():.3f} |"
        )

    # Plot edge distribution + edge vs bet_correct
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(edge_f, bins=50, color="#3b82f6", alpha=0.7)
    axes[0].axvline(0, color="black", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("edge = p_hat_logreg − market_implied_prob")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{name} — edge distribution")

    # Edge vs bet_correct mean rate, binned
    edges = np.linspace(edge_f.min(), edge_f.max(), 21)
    idx = np.digitize(edge_f, edges[1:-1], right=False)
    xs, ys, ns = [], [], []
    for b in range(20):
        m = idx == b
        if m.sum() < 30:
            continue
        xs.append(float(edge_f[m].mean()))
        ys.append(float(y_f[m].mean()))
        ns.append(int(m.sum()))
    axes[1].axhline(
        float(y_f.mean()),
        color="black",
        linestyle=":",
        alpha=0.4,
        label=f"cohort bc_rate ({y_f.mean():.3f})",
    )
    axes[1].plot(xs, ys, "o-", color="#10b981")
    for x, y_, n in zip(xs, ys, ns):
        axes[1].annotate(
            f"n={n:,}", (x, y_), xytext=(3, 3), textcoords="offset points", fontsize=6
        )
    axes[1].set_xlabel("edge bin midpoint")
    axes[1].set_ylabel("realised bet_correct rate in bin")
    axes[1].set_title(f"{name} — does edge magnitude/sign predict bet_correct?")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"edge_{name}.png", dpi=140)
    plt.close(fig)

    return {
        "edge_mean": float(edge_f.mean()),
        "edge_std": float(edge_f.std()),
        "roc_of_edge": edge_roc,
        "roc_of_abs_edge": abs_edge_roc,
        "partial_r": partial_r,
        "roc_of_edge_residual": resid_roc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    data = load_and_prep()

    print("[fit] LogisticRegression(class_weight='balanced', max_iter=2000)")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(data["Xtr"], data["ytr"])
    print("[fit] done")

    p_tr = model.predict_proba(data["Xtr"])[:, 1]
    p_va = model.predict_proba(data["Xva"])[:, 1]
    p_te = model.predict_proba(data["Xte"])[:, 1]

    lines = [
        "# Residual-edge analysis (RQ1b per plan §5.2)",
        "",
        "_Does the trained model's `p_hat` carry predictive information "
        "BEYOND what `market_implied_prob` already encodes? Evaluated via "
        "calibration, edge distribution, and residual ROC._",
        "",
        "## Model: LogisticRegression(class_weight='balanced') on leakage-free "
        "features (32 features after all P0 pruning — see `data-pipeline-issues.md`).",
        "",
    ]

    # Calibration plots per subgroup
    calibration_per_subgroup(data["train"], p_tr, data["ytr"], "train", OUT_DIR)
    calibration_per_subgroup(data["val"], p_va, data["yva"], "val", OUT_DIR)
    calibration_per_subgroup(data["test"], p_te, data["yte"], "test", OUT_DIR)

    # Edge analysis per split
    summary = {}
    for name, df, p, y in [
        ("train", data["train"], p_tr, data["ytr"]),
        ("val", data["val"], p_va, data["yva"]),
        ("test", data["test"], p_te, data["yte"]),
    ]:
        summary[name] = edge_analysis(df, p, y, name, OUT_DIR, lines)

    lines.append("\n## Plots\n")
    lines.append(
        "- `calibration_{train,val,test}.png` — left panel: LogReg p_hat; right panel: market_implied_prob. Per-subgroup calibration curves."
    )
    lines.append(
        "- `edge_{train,val,test}.png` — edge distribution + edge vs realised bet_correct rate (binned)."
    )

    (OUT_DIR / "report.md").write_text("\n".join(lines))
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Stdout summary
    print("\n" + "=" * 80)
    print("RESIDUAL-EDGE ANALYSIS — KEY NUMBERS")
    print("=" * 80)
    for name, s in summary.items():
        print(f"\n{name}:")
        print(f"  edge mean        = {s['edge_mean']:+.4f}")
        print(f"  edge std         = {s['edge_std']:.4f}")
        print(f"  ROC of edge → bc = {s['roc_of_edge']:.4f}")
        print(f"  ROC of |edge| → bc = {s['roc_of_abs_edge']:.4f}")
        print(f"  Partial r(edge, bc | mip) = {s['partial_r']:+.4f}")
        print(
            f"  ROC of edge residual (after MIP subtracted) = {s['roc_of_edge_residual']:.4f}"
        )
    print(f"\n[done] outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
