"""
03_baselines_sweep.py

Runs a bank of non-neural baselines on the same leakage-free train/val/test
frame. Same prep pipeline as 02_baseline_logreg.py; different estimators.
Produces a side-by-side comparison table so we can see where the signal
genuinely lives before committing to any one model family.

Models in sweep:
  - LogReg (L2)            — linear, already in 02, re-run here for consistency
  - LogReg (L1, saga)      — sparse linear; reveals which features are load-bearing
  - Random Forest          — tree bagging
  - HistGradientBoosting   — sklearn's GBM, strong on tabular
  - Extra Trees            — randomised forest

MLP is NOT here — different training dynamics (epochs, early stopping);
gets its own script.

Outputs under `alex/outputs/baselines/sweep/`:
  - metrics_summary.json            — side-by-side ROC/PR/Brier/ECE
  - <model>/metrics.json            — per-model detail
  - <model>/per_market.json         — per-market test breakdown
  - <model>/feature_importance.json — signed coef or tree importances
  - comparison_table.md             — human-readable summary
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent  # .../ML/report
DATA_DIR = ROOT / "data" / "experiments"
OUT_DIR = ROOT / "alex" / "outputs" / "baselines" / "sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical feature-exclusion set (mirrors alex/notes/feature-exclusion-list.md)
# ---------------------------------------------------------------------------

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
    # P0-11: (side, outcomeIndex) together deterministically encode bet_correct
    # within any market, and the mapping flips across resolution types. Tree
    # models exploited the interaction and catastrophically inverted on test
    # (ROC 0.00-0.04). Drop both to force models to predict from wallet/market
    # context alone.
    "side",
    "outcomeIndex",
    # P0-12: indirectly direction-dependent features. These use signed
    # position, same-side filtering, or side/outcomeIndex aggregates —
    # re-introducing the inversion channel through the back door. Drop all.
    "wallet_position_size_before_trade",  # signed cumulative position
    "trade_size_vs_position_pct",  # uses signed position
    "is_position_flip",  # detects sign change of signed pos
    "wallet_cumvol_same_side_last_10min",  # explicitly same-side filtered
    "wallet_directional_purity_in_market",  # share of trades on one outcomeIndex
    "wallet_has_both_sides_in_market",  # indicator on outcomeIndex distribution
    "market_buy_share_running",  # running share of BUY (0.38 train vs 0.67 test)
}

WINSORISE_COLS = ["trade_value_usd", "wallet_prior_volume_usd"]
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece_15bin": expected_calibration_error(y_true, y_prob),
    }


def per_market_metrics(df: pd.DataFrame, p: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for q, g in df.groupby("question"):
        positions = df.index.get_indexer(g.index.to_numpy())
        p_m, y_m = p[positions], y[positions]
        roc = float(roc_auc_score(y_m, p_m)) if len(set(y_m)) > 1 else None
        pr = float(average_precision_score(y_m, p_m)) if len(set(y_m)) > 1 else None
        rows.append(
            {
                "question": q,
                "n": int(len(g)),
                "is_yes": int(g["is_yes"].iloc[0]),
                "bc_rate": float(y_m.mean()),
                "p_hat_mean": float(p_m.mean()),
                "p_hat_std": float(p_m.std()),
                "roc_auc": roc,
                "pr_auc": pr,
                "brier": float(brier_score_loss(y_m, p_m)),
            }
        )
    return sorted(rows, key=lambda r: -r["n"])


# ---------------------------------------------------------------------------
# Data prep (shared across all models)
# ---------------------------------------------------------------------------


def load_and_prep() -> tuple:
    print(f"[load] {DATA_DIR}")
    train = pd.read_parquet(DATA_DIR / "train.parquet").reset_index(drop=True)
    val = pd.read_parquet(DATA_DIR / "val.parquet").reset_index(drop=True)
    test = pd.read_parquet(DATA_DIR / "test.parquet").reset_index(drop=True)
    for nm, df in [("train", train), ("val", val), ("test", test)]:
        print(f"  {nm}: {len(df):,} rows × {df.shape[1]} cols")

    features = [c for c in train.columns if c not in NON_FEATURE_COLS]
    print(f"[features] {len(features)} feature cols")

    for df in (train, val, test):
        if "side" in df.columns:
            df["side"] = (df["side"].astype(str).str.upper() == "BUY").astype(int)

    y_train = train["bet_correct"].astype(int).to_numpy()
    y_val = val["bet_correct"].astype(int).to_numpy()
    y_test = test["bet_correct"].astype(int).to_numpy()

    X_train = train[features].copy()
    X_val = val[features].copy()
    X_test = test[features].copy()

    # Winsorise with train-only bounds
    bounds = {}
    for c in WINSORISE_COLS:
        if c in X_train.columns:
            lo, hi = X_train[c].quantile([0.01, 0.99]).tolist()
            bounds[c] = (lo, hi)
            X_train[c] = X_train[c].clip(lo, hi)
            X_val[c] = X_val[c].clip(lo, hi)
            X_test[c] = X_test[c].clip(lo, hi)

    # Impute + scale (for linear models). Tree models are scale-invariant so
    # re-use the same arrays for consistency.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xva = scaler.transform(imputer.transform(X_val))
    Xte = scaler.transform(imputer.transform(X_test))

    return {
        "features": features,
        "bounds": bounds,
        "X_train": Xtr,
        "y_train": y_train,
        "train": train,
        "X_val": Xva,
        "y_val": y_val,
        "val": val,
        "X_test": Xte,
        "y_test": y_test,
        "test": test,
    }


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def make_models() -> dict:
    return {
        "logreg_l2": LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "logreg_l1": LogisticRegression(
            penalty="l1",
            C=1.0,
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "hist_gbm": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "gaussian_nb": GaussianNB(),
    }


# ---------------------------------------------------------------------------
# Training + scoring
# ---------------------------------------------------------------------------


def fit_score_save(name: str, model, data: dict, summary: dict) -> None:
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n[{name}] fitting …", flush=True)
    model.fit(data["X_train"], data["y_train"])
    fit_sec = time.time() - t0
    print(f"[{name}] fit done in {fit_sec:.1f}s")

    p_train = model.predict_proba(data["X_train"])[:, 1]
    p_val = model.predict_proba(data["X_val"])[:, 1]
    p_test = model.predict_proba(data["X_test"])[:, 1]

    metrics = {
        "model": name,
        "fit_seconds": round(fit_sec, 1),
        "n_features": len(data["features"]),
        "train": probability_metrics(data["y_train"], p_train),
        "val": probability_metrics(data["y_val"], p_val),
        "test": probability_metrics(data["y_test"], p_test),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    per_market = {
        "val": per_market_metrics(data["val"], p_val, data["y_val"]),
        "test": per_market_metrics(data["test"], p_test, data["y_test"]),
    }
    (out / "per_market.json").write_text(json.dumps(per_market, indent=2))

    # Feature importance — signed for linear, gain-based for trees
    features = data["features"]
    if hasattr(model, "coef_"):
        vals = model.coef_[0]
        fi = [
            {"feature": f, "coef": float(v), "abs_coef": float(abs(v))}
            for f, v in zip(features, vals)
        ]
        fi.sort(key=lambda r: r["abs_coef"], reverse=True)
        (out / "feature_importance.json").write_text(
            json.dumps(
                {"intercept": float(model.intercept_[0]), "features": fi}, indent=2
            )
        )
    elif hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        fi = [{"feature": f, "importance": float(v)} for f, v in zip(features, vals)]
        fi.sort(key=lambda r: r["importance"], reverse=True)
        (out / "feature_importance.json").write_text(
            json.dumps({"features": fi}, indent=2)
        )
    else:
        # HistGradientBoosting doesn't expose feature_importances_ by default
        # (permutation importance is the recommended path). Skip here; add
        # permutation importance in a follow-up if we commit to HGBM.
        pass

    summary[name] = {
        "fit_seconds": round(fit_sec, 1),
        "train_roc": metrics["train"]["roc_auc"],
        "val_roc": metrics["val"]["roc_auc"],
        "test_roc": metrics["test"]["roc_auc"],
        "train_pr": metrics["train"]["pr_auc"],
        "val_pr": metrics["val"]["pr_auc"],
        "test_pr": metrics["test"]["pr_auc"],
        "train_brier": metrics["train"]["brier"],
        "val_brier": metrics["val"]["brier"],
        "test_brier": metrics["test"]["brier"],
        "train_ece": metrics["train"]["ece_15bin"],
        "val_ece": metrics["val"]["ece_15bin"],
        "test_ece": metrics["test"]["ece_15bin"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_comparison_table(summary: dict) -> None:
    lines = [
        "# Baseline sweep — comparison",
        "",
        "_Generated by `alex/scripts/03_baselines_sweep.py`._",
        "",
        "## ROC-AUC",
        "",
        "| Model | Train | Val | Test | Fit (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| `{name}` | {s['train_roc']:.4f} | {s['val_roc']:.4f} | "
            f"{s['test_roc']:.4f} | {s['fit_seconds']:.1f} |"
        )
    lines += [
        "",
        "## PR-AUC",
        "",
        "| Model | Train | Val | Test |",
        "|---|---:|---:|---:|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| `{name}` | {s['train_pr']:.4f} | {s['val_pr']:.4f} | {s['test_pr']:.4f} |"
        )
    lines += [
        "",
        "## Brier score (lower is better; 0.25 = random on balanced target)",
        "",
        "| Model | Train | Val | Test |",
        "|---|---:|---:|---:|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| `{name}` | {s['train_brier']:.4f} | {s['val_brier']:.4f} | {s['test_brier']:.4f} |"
        )
    lines += [
        "",
        "## ECE (15-bin equal-width; lower is better)",
        "",
        "| Model | Train | Val | Test |",
        "|---|---:|---:|---:|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| `{name}` | {s['train_ece']:.4f} | {s['val_ece']:.4f} | {s['test_ece']:.4f} |"
        )

    (OUT_DIR / "comparison_table.md").write_text("\n".join(lines))


def plot_model_comparison(summary: dict) -> None:
    names = list(summary.keys())
    metrics = ["train_roc", "val_roc", "test_roc"]
    colors = {"train_roc": "#9ca3af", "val_roc": "#f59e0b", "test_roc": "#10b981"}
    labels = {"train_roc": "train", "val_roc": "val", "test_roc": "test"}

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(metrics):
        vals = [summary[n][m] for n in names]
        ax.bar(x + (i - 1) * width, vals, width, color=colors[m], label=labels[m])
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(0.5, color="black", linestyle=":", lw=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.45, max(0.75, max(summary[n]["test_roc"] for n in names) + 0.03))
    ax.set_title("Baselines — ROC-AUC across cohorts")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_comparison.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def score_naive_market(data: dict, summary: dict) -> None:
    """Reference baseline from plan §5.4: p_hat = market_implied_prob.
    Not a trained model — pure reference. Only included where the column
    is non-null."""
    name = "naive_market"
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    def score(df):
        p = df["market_implied_prob"].to_numpy(dtype=float)
        y = df["bet_correct"].to_numpy(dtype=int)
        mask = ~np.isnan(p)
        return p[mask], y[mask], mask.sum()

    p_tr, y_tr, n_tr = score(data["train"])
    p_va, y_va, n_va = score(data["val"])
    p_te, y_te, n_te = score(data["test"])
    print(
        f"\n[{name}] using market_implied_prob directly "
        f"(train:{n_tr:,}  val:{n_va:,}  test:{n_te:,})"
    )

    metrics = {
        "model": name,
        "note": "p_hat = market_implied_prob (plan §5.4 reference)",
        "train": probability_metrics(y_tr, p_tr),
        "val": probability_metrics(y_va, p_va),
        "test": probability_metrics(y_te, p_te),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    summary[name] = {
        "fit_seconds": 0.0,
        "train_roc": metrics["train"]["roc_auc"],
        "val_roc": metrics["val"]["roc_auc"],
        "test_roc": metrics["test"]["roc_auc"],
        "train_pr": metrics["train"]["pr_auc"],
        "val_pr": metrics["val"]["pr_auc"],
        "test_pr": metrics["test"]["pr_auc"],
        "train_brier": metrics["train"]["brier"],
        "val_brier": metrics["val"]["brier"],
        "test_brier": metrics["test"]["brier"],
        "train_ece": metrics["train"]["ece_15bin"],
        "val_ece": metrics["val"]["ece_15bin"],
        "test_ece": metrics["test"]["ece_15bin"],
    }


def main() -> None:
    data = load_and_prep()
    models = make_models()
    summary: dict[str, dict] = {}

    for name, model in models.items():
        fit_score_save(name, model, data, summary)

    # Reference baseline — not a trained model
    score_naive_market(data, summary)

    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    write_comparison_table(summary)
    plot_model_comparison(summary)

    # Pretty summary table to stdout
    print("\n" + "=" * 90)
    print(
        f"{'model':<18}  {'train_roc':>10}  {'val_roc':>9}  {'test_roc':>9}  "
        f"{'test_pr':>9}  {'test_brier':>11}  {'fit':>6}"
    )
    print("-" * 90)
    for name, s in summary.items():
        print(
            f"{name:<18}  {s['train_roc']:>10.4f}  {s['val_roc']:>9.4f}  "
            f"{s['test_roc']:>9.4f}  {s['test_pr']:>9.4f}  {s['test_brier']:>11.4f}  "
            f"{s['fit_seconds']:>5.1f}s"
        )
    print("=" * 90)
    print(f"\n[done] outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
