"""
02_baseline_logreg.py

Logistic-regression baseline on the market-cohort train/val/test frame.
Anchors every subsequent model — if LogReg barely beats random, the
features aren't carrying much signal and no amount of MLP depth will fix
that.

Pipeline:
  1. Load train/val/test parquets.
  2. Drop non-features + leaky features (per alex/notes/feature-exclusion-list.md).
  3. Encode `side` → 0/1.
  4. Winsorise `trade_value_usd` + `wallet_prior_volume_usd` at 1st/99th (train-only bounds).
  5. Median-impute + standardise (train-only fit).
  6. Fit LogisticRegression(class_weight='balanced', max_iter=2000).
  7. Score train/val/test: ROC-AUC, PR-AUC, Brier, ECE.
  8. Per-market breakdown on val + test (catches market-level performance variance).
  9. Signed coefficients dumped so we can inspect what the model actually latched onto.

Outputs under `alex/outputs/baselines/logreg/`.
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent  # .../ML/report
DATA_DIR = ROOT / "data" / "experiments"
OUT_DIR = ROOT / "alex" / "outputs" / "baselines" / "logreg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature exclusion — canonical set from alex/notes/feature-exclusion-list.md
# ---------------------------------------------------------------------------

NON_FEATURE_COLS = {
    # Class 1: identifiers / metadata
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
    # Class 2: raw cols superseded by derived features
    "size",
    "price",
    "timestamp",
    # Class 3: filter / label / benchmark / split
    "settlement_minus_trade_sec",
    "bet_correct",
    "market_implied_prob",
    "split",
    # Class 4a: leaky features
    "wallet_is_whale_in_market",  # P0-1: end-of-market p95 threshold
    "is_position_exit",  # P0-2: denominator uses current trade size
    # Class 4b: market-identifying absolute-scale (P0-8)
    "time_to_settlement_s",
    "log_time_to_settlement",
    "market_volume_so_far_usd",
    "market_vol_1h_log",
    "market_vol_24h_log",
    "market_trade_count_so_far",
    "size_x_time_to_settlement",  # interaction with raw time_to_settlement_s
    # Class 4c: temporal leak
    "wallet_prior_win_rate",  # P0-9: bet_correct cumsum on prior trades not resolved at t
}

WINSORISE_COLS = ["trade_value_usd", "wallet_prior_volume_usd"]
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width ECE. Both inputs as float arrays in [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
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


# ---------------------------------------------------------------------------
# Prep
# ---------------------------------------------------------------------------


def encode_side(df: pd.DataFrame) -> pd.DataFrame:
    if "side" in df.columns:
        df = df.copy()
        df["side"] = (df["side"].astype(str).str.upper() == "BUY").astype(int)
    return df


def winsorise(
    df: pd.DataFrame, cols: list[str], bounds: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Clip cols at 1st/99th percentile. If bounds given, apply them (for val/test)."""
    out = df.copy()
    if bounds is None:
        bounds = {}
        for c in cols:
            if c in out.columns:
                lo, hi = out[c].quantile([0.01, 0.99]).tolist()
                bounds[c] = (lo, hi)
    for c, (lo, hi) in bounds.items():
        if c in out.columns:
            out[c] = out[c].clip(lo, hi)
    return out, bounds


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def per_market_metrics(df: pd.DataFrame, p: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for q, g in df.groupby("question"):
        idx = g.index.to_numpy()
        # df was reset_index'd before calling; safe to use positional indexes
        positions = df.index.get_indexer(idx)
        p_m = p[positions]
        y_m = y[positions]
        if len(set(y_m)) < 2:
            roc = None
            pr = None
        else:
            roc = float(roc_auc_score(y_m, p_m))
            pr = float(average_precision_score(y_m, p_m))
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


def plot_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, path: Path, title: str
) -> None:
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    xs, ys, ns = [], [], []
    for b in range(10):
        mask = idx == b
        if not mask.any():
            continue
        xs.append(y_prob[mask].mean())
        ys.append(y_true[mask].mean())
        ns.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4, label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#3b82f6", label="LogReg")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(
            f"n={n:,}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7
        )
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"[load] {DATA_DIR}")
    train = pd.read_parquet(DATA_DIR / "train.parquet").reset_index(drop=True)
    val = pd.read_parquet(DATA_DIR / "val.parquet").reset_index(drop=True)
    test = pd.read_parquet(DATA_DIR / "test.parquet").reset_index(drop=True)
    for nm, df in [("train", train), ("val", val), ("test", test)]:
        print(f"  {nm}: {len(df):,} rows × {df.shape[1]} cols")

    features = [c for c in train.columns if c not in NON_FEATURE_COLS]
    print(f"[features] {len(features)} feature cols")

    train = encode_side(train)
    val = encode_side(val)
    test = encode_side(test)

    y_train = train["bet_correct"].astype(int).to_numpy()
    y_val = val["bet_correct"].astype(int).to_numpy()
    y_test = test["bet_correct"].astype(int).to_numpy()

    X_train = train[features]
    X_val = val[features]
    X_test = test[features]

    X_train_w, bounds = winsorise(X_train, WINSORISE_COLS)
    X_val_w, _ = winsorise(X_val, WINSORISE_COLS, bounds)
    X_test_w, _ = winsorise(X_test, WINSORISE_COLS, bounds)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(imputer.fit_transform(X_train_w))
    X_val_np = scaler.transform(imputer.transform(X_val_w))
    X_test_np = scaler.transform(imputer.transform(X_test_w))

    print("[fit] LogisticRegression(class_weight='balanced', max_iter=2000)")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train_np, y_train)

    p_train = model.predict_proba(X_train_np)[:, 1]
    p_val = model.predict_proba(X_val_np)[:, 1]
    p_test = model.predict_proba(X_test_np)[:, 1]

    # Aggregate metrics
    metrics = {
        "features": features,
        "n_features": len(features),
        "winsorise_bounds": {k: list(v) for k, v in bounds.items()},
        "train": probability_metrics(y_train, p_train),
        "val": probability_metrics(y_val, p_val),
        "test": probability_metrics(y_test, p_test),
    }

    # Per-market breakdowns
    per_market = {
        "val": per_market_metrics(val, p_val, y_val),
        "test": per_market_metrics(test, p_test, y_test),
    }

    # Signed coefficients (standardised scale)
    coef = model.coef_[0]
    intercept = float(model.intercept_[0])
    feature_importance = sorted(
        [
            {"feature": f, "coef_std": float(c), "abs_coef_std": float(abs(c))}
            for f, c in zip(features, coef)
        ],
        key=lambda r: r["abs_coef_std"],
        reverse=True,
    )

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT_DIR / "per_market.json").write_text(json.dumps(per_market, indent=2))
    (OUT_DIR / "feature_importance.json").write_text(
        json.dumps({"intercept": intercept, "features": feature_importance}, indent=2)
    )

    plot_calibration(
        y_val, p_val, OUT_DIR / "calibration_val.png", "LogReg — validation calibration"
    )
    plot_calibration(
        y_test, p_test, OUT_DIR / "calibration_test.png", "LogReg — test calibration"
    )

    # Summary printout
    print("\n" + "=" * 70)
    print("SUMMARY — LogReg baseline")
    print("=" * 70)
    for split in ("train", "val", "test"):
        m = metrics[split]
        print(
            f"  {split:<6}  ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}  "
            f"Brier={m['brier']:.4f}  ECE={m['ece_15bin']:.4f}  "
            f"n={m['n']:,}  bc={m['positive_rate']:.3f}"
        )

    print(f"\n[per-market test] top 5 by n:")
    for r in per_market["test"][:5]:
        roc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "  —   "
        print(
            f"  ROC={roc_str}  n={r['n']:>6,}  bc={r['bc_rate']:.3f}  "
            f"p_hat={r['p_hat_mean']:.3f}±{r['p_hat_std']:.3f}  {r['question'][:52]}"
        )

    print(f"\n[coefficients] top 10 by |coef|:")
    for r in feature_importance[:10]:
        sign = "+" if r["coef_std"] >= 0 else ""
        print(f"  {sign}{r['coef_std']:+.4f}  {r['feature']}")

    print(f"\n[done] outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
