"""Baseline models on the Iran-strike labeled dataset.

Four models:
  1. Naive market — `p_hat = market_implied_prob`. The efficient-market null.
  2. Logistic regression (plain + L1/Lasso) — linear baseline + sparse ranking.
  3. Random Forest — non-linear baseline + feature importance.
  4. Isolation Forest — unsupervised anomaly baseline.

All trained on the `train` bucket, evaluated on `val` and `test` buckets.
Causal: `price` / `market_implied_prob` is excluded from MLP features (we drop
it here too), but we use it as the benchmark for the gap-based metrics.

Outputs:
  data/baseline_outputs/
    metrics.csv             — per-model PR-AUC, ROC-AUC, accuracy, Brier, log-loss
    lasso_coefs.csv         — sorted L1 coefficients (feature ranking)
    rf_feature_importance.csv — sorted RF importances
    isolation_scores_quantiles.csv — IF anomaly-score distribution
    gap_evaluation.csv      — gap = p_hat - market_implied_prob, ROC-AUC for correctness

Usage:
  python scripts/baselines.py
    [--labeled data/iran_strike_labeled.parquet]
    [--out data/baseline_outputs/]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]

# Features we feed to the models (everything numeric that isn't target/meta)
DROP_COLS = {
    # identifiers / meta
    "timestamp",
    "block_number",
    "transaction_hash",
    "condition_id",
    "maker",
    "taker",
    "nonusdc_side",
    "resolved",
    "winner_token",
    "settlement_ts",
    "bucket",
    "wallet",
    "side",
    "question",
    # target
    "bet_correct",
    # `market_implied_prob` stays out — it is the BENCHMARK we compute the gap
    # against (gap = p_hat − market_implied_prob). `price` is kept as a feature
    # because without it the classifier can't distinguish YES-leaning markets
    # from NO-leaning ones and falls back to the training outcome prior →
    # catastrophic test-set inversion on RF (ROC 0.15). Accepting a slight dent
    # to the "p_hat independent of market" framing is worth a model that
    # actually generalises across markets. Documented in design-decisions.md.
    "market_implied_prob",
    # raw integer positions (we keep log-transforms only)
    "usd_amount",
    "token_amount",
    "wallet_t1_position_before",
    "wallet_t2_position_before",
    "wallet_t1_cumvol_in_market",
    "wallet_t2_cumvol_in_market",
    "wallet_position_same_token_before",
    "wallet_total_cumvol_in_market",
    "market_cumvol",
    "market_vol_1h",
    "market_vol_24h",
    "market_trades_1h",
    "wallet_prior_trades",  # retain the _log variant
}

# Features the EDA flagged for clipping or dropping
CLIP_COLS = {"size_vs_market_cumvol_pct": (0.0, None)}  # clip to (0, 99th pct)


def load_and_split(
    labeled_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(labeled_path)
    print(f"loaded {len(df):,} rows × {len(df.columns)} cols")

    # Apply EDA-driven preprocessing
    for col, (lo, hi) in CLIP_COLS.items():
        if col in df.columns:
            hi = hi if hi is not None else df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lo, upper=hi)
            print(f"  clipped {col} to [{lo}, {hi:.3f}]")

    tr = df[df["bucket"] == "train"].reset_index(drop=True)
    va = df[df["bucket"] == "val"].reset_index(drop=True)
    te = df[df["bucket"] == "test"].reset_index(drop=True)
    print(f"  train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    return tr, va, te


def xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = df["bet_correct"].astype(int).values
    return X, y


def report_metrics(name: str, y: np.ndarray, proba: np.ndarray) -> dict:
    pr_auc = average_precision_score(y, proba)
    roc_auc = roc_auc_score(y, proba)
    brier = brier_score_loss(y, proba)
    try:
        ll = log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))
    except Exception:
        ll = np.nan
    acc = ((proba >= 0.5).astype(int) == y).mean()
    return {
        "model": name,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier": round(brier, 4),
        "log_loss": round(ll, 4),
        "accuracy": round(acc, 4),
    }


def naive_market(df: pd.DataFrame) -> np.ndarray:
    """Return p_hat(correct) = market_implied_prob (efficient-market null)."""
    return df["market_implied_prob"].values


def run_logreg(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    X_others: dict[str, np.ndarray],
    feature_cols: list[str],
    penalty: str = "l2",
) -> tuple[dict[str, np.ndarray], LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(
        penalty=penalty,
        C=1.0,
        solver=solver,
        max_iter=500,
        class_weight="balanced",
        n_jobs=None,
    )
    clf.fit(Xtr_s, ytr)
    out = {}
    for bucket, X in X_others.items():
        Xs = scaler.transform(X)
        out[bucket] = clf.predict_proba(Xs)[:, 1]
    return out, clf, scaler


def run_rf(
    Xtr: np.ndarray, ytr: np.ndarray, X_others: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], RandomForestClassifier]:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(Xtr, ytr)
    out = {b: clf.predict_proba(X)[:, 1] for b, X in X_others.items()}
    return out, clf


def run_isolation_forest(
    Xtr: np.ndarray, X_others: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], IsolationForest]:
    # Unsupervised — note y not used
    clf = IsolationForest(
        n_estimators=200, contamination=0.1, n_jobs=-1, random_state=42
    )
    clf.fit(Xtr)

    # Higher score = more normal. Convert to anomaly score 0..1.
    def anomaly_score(X: np.ndarray) -> np.ndarray:
        s = -clf.score_samples(X)  # higher = more anomalous
        # Normalize to 0..1 by train-set quantiles
        lo, hi = np.quantile(-clf.score_samples(Xtr), [0.05, 0.95])
        return np.clip((s - lo) / max(hi - lo, 1e-9), 0, 1)

    out = {b: anomaly_score(X) for b, X in X_others.items()}
    return out, clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labeled", default=str(ROOT / "data" / "iran_strike_labeled.parquet")
    )
    ap.add_argument("--out", default=str(ROOT / "data" / "baseline_outputs"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va, te = load_and_split(Path(args.labeled))

    # Determine feature set — all numeric except drops
    numeric_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in DROP_COLS]
    print(f"\n{len(feature_cols)} features for models:")
    for c in feature_cols:
        print(f"  - {c}")

    Xtr, ytr = xy(tr, feature_cols)
    Xva, yva = xy(va, feature_cols)
    Xte, yte = xy(te, feature_cols)
    others = {"val": Xva, "test": Xte}

    all_metrics: list[dict] = []

    # ---- 1. Naive market baseline ----
    print("\n[1/4] naive market baseline (p_hat = market_implied_prob)")
    for bucket, subdf, y in [("val", va, yva), ("test", te, yte)]:
        proba = naive_market(subdf)
        m = report_metrics(f"naive_market @ {bucket}", y, proba)
        all_metrics.append(m)
        print(f"  {bucket}: {m}")

    # ---- 2. Logistic regression (L2) ----
    print("\n[2/4] logistic regression (L2)")
    pred_logreg, lr_clf, scaler = run_logreg(
        Xtr, ytr, others, feature_cols, penalty="l2"
    )
    for bucket, y in [("val", yva), ("test", yte)]:
        m = report_metrics(f"logreg_l2 @ {bucket}", y, pred_logreg[bucket])
        all_metrics.append(m)
        print(f"  {bucket}: {m}")

    # ---- 2b. Logistic regression (L1/Lasso) for feature ranking ----
    print("\n[2b/4] logistic regression (L1)")
    pred_l1, l1_clf, _ = run_logreg(Xtr, ytr, others, feature_cols, penalty="l1")
    for bucket, y in [("val", yva), ("test", yte)]:
        m = report_metrics(f"logreg_l1 @ {bucket}", y, pred_l1[bucket])
        all_metrics.append(m)
        print(f"  {bucket}: {m}")
    lasso_coefs = pd.DataFrame(
        {"feature": feature_cols, "coef": l1_clf.coef_.ravel()}
    ).sort_values("coef", key=np.abs, ascending=False)
    lasso_coefs.to_csv(out_dir / "lasso_coefs.csv", index=False)
    print(f"  top 10 L1 coefs (by |coef|):")
    print(lasso_coefs.head(10).to_string(index=False))
    n_nonzero = (lasso_coefs["coef"] != 0).sum()
    print(f"  non-zero features: {n_nonzero}/{len(feature_cols)}")

    # ---- 3. Random Forest ----
    print("\n[3/4] random forest")
    pred_rf, rf_clf = run_rf(Xtr, ytr, others)
    for bucket, y in [("val", yva), ("test", yte)]:
        m = report_metrics(f"random_forest @ {bucket}", y, pred_rf[bucket])
        all_metrics.append(m)
        print(f"  {bucket}: {m}")
    rf_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": rf_clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    rf_imp.to_csv(out_dir / "rf_feature_importance.csv", index=False)
    print(f"  top 10 RF importances:")
    print(rf_imp.head(10).to_string(index=False))

    # ---- 4. Isolation Forest (unsupervised anomaly) ----
    print("\n[4/4] isolation forest (unsupervised)")
    pred_if, if_clf = run_isolation_forest(Xtr, others)
    for bucket, y in [("val", yva), ("test", yte)]:
        # Isolation Forest score ≠ P(correct) — but report ROC-AUC vs label anyway to
        # see if anomalous trades are systematically more/less correct.
        m = report_metrics(f"isolation_forest @ {bucket}", y, pred_if[bucket])
        all_metrics.append(m)
        print(f"  {bucket}: {m}")
    q = pd.Series(pred_if["val"]).quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    pd.Series(q).to_csv(out_dir / "isolation_scores_quantiles.csv")

    # ---- 5. Gap evaluation (p_hat - market_implied_prob) ----
    print("\n[5] gap evaluation: (p_hat − market_implied_prob) vs bet_correct")
    gap_rows = []
    for name, preds in [
        ("logreg_l2", pred_logreg),
        ("logreg_l1", pred_l1),
        ("random_forest", pred_rf),
    ]:
        for bucket, subdf, y in [("val", va, yva), ("test", te, yte)]:
            gap = preds[bucket] - subdf["market_implied_prob"].values
            try:
                # Does the gap itself predict correctness?
                gap_auc = roc_auc_score(y, gap)
            except Exception:
                gap_auc = float("nan")
            mean_gap = gap.mean()
            mean_abs_gap = np.abs(gap).mean()
            gap_rows.append(
                {
                    "model": name,
                    "bucket": bucket,
                    "mean_gap": round(mean_gap, 4),
                    "mean_abs_gap": round(mean_abs_gap, 4),
                    "gap_auc": round(gap_auc, 4),
                }
            )
            print(
                f"  {name} @ {bucket}: mean_gap={mean_gap:+.4f}  |gap|={mean_abs_gap:.4f}  gap_auc={gap_auc:.4f}"
            )
    pd.DataFrame(gap_rows).to_csv(out_dir / "gap_evaluation.csv", index=False)

    # ---- Save metrics ----
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    print(f"\nall outputs in {out_dir}/")
    print("\nsummary (val only):")
    print(metrics_df[metrics_df["model"].str.contains("@ val")].to_string(index=False))


if __name__ == "__main__":
    main()
