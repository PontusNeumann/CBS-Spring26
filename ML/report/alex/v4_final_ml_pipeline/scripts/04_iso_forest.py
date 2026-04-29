"""
09_iso_forest.py

Standalone Isolation Forest run on v3 feature set. Was queued in 07_sweep.py
behind the hung MLP — running it independently here so we get the unsupervised
result without waiting for the full sweep.

Hypothesis: insider/informed trades are statistically anomalous in the trade
feature space. IsoForest assigns higher anomaly scores to outliers; if those
correlate with `bet_correct`, we have a parallel insider-detection signal that
doesn't depend on the supervised target.

Outputs:
  alex/outputs/sweep_idea1/iso_forest/
    metrics.json
    test_anomaly_scores.parquet  (anomaly score + bet_correct for cross-checks)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "sweep_idea1" / "iso_forest"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def main():
    print("=" * 60)
    print("Isolation Forest on v3 features")
    print("=" * 60)
    train = pd.read_parquet(DATA / "train_features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    feature_cols = json.loads((DATA / "feature_cols.json").read_text())
    print(f"train: {train.shape}, test: {test.shape}, n_features: {len(feature_cols)}")

    X_train = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train["bet_correct"].astype(int)
    X_test = test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test["bet_correct"].astype(int)

    print("[fit] standardising and fitting IsoForest...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    iso = IsolationForest(
        n_estimators=200,
        max_samples=min(50000, len(X_tr)),
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso.fit(X_tr)

    print("[score] computing anomaly scores...")
    train_score = -iso.score_samples(X_tr)
    test_score = -iso.score_samples(X_te)

    s_min, s_max = train_score.min(), train_score.max()
    train_scaled = (train_score - s_min) / (s_max - s_min + 1e-9)
    test_scaled = np.clip((test_score - s_min) / (s_max - s_min + 1e-9), 0, 1)

    train_corr = float(np.corrcoef(train_scaled, y_train)[0, 1])
    test_corr = float(np.corrcoef(test_scaled, y_test)[0, 1])
    test_score_auc = float(roc_auc_score(y_test, test_scaled))

    test_metrics_block = {}
    for k_pct in [0.001, 0.005, 0.01, 0.05, 0.10]:
        k = max(1, int(len(test_scaled) * k_pct))
        top_idx = np.argsort(test_scaled)[-k:]
        prec = float(y_test.iloc[top_idx].mean())
        test_metrics_block[f"top_{k_pct * 100:g}pct_precision"] = prec
        test_metrics_block[f"top_{k_pct * 100:g}pct_n"] = k

    # Per-market anomaly distribution
    test_with = test[["market_id"]].copy()
    test_with["anomaly_score"] = test_scaled
    test_with["bet_correct"] = y_test.values
    per_market = (
        test_with.groupby("market_id")
        .agg(
            n=("anomaly_score", "size"),
            mean_anom=("anomaly_score", "mean"),
            top1pct_winrate=(
                "bet_correct",
                lambda y: float(
                    test_with.loc[y.index]
                    .nlargest(max(1, int(len(y) * 0.01)), "anomaly_score")[
                        "bet_correct"
                    ]
                    .mean()
                ),
            ),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    summary = {
        "model": "iso_forest",
        "anomaly_score_target_corr_train": train_corr,
        "anomaly_score_target_corr_test": test_corr,
        "anomaly_score_test_auc": test_score_auc,
        **test_metrics_block,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "per_market": per_market,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    test_with.to_parquet(OUT / "test_anomaly_scores.parquet", index=False)

    print("=" * 60)
    print(f"DONE — outputs in {OUT}")
    print(f"  anomaly→target corr (train): {train_corr:+.4f}")
    print(f"  anomaly→target corr (test):  {test_corr:+.4f}")
    print(f"  anomaly score → test AUC:    {test_score_auc:.4f}")
    for k in [0.1, 1, 5, 10]:
        key = f"top_{k}pct_precision"
        if key in test_metrics_block:
            n = test_metrics_block[f"top_{k}pct_n"]
            print(f"  top-{k}% precision (n={n}): {test_metrics_block[key]:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
