"""
02_features.py — Document the feature taxonomy, fit the scaler, build IsoForest anomaly score.

Source-of-truth merge of:
  alex/v5_final_ml_pipeline/scripts/_common.py        (loader + cost/edge helpers)
  alex/v5_final_ml_pipeline/scripts/04_iso_forest.py  (IsoForest anomaly score as feature)

Note on feature engineering: the heavy feature engineering (70 trade/market/wallet
features) was performed upstream and the result was bundled into the consolidated
parquet shipped with the submission. This script:
  - documents the feature taxonomy that the consolidated parquet exposes
  - fits the StandardScaler used by every linear or NN model in the pipeline
  - writes an unsupervised Isolation Forest anomaly score as a diagnostic
    (curriculum: outlier detection). It is not joined into the supervised X matrix.

Run:
  python 02_features.py

Outputs:
  outputs/data/feature_taxonomy.json     groups: trade / market / wallet / microstructure
  outputs/data/scaler.joblib             StandardScaler fitted on train rows only
  outputs/data/iso_forest_scores.parquet anomaly score per (split, market_id, timestamp)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# what: pull shared paths and the project random seed
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, OUTPUTS_DIR, RANDOM_SEED  # noqa: E402

TARGET = "bet_correct"
META_COLS = ["split", "market_id", "ts_dt", "timestamp"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the parquet, split train/test, read the feature list saved by 01_data_prep."""
    # what: read everything 01_data_prep produced + the dataset
    df = pd.read_parquet(DATA_DIR / "consolidated_modeling_data.parquet")
    feature_cols = json.loads((OUTPUTS_DIR / "data" / "feature_cols.json").read_text())
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  loaded train={len(train):,} test={len(test):,} features={len(feature_cols)}")
    return train, test, feature_cols


def build_taxonomy(feature_cols: list[str]) -> dict:
    """Group features by source so the report's methodology section can describe them clearly."""
    # what: bucket each feature name into a thematic group for the methodology table
    # how: simple prefix / substring rules; the upstream naming convention makes this safe
    # why: lets the report cite "X trade-microstructure features, Y wallet features, ..."
    taxonomy: dict[str, list[str]] = {
        "wallet": [], "market": [], "trade": [], "microstructure": [], "history": [], "other": [],
    }
    for c in feature_cols:
        if c.startswith("wallet_") or c == "days_from_first_usdc_to_t":
            taxonomy["wallet"].append(c)
        elif c.startswith("market_") or "deadline" in c or "resolution" in c:
            taxonomy["market"].append(c)
        elif "kyle" in c or "spread" in c or "depth" in c or "imbalance" in c or "lambda" in c:
            taxonomy["microstructure"].append(c)
        elif "taker_" in c or "wallet_prior" in c or "history" in c:
            taxonomy["history"].append(c)
        elif c in {"price", "pre_trade_price", "pre_yes_price_corrected", "token_amount",
                   "usd_amount", "side_buy", "outcome_yes", "nonusdc_side"}:
            taxonomy["trade"].append(c)
        else:
            taxonomy["other"].append(c)
    counts = {k: len(v) for k, v in taxonomy.items()}
    print("  feature taxonomy:", counts)
    return {"counts": counts, "groups": taxonomy}


def fit_scaler(train: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    """Fit StandardScaler on the train split only (no test leakage)."""
    # what: replace any inf with NaN, then fill NaN with 0, then fit
    # why: linear and MLP models need standardised inputs; trees do not but the scaler is harmless
    # how: fit on train rows ONLY — fitting on test would leak distribution info
    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    scaler = StandardScaler().fit(X_train)
    print(f"  scaler fit on {len(X_train):,} rows × {len(feature_cols)} features")
    return scaler


def add_iso_forest_score(train: pd.DataFrame, test: pd.DataFrame,
                          feature_cols: list[str], scaler: StandardScaler) -> pd.DataFrame:
    """Train an Isolation Forest on train, score every row (train + test).

    Curriculum link: outlier detection (lecture 7). Hypothesis: unusual trades sit
    further from the joint feature distribution than retail trades, so a higher
    anomaly score should correlate with bet_correct.
    """
    # what: prepare the matrices in the same way as the supervised models
    X_train = scaler.transform(
        train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values)
    X_test = scaler.transform(
        test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values)

    # what: 200 trees, sub-sampled to 50k for speed; default contamination
    # how: IsoForest splits points randomly until isolated; faster isolation = more anomalous
    # why: pure-unsupervised signal, decoupled from the supervised target
    print("  fitting Isolation Forest (200 trees, sample 50k) ...")
    iso = IsolationForest(n_estimators=200, max_samples=min(50_000, len(X_train)),
                          contamination="auto", random_state=RANDOM_SEED, n_jobs=-1)
    iso.fit(X_train)

    # what: score_samples returns higher = more normal; we negate so higher = more anomalous
    # why: easier to interpret as "anomaly score" for the report
    train_score = -iso.score_samples(X_train)
    test_score = -iso.score_samples(X_test)
    s_min, s_max = train_score.min(), train_score.max()
    train_scaled = (train_score - s_min) / (s_max - s_min + 1e-9)
    test_scaled = np.clip((test_score - s_min) / (s_max - s_min + 1e-9), 0, 1)

    # what: report the correlation between anomaly score and target as a quick sanity check
    train_corr = float(np.corrcoef(train_scaled, train[TARGET])[0, 1])
    test_corr = float(np.corrcoef(test_scaled, test[TARGET])[0, 1])
    print(f"  anomaly-target correlation: train {train_corr:+.4f}  test {test_corr:+.4f}")

    # what: stack (split, market_id, timestamp, anomaly_score) for downstream join
    rows = []
    for split, sub, scores in [("train", train, train_scaled), ("test", test, test_scaled)]:
        rows.append(pd.DataFrame({
            "split": split,
            "market_id": sub["market_id"].astype(str).values,
            "timestamp": sub["timestamp"].values,
            "anomaly_score": scores,
        }))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    print("=" * 60)
    print("Stage 2 — Feature taxonomy, scaler, anomaly score")
    print("=" * 60)
    out_dir = OUTPUTS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # what: load + taxonomy
    train, test, feature_cols = load_data()
    taxonomy = build_taxonomy(feature_cols)
    (out_dir / "feature_taxonomy.json").write_text(json.dumps(taxonomy, indent=2))

    # what: fit + persist scaler
    scaler = fit_scaler(train, feature_cols)
    joblib.dump(scaler, out_dir / "scaler.joblib")

    # what: build anomaly score
    iso_scores = add_iso_forest_score(train, test, feature_cols, scaler)
    iso_scores.to_parquet(out_dir / "iso_forest_scores.parquet", index=False)

    print(f"\nStage 2 complete. {len(feature_cols)} features ready for modeling.")
    return 0


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    sys.exit(main())
