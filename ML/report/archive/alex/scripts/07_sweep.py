"""
07_sweep.py

Multi-model sweep on idea1 feature set (~65 features). Lecture-aligned.

Models:
  1. LogReg L2 (L04)            — linear anchor
  2. LogReg L1 (L04)            — elbow / feature selection
  3. Decision Tree (L05)        — interpretable
  4. Random Forest (L05)        — bagged trees
  5. HistGradientBoosting (L05) — fast boosting
  6. PCA(20) → LogReg (L05)     — dim-reduction pipeline
  7. MLP via TF/Keras (L09)     — deep learning
  8. Isolation Forest (L07)     — UNSUPERVISED insider-trade detector

For 1-7: 5-fold GroupKFold CV on train, fit isotonic on OOF, final fit on full train,
score on test. Save metrics + feature importance.

For 8: Fit IsoForest on train features, score test. Compute correlation between
anomaly score and bet_correct (does anomaly mean informed?).

Outputs: alex/outputs/sweep_idea1/<model_name>/{metrics.json, ...}
        alex/outputs/sweep_idea1/comparison_table.md
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "sweep_idea1"
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def metrics(y_true, y_prob, label):
    return {
        f"{label}_auc": float(roc_auc_score(y_true, y_prob)),
        f"{label}_brier": float(brier_score_loss(y_true, y_prob)),
        f"{label}_ece": expected_calibration_error(y_true, y_prob),
        f"{label}_n": int(len(y_true)),
        f"{label}_pos_rate": float(y_true.mean()),
    }


def per_market_metrics(test_df, y_test, y_prob_cal, min_n=20):
    """y_test can be pandas Series or numpy array."""
    y_arr = np.asarray(y_test)
    out = {}
    mids = test_df["market_id"].values
    for mid in test_df["market_id"].unique():
        mask = mids == mid
        if mask.sum() < min_n or len(np.unique(y_arr[mask])) < 2:
            continue
        out[str(mid)] = {
            "n": int(mask.sum()),
            "pos_rate": float(y_arr[mask].mean()),
            "auc": float(roc_auc_score(y_arr[mask], y_prob_cal[mask])),
            "brier": float(brier_score_loss(y_arr[mask], y_prob_cal[mask])),
        }
    return out


# ---------------------------------------------------------------------------
# CV helper — returns OOF predictions for a given estimator factory
# ---------------------------------------------------------------------------


def cv_oof(make_estimator, X, y, groups, n_folds=N_FOLDS, scale=True):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(y), dtype=float)
    fold_aucs = []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X.iloc[tr_idx])
            X_va = scaler.transform(X.iloc[va_idx])
        else:
            X_tr = X.iloc[tr_idx].values
            X_va = X.iloc[va_idx].values
        clf = make_estimator()
        clf.fit(X_tr, y.iloc[tr_idx])
        if hasattr(clf, "predict_proba"):
            preds = clf.predict_proba(X_va)[:, 1]
        else:
            preds = clf.decision_function(X_va)
            # Convert to [0,1] via min-max within fold
            preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-9)
        oof[va_idx] = preds
        fold_aucs.append(roc_auc_score(y.iloc[va_idx], preds))
    return oof, fold_aucs


def fit_final_with_calibrator(
    make_estimator, X_train, y_train, X_test, oof, scale=True
):
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        scaler = None
        X_tr = X_train.values
        X_te = X_test.values

    clf = make_estimator()
    clf.fit(X_tr, y_train)
    if hasattr(clf, "predict_proba"):
        raw = clf.predict_proba(X_te)[:, 1]
    else:
        raw = clf.decision_function(X_te)
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof, y_train)
    cal_pred = cal.transform(raw)
    return scaler, clf, cal, raw, cal_pred


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


def evaluate_supervised(
    name,
    make_estimator,
    X_train,
    y_train,
    g_train,
    X_test,
    y_test,
    test_df,
    scale=True,
    importance_fn=None,
):
    print(f"\n[{name}] CV...")
    oof, fold_aucs = cv_oof(make_estimator, X_train, y_train, g_train, scale=scale)
    cv_oof_auc = float(roc_auc_score(y_train, oof))
    cv_brier = float(brier_score_loss(y_train, oof))
    print(f"  OOF AUC: {cv_oof_auc:.4f} (folds: {[f'{a:.3f}' for a in fold_aucs]})")

    print(f"[{name}] final fit + test scoring...")
    scaler, clf, cal, raw, cal_pred = fit_final_with_calibrator(
        make_estimator, X_train, y_train, X_test, oof, scale=scale
    )

    test_metrics_raw = metrics(y_test.values, raw, "test_raw")
    test_metrics_cal = metrics(y_test.values, cal_pred, "test_calibrated")
    pm = per_market_metrics(test_df, y_test.values, cal_pred)

    out_dir = OUT / name
    out_dir.mkdir(exist_ok=True)
    summary = {
        "model": name,
        "cv_oof_auc": cv_oof_auc,
        "cv_oof_brier": cv_brier,
        "cv_fold_aucs": [float(a) for a in fold_aucs],
        "cv_fold_auc_mean": float(np.mean(fold_aucs)),
        "cv_fold_auc_std": float(np.std(fold_aucs)),
        **test_metrics_raw,
        **test_metrics_cal,
        "per_market_auc_min": float(min(v["auc"] for v in pm.values())) if pm else None,
        "per_market_auc_max": float(max(v["auc"] for v in pm.values())) if pm else None,
        "per_market_auc_mean": float(np.mean([v["auc"] for v in pm.values()]))
        if pm
        else None,
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_market_test.json").write_text(json.dumps(pm, indent=2))

    # Importance
    if importance_fn is not None:
        imp = importance_fn(clf, list(X_train.columns))
        (out_dir / "feature_importance.json").write_text(json.dumps(imp, indent=2))

    print(f"  test AUC (cal): {test_metrics_cal['test_calibrated_auc']:.4f}")
    print(f"  test Brier (cal): {test_metrics_cal['test_calibrated_brier']:.4f}")
    print(
        f"  per-market AUC: [{summary['per_market_auc_min']:.3f}, {summary['per_market_auc_max']:.3f}]"
    )
    return summary


# ---------------------------------------------------------------------------
# Importance helpers
# ---------------------------------------------------------------------------


def logreg_importance(clf, feat_names):
    coefs = clf.coef_[0].tolist()
    return dict(sorted(zip(feat_names, coefs), key=lambda kv: abs(kv[1]), reverse=True))


def tree_importance(clf, feat_names):
    imps = clf.feature_importances_.tolist()
    return dict(sorted(zip(feat_names, imps), key=lambda kv: kv[1], reverse=True))


def pca_logreg_importance(pipe, feat_names):
    # Pipeline has "pca" then "lr". Use LR coefs in PCA-space; not directly interpretable
    # in original feature space, but report top-K PCA components by |coef|.
    lr = pipe.named_steps["lr"]
    return {f"pc_{i}": float(c) for i, c in enumerate(lr.coef_[0])}


# ---------------------------------------------------------------------------
# MLP — TF/Keras
# ---------------------------------------------------------------------------


def build_keras_mlp(input_dim):
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(RANDOM_SEED)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, kernel_initializer="glorot_uniform"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("selu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, kernel_initializer="glorot_uniform"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("selu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


def evaluate_keras_mlp(X_train, y_train, g_train, X_test, y_test, test_df):
    import tensorflow as tf
    from tensorflow import keras

    name = "mlp_keras"
    print(f"\n[{name}] CV...")
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(y_train), dtype=float)
    fold_aucs = []
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, g_train)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train.iloc[tr_idx])
        X_va = scaler.transform(X_train.iloc[va_idx])

        # Inner chronological split for early stopping (last 10% by row order)
        n_inner_val = max(int(0.1 * len(X_tr)), 1000)
        X_tr_inner = X_tr[:-n_inner_val]
        y_tr_inner = y_train.iloc[tr_idx].values[:-n_inner_val]
        X_inner_val = X_tr[-n_inner_val:]
        y_inner_val = y_train.iloc[tr_idx].values[-n_inner_val:]

        model = build_keras_mlp(X_train.shape[1])
        es = keras.callbacks.EarlyStopping(
            monitor="val_AUC", mode="max", patience=3, restore_best_weights=True
        )
        model.fit(
            X_tr_inner,
            y_tr_inner,
            validation_data=(X_inner_val, y_inner_val),
            epochs=20,
            batch_size=4096,
            verbose=0,
            callbacks=[es],
        )
        preds = model.predict(X_va, batch_size=4096, verbose=0).ravel()
        oof[va_idx] = preds
        fold_auc = roc_auc_score(y_train.iloc[va_idx], preds)
        fold_aucs.append(fold_auc)
        print(f"  fold {fold_idx + 1}: AUC = {fold_auc:.4f}")
        keras.backend.clear_session()

    cv_oof_auc = float(roc_auc_score(y_train, oof))
    print(f"  OOF AUC: {cv_oof_auc:.4f}")

    print(f"[{name}] final fit + test scoring...")
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)
    n_inner_val = max(int(0.1 * len(X_tr_scaled)), 1000)
    final_model = build_keras_mlp(X_train.shape[1])
    es = keras.callbacks.EarlyStopping(
        monitor="val_AUC", mode="max", patience=3, restore_best_weights=True
    )
    final_model.fit(
        X_tr_scaled[:-n_inner_val],
        y_train.values[:-n_inner_val],
        validation_data=(X_tr_scaled[-n_inner_val:], y_train.values[-n_inner_val:]),
        epochs=20,
        batch_size=4096,
        verbose=0,
        callbacks=[es],
    )
    raw = final_model.predict(X_te_scaled, batch_size=4096, verbose=0).ravel()
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof, y_train)
    cal_pred = cal.transform(raw)

    pm = per_market_metrics(test_df, y_test.values, cal_pred)
    test_metrics_raw = metrics(y_test.values, raw, "test_raw")
    test_metrics_cal = metrics(y_test.values, cal_pred, "test_calibrated")

    out_dir = OUT / name
    out_dir.mkdir(exist_ok=True)
    summary = {
        "model": name,
        "cv_oof_auc": cv_oof_auc,
        "cv_oof_brier": float(brier_score_loss(y_train, oof)),
        "cv_fold_aucs": [float(a) for a in fold_aucs],
        "cv_fold_auc_mean": float(np.mean(fold_aucs)),
        "cv_fold_auc_std": float(np.std(fold_aucs)),
        **test_metrics_raw,
        **test_metrics_cal,
        "per_market_auc_min": float(min(v["auc"] for v in pm.values())) if pm else None,
        "per_market_auc_max": float(max(v["auc"] for v in pm.values())) if pm else None,
        "per_market_auc_mean": float(np.mean([v["auc"] for v in pm.values()]))
        if pm
        else None,
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "architecture": "Dense(64)-BN-SELU-Dropout(0.3)-Dense(32)-BN-SELU-Dropout(0.3)-Dense(1)-sigmoid",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_market_test.json").write_text(json.dumps(pm, indent=2))
    print(f"  test AUC (cal): {test_metrics_cal['test_calibrated_auc']:.4f}")
    return summary


# ---------------------------------------------------------------------------
# Isolation Forest — UNSUPERVISED insider detector
# ---------------------------------------------------------------------------


def evaluate_isolation_forest(X_train, y_train, X_test, y_test, test_df):
    name = "iso_forest"
    print(f"\n[{name}] fitting...")
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

    # Anomaly score: higher = more anomalous
    train_score = -iso.score_samples(X_tr)  # negative because lower = more abnormal
    test_score = -iso.score_samples(X_te)

    # Min-max scale to [0,1] using train bounds
    s_min, s_max = train_score.min(), train_score.max()
    train_scaled = (train_score - s_min) / (s_max - s_min + 1e-9)
    test_scaled = (test_score - s_min) / (s_max - s_min + 1e-9)
    test_scaled = np.clip(test_scaled, 0, 1)

    # Does anomaly correlate with bet_correct?
    train_corr = float(np.corrcoef(train_scaled, y_train)[0, 1])
    test_corr = float(np.corrcoef(test_scaled, y_test)[0, 1])

    # Top-k precision: among top-k anomaly trades, what fraction won?
    test_metrics_block = {}
    for k_pct in [0.01, 0.05, 0.10]:
        k = int(len(test_scaled) * k_pct)
        top_idx = np.argsort(test_scaled)[-k:]
        prec = float(y_test.iloc[top_idx].mean())
        test_metrics_block[f"top_{int(k_pct * 100)}pct_precision"] = prec
        test_metrics_block[f"top_{int(k_pct * 100)}pct_n"] = k

    # AUC of using anomaly score directly as predictor
    test_score_auc = float(roc_auc_score(y_test, test_scaled))

    out_dir = OUT / name
    out_dir.mkdir(exist_ok=True)
    summary = {
        "model": name,
        "anomaly_score_target_corr_train": train_corr,
        "anomaly_score_target_corr_test": test_corr,
        "anomaly_score_test_auc": test_score_auc,
        **test_metrics_block,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    # Save test anomaly scores for cross-checking with supervised
    pd.DataFrame(
        {
            "market_id": test_df["market_id"].values,
            "timestamp": test_df["timestamp"].values,
            "anomaly_score": test_scaled,
            "bet_correct": y_test.values,
        }
    ).to_parquet(out_dir / "test_anomaly_scores.parquet", index=False)

    print(f"  anomaly-target corr (train): {train_corr:.4f}")
    print(f"  anomaly-target corr (test):  {test_corr:.4f}")
    print(f"  anomaly score → test AUC:    {test_score_auc:.4f}")
    print(
        f"  top-1% precision:            {test_metrics_block['top_1pct_precision']:.3f}"
    )
    print(
        f"  top-5% precision:            {test_metrics_block['top_5pct_precision']:.3f}"
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("idea1 sweep: 7 supervised models + isolation forest")
    print("=" * 60)
    train = pd.read_parquet(DATA / "train_features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    feature_cols = json.loads((DATA / "feature_cols.json").read_text())
    print(f"train: {train.shape}, test: {test.shape}, n_features: {len(feature_cols)}")

    X_train = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train["bet_correct"].astype(int)
    g_train = train["market_id"]
    X_test = test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test["bet_correct"].astype(int)
    print(f"class balance train: {y_train.mean():.3f}, test: {y_test.mean():.3f}\n")

    summaries = []

    # 1. LogReg L2
    summaries.append(
        evaluate_supervised(
            "logreg_l2",
            lambda: LogisticRegression(
                C=1.0,
                penalty="l2",
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_SEED,
            ),
            X_train,
            y_train,
            g_train,
            X_test,
            y_test,
            test,
            scale=True,
            importance_fn=logreg_importance,
        )
    )

    # 2. LogReg L1 (sparse for elbow)
    summaries.append(
        evaluate_supervised(
            "logreg_l1",
            lambda: LogisticRegression(
                C=0.1,
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_SEED,
            ),
            X_train,
            y_train,
            g_train,
            X_test,
            y_test,
            test,
            scale=True,
            importance_fn=logreg_importance,
        )
    )

    # 3. Decision Tree
    summaries.append(
        evaluate_supervised(
            "decision_tree",
            lambda: DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=200,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            X_train,
            y_train,
            g_train,
            X_test,
            y_test,
            test,
            scale=False,
            importance_fn=tree_importance,
        )
    )

    # 4. Random Forest
    summaries.append(
        evaluate_supervised(
            "random_forest",
            lambda: RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=200,
                n_jobs=-1,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            X_train,
            y_train,
            g_train,
            X_test,
            y_test,
            test,
            scale=False,
            importance_fn=tree_importance,
        )
    )

    # 5. HistGBM
    summaries.append(
        evaluate_supervised(
            "hist_gbm",
            lambda: HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=8,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            X_train,
            y_train,
            g_train,
            X_test,
            y_test,
            test,
            scale=False,
            importance_fn=None,  # HistGBM doesn't expose feature_importances_ directly
        )
    )

    # 6. PCA → LogReg
    summaries.append(
        evaluate_supervised(
            "pca_logreg",
            lambda: Pipeline(
                [
                    ("pca", PCA(n_components=20, random_state=RANDOM_SEED)),
                    (
                        "lr",
                        LogisticRegression(
                            C=1.0,
                            penalty="l2",
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            X_train,
            y_train,
            g_train,
            X_test,
            y_test,
            test,
            scale=True,
            importance_fn=pca_logreg_importance,
        )
    )

    # 7. MLP via TF/Keras
    try:
        summaries.append(
            evaluate_keras_mlp(X_train, y_train, g_train, X_test, y_test, test)
        )
    except Exception as e:
        print(f"[mlp_keras] FAILED: {e}")
        summaries.append({"model": "mlp_keras", "error": str(e)})

    # 8. Isolation Forest (unsupervised)
    summaries.append(evaluate_isolation_forest(X_train, y_train, X_test, y_test, test))

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    rows = []
    for s in summaries:
        if "error" in s:
            rows.append({"model": s["model"], "ERROR": s["error"]})
            continue
        if s["model"] == "iso_forest":
            rows.append(
                {
                    "model": s["model"],
                    "test_auc_cal": f"{s['anomaly_score_test_auc']:.3f}",
                    "top1pct_prec": f"{s['top_1pct_precision']:.3f}",
                    "top5pct_prec": f"{s['top_5pct_precision']:.3f}",
                    "anom_target_corr": f"{s['anomaly_score_target_corr_test']:+.3f}",
                }
            )
        else:
            rows.append(
                {
                    "model": s["model"],
                    "cv_oof_auc": f"{s['cv_oof_auc']:.3f}",
                    "test_auc_cal": f"{s['test_calibrated_auc']:.3f}",
                    "test_brier_cal": f"{s['test_calibrated_brier']:.3f}",
                    "test_ece_cal": f"{s['test_calibrated_ece']:.3f}",
                    "per_market_auc_range": f"[{s['per_market_auc_min']:.2f}, {s['per_market_auc_max']:.2f}]",
                }
            )

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(OUT / "comparison_table.csv", index=False)
    (OUT / "comparison_table.md").write_text(summary_df.to_markdown(index=False))
    (OUT / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"\noutputs: {OUT}")


if __name__ == "__main__":
    main()
