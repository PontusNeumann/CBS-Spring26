"""
03_train_models.py — Train and compare seven supervised models with GroupKFold CV.

Source-of-truth merge of:
  alex/v5_final_ml_pipeline/scripts/03_sweep.py            (the model sweep harness)
  alex/v5_final_ml_pipeline/scripts/03c_mlp_sklearn_only.py (sklearn MLP swap)
  alex/v5_final_ml_pipeline/scripts/_backtest_worker.py   (raw-pred saving for backtest)
  alex/v5_final_ml_pipeline/scripts/08_complexity_benchmark.py (filled in here — was skeleton)

Models (curriculum mapping in parentheses):
  - logreg_l2       Logistic regression, L2  (Lecture 4)
  - logreg_l1       Logistic regression, L1  (Lecture 4)
  - decision_tree   Decision Tree            (Lecture 5)
  - random_forest   Random Forest            (Lecture 5)
  - hist_gbm        Histogram Gradient Boosting (Lecture 5)
  - lightgbm        LightGBM (alt boosting library)
  - pca_logreg      PCA(K=elbow) -> LogReg   (Lectures 5 + 4)
  - mlp_sklearn     Multi-layer perceptron   (Lecture 9)

Cross-validation: GroupKFold(5) with group=market_id so no market spans train and val.
Calibration: isotonic refit on out-of-fold predictions (handed off to 04_calibration.py).
Complexity: fit time, predict time per 1k rows, parameter count -> required by guidelines.

Run:
  python 03_train_models.py

Outputs:
  outputs/models/<name>/metrics.json           AUC, Brier, ECE on calibrated test preds
  outputs/models/<name>/preds_test.npz         raw + cal test predictions (for 05_backtest)
  outputs/models/<name>/preds_oof.npy          OOF predictions (for 04_calibration)
  outputs/metrics/comparison.csv               headline table for the report
  outputs/metrics/complexity.csv               fit/predict time + parameter count
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# what: shared paths/seeds
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, OUTPUTS_DIR, RANDOM_SEED, N_FOLDS  # noqa: E402

TARGET = "bet_correct"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE = weighted mean | empirical_pos_rate - mean_predicted_prob | per bin."""
    # what: bucket predictions into n_bins, compare per-bucket avg-prob to per-bucket actual rate
    # why: AUC ignores calibration; ECE captures whether 0.8 really means "80% chance"
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def metric_block(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute the standard scoring block for one prediction array."""
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }


def load_xy() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Load data and the canonical feature list saved by 01_data_prep."""
    # what: read the consolidated parquet + the feature list saved by 01_data_prep
    df = pd.read_parquet(DATA_DIR / "consolidated_modeling_data.parquet")
    feature_cols = json.loads((OUTPUTS_DIR / "data" / "feature_cols.json").read_text())

    # what: replace inf with nan, then nan with 0 — same convention used end-to-end
    # why: tree models tolerate nan but linear/MLP need finite floats
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)
    g_train = train["market_id"]
    print(f"  train={len(X_train):,}  test={len(X_test):,}  features={len(feature_cols)}")
    return X_train, y_train, g_train, X_test, y_test, feature_cols, test


# ----------------------------------------------------------------------------
# Cross-validation harness
# ----------------------------------------------------------------------------

def cv_oof(make_estimator, X: pd.DataFrame, y: pd.Series, groups: pd.Series, scale: bool) -> tuple[np.ndarray, list[float]]:
    """Run 5-fold GroupKFold CV, return out-of-fold predictions + per-fold AUCs."""
    # what: predefined splitter; group=market_id keeps each market in one fold only
    # why: random split would let the same market appear in both train and val of one fold => leakage
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        # what: refit scaler PER FOLD on training rows only (no test leakage)
        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X.iloc[tr_idx])
            X_va = scaler.transform(X.iloc[va_idx])
        else:
            X_tr = X.iloc[tr_idx].values
            X_va = X.iloc[va_idx].values
        clf = make_estimator()
        clf.fit(X_tr, y.iloc[tr_idx])
        # what: get probabilities; if model has only decision_function, min-max it within fold
        if hasattr(clf, "predict_proba"):
            preds = clf.predict_proba(X_va)[:, 1]
        else:
            raw = clf.decision_function(X_va)
            preds = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        oof[va_idx] = preds
        fold_aucs.append(float(roc_auc_score(y.iloc[va_idx], preds)))
        print(f"    fold {fold_idx}/{N_FOLDS}: val AUC={fold_aucs[-1]:.4f}")
    return oof, fold_aucs


def fit_final(make_estimator, X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, scale: bool):
    """Fit one final model on the FULL train set, return raw test predictions + the fitted clf."""
    # what: fit one final model on all training rows for the headline test scoring
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        scaler, X_tr, X_te = None, X_train.values, X_test.values
    clf = make_estimator()
    clf.fit(X_tr, y_train)
    if hasattr(clf, "predict_proba"):
        raw = clf.predict_proba(X_te)[:, 1]
    else:
        rawd = clf.decision_function(X_te)
        raw = (rawd - rawd.min()) / (rawd.max() - rawd.min() + 1e-9)
    return clf, scaler, raw


def evaluate_model(name: str, make_estimator, X_train, y_train, g_train, X_test, y_test,
                    scale: bool) -> dict:
    """End-to-end: CV + final fit + save preds + return summary."""
    print(f"\n[{name}]")
    out_dir = OUTPUTS_DIR / "models" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # what: 5-fold OOF for downstream calibration
    oof, fold_aucs = cv_oof(make_estimator, X_train, y_train, g_train, scale=scale)
    cv_oof_auc = float(roc_auc_score(y_train, oof))
    print(f"  OOF AUC = {cv_oof_auc:.4f}  (folds {[f'{a:.3f}' for a in fold_aucs]})")
    np.save(out_dir / "preds_oof.npy", oof.astype("float32"))

    # what: full-train fit + test scoring (raw probabilities only; calibration in 04_)
    clf, _, raw = fit_final(make_estimator, X_train, y_train, X_test, scale=scale)
    np.savez_compressed(out_dir / "preds_test.npz", raw=raw.astype("float32"))
    test_metrics = metric_block(y_test.values, raw)

    # what: assemble + persist summary
    summary = {
        "model": name,
        "cv_oof_auc": cv_oof_auc,
        "cv_fold_aucs": fold_aucs,
        "cv_fold_mean": float(np.mean(fold_aucs)),
        "cv_fold_std": float(np.std(fold_aucs)),
        "test_raw": test_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"  test AUC (raw) = {test_metrics['auc']:.4f}  Brier = {test_metrics['brier']:.4f}")
    return summary


# ----------------------------------------------------------------------------
# PCA elbow helper for the PCA -> LogReg pipeline
# ----------------------------------------------------------------------------

def pca_elbow_k(X_train: pd.DataFrame, max_components: int = 50) -> int:
    """Pick K via the geometric-elbow method on the cumulative variance curve."""
    # what: standardise, fit PCA with k_max components, find the elbow
    # why: principled K choice (no magic 0.95 threshold) — perpendicular distance from chord
    scaler = StandardScaler()
    X = scaler.fit_transform(X_train)
    k_max = min(max_components, X.shape[1])
    pca = PCA(n_components=k_max, random_state=RANDOM_SEED).fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    xs = np.arange(1, k_max + 1, dtype=float)
    p1, p2 = np.array([xs[0], cumvar[0]]), np.array([xs[-1], cumvar[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    points = np.column_stack([xs, cumvar])
    # how: perpendicular distance from each (k, cumvar[k]) to the chord between endpoints
    dists = np.abs(np.cross(line_vec, points - p1)) / line_len
    return int(max(2, xs[int(np.argmax(dists))]))


# ----------------------------------------------------------------------------
# Complexity benchmark (REQUIRED by guidelines: model complexity vs baseline)
# ----------------------------------------------------------------------------

def complexity_proxy(model) -> int:
    """Return a parameter / size proxy for the fitted model."""
    # what: each model family stores complexity differently; report a number that scales sensibly
    # why: methodology section needs a concrete "size" number to compare 200-tree RF vs 64-unit MLP
    if hasattr(model, "coef_"):                      # logistic regression
        return int(model.coef_.size)
    if hasattr(model, "estimators_"):                # bagged trees / forests
        return int(sum(t.tree_.node_count for t in model.estimators_))
    if hasattr(model, "_predictors"):                # HistGBM
        return int(sum(p[0].nodes.size for p in model._predictors))
    if hasattr(model, "tree_"):                      # single decision tree
        return int(model.tree_.node_count)
    if hasattr(model, "coefs_"):                     # MLPClassifier
        return int(sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_))
    if hasattr(model, "booster_"):                   # LightGBM
        return int(model.booster_.num_trees())
    return -1


def benchmark_complexity(make_estimator, X_train, y_train, X_test, scale: bool, repeats: int = 3) -> dict:
    """Measure fit time, predict time per 1k rows, and parameter count."""
    # what: time one fit on the full train set, then time predict_proba over `repeats` runs
    # why: the report's complexity section needs apples-to-apples wall-clock numbers
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr, X_te = X_train.values, X_test.values
    t0 = time.time()
    clf = make_estimator()
    clf.fit(X_tr, y_train)
    fit_sec = time.time() - t0
    # how: median of `repeats` predict calls smooths out OS noise
    times = []
    for _ in range(repeats):
        t0 = time.time()
        if hasattr(clf, "predict_proba"):
            clf.predict_proba(X_te)
        else:
            clf.decision_function(X_te)
        times.append((time.time() - t0) * 1000.0 / len(X_te))   # seconds per 1k rows
    return {"fit_sec": float(fit_sec),
            "predict_per_1k_sec": float(np.median(times)),
            "n_params": int(complexity_proxy(clf))}


# ----------------------------------------------------------------------------
# Model factories
# ----------------------------------------------------------------------------

def make_factories(pca_k: int) -> list[tuple]:
    """Return list of (name, factory, scale_required)."""
    # what: every entry is (name, callable that returns a fresh estimator, whether to scale X)
    # why: lambda re-construction guarantees no state leaks across folds
    factories = [
        ("logreg_l2", lambda: LogisticRegression(C=1.0, penalty="l2", class_weight="balanced",
                                                  max_iter=2000, random_state=RANDOM_SEED), True),
        ("logreg_l1", lambda: LogisticRegression(C=0.1, penalty="l1", solver="liblinear",
                                                  class_weight="balanced", max_iter=2000,
                                                  random_state=RANDOM_SEED), True),
        ("decision_tree", lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=200,
                                                          class_weight="balanced",
                                                          random_state=RANDOM_SEED), False),
        ("random_forest", lambda: RandomForestClassifier(n_estimators=200, max_depth=10,
                                                          min_samples_leaf=200, n_jobs=-1,
                                                          class_weight="balanced",
                                                          random_state=RANDOM_SEED), False),
        ("hist_gbm", lambda: HistGradientBoostingClassifier(max_iter=200, max_depth=8,
                                                             learning_rate=0.05,
                                                             class_weight="balanced",
                                                             random_state=RANDOM_SEED), False),
        ("pca_logreg", lambda: Pipeline([
            ("pca", PCA(n_components=pca_k, random_state=RANDOM_SEED)),
            ("lr",  LogisticRegression(C=1.0, penalty="l2", class_weight="balanced",
                                       max_iter=2000, random_state=RANDOM_SEED))]), True),
        ("mlp_sklearn", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                              solver="adam", alpha=1e-4, batch_size=4096,
                                              learning_rate_init=1e-3, max_iter=50,
                                              early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=5, random_state=RANDOM_SEED), True),
    ]
    # what: optional LightGBM if installed; common alternative gradient-boosting library
    try:
        import lightgbm as lgb
        factories.append(("lightgbm",
                          lambda: lgb.LGBMClassifier(n_estimators=400, num_leaves=63,
                                                     learning_rate=0.05, min_child_samples=200,
                                                     class_weight="balanced",
                                                     random_state=RANDOM_SEED, verbosity=-1),
                          False))
    except ImportError:
        print("  note: lightgbm not installed; skipping (pip install lightgbm to enable)")
    return factories


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Stage 3 — Train and compare models")
    print("=" * 60)
    metrics_dir = OUTPUTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # what: load + size + factories
    X_train, y_train, g_train, X_test, y_test, feature_cols, _test_df = load_xy()
    pca_k = pca_elbow_k(X_train)
    print(f"  PCA elbow K = {pca_k}")
    factories = make_factories(pca_k)

    # what: train every model end-to-end
    summaries = []
    for name, factory, scale in factories:
        try:
            summaries.append(evaluate_model(name, factory, X_train, y_train, g_train,
                                             X_test, y_test, scale=scale))
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            summaries.append({"model": name, "error": str(e)})

    # what: comparison table for the report
    rows = [{"model": s["model"],
             "cv_oof_auc": round(s["cv_oof_auc"], 4),
             "test_auc_raw": round(s["test_raw"]["auc"], 4),
             "test_brier_raw": round(s["test_raw"]["brier"], 4),
             "test_ece_raw": round(s["test_raw"]["ece"], 4)}
            for s in summaries if "error" not in s]
    df = pd.DataFrame(rows).sort_values("test_auc_raw", ascending=False)
    df.to_csv(metrics_dir / "comparison.csv", index=False)
    print("\n" + df.to_string(index=False))

    # what: complexity table — fit time + predict latency + parameter count
    print("\nBenchmarking complexity (fit time / predict latency / param count) ...")
    complexity_rows = []
    for name, factory, scale in factories:
        try:
            comp = benchmark_complexity(factory, X_train, y_train, X_test, scale=scale)
            complexity_rows.append({"model": name, **comp})
            print(f"  {name:14s}  fit={comp['fit_sec']:7.2f}s  "
                  f"predict={comp['predict_per_1k_sec']*1000:6.2f}ms/1k  "
                  f"n_params={comp['n_params']:,}")
        except Exception as e:
            print(f"  {name}: skipped ({e})")
    pd.DataFrame(complexity_rows).to_csv(metrics_dir / "complexity.csv", index=False)

    print(f"\nStage 3 complete. Outputs in {metrics_dir.relative_to(OUTPUTS_DIR.parent)}.")
    print("Proceed to 04_calibration.py.")
    return 0


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    sys.exit(main())
