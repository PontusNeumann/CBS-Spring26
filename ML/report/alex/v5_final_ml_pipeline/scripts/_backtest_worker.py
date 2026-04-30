"""
_backtest_worker.py — train ONE model, save predictions to disk.

Used by 10_backtest.py for parallel multi-model training.

Usage:
    python _backtest_worker.py --model logreg_l2 --out path/preds.npz

Each worker:
  1. Loads train_features.parquet + test_features.parquet (independent — avoids IPC)
  2. Runs 5-fold GroupKFold CV → OOF predictions (for calibration)
  3. Final fit on full train
  4. Isotonic calibration on OOF
  5. Saves {raw, cal, oof, feature_importances?} as compressed npz

n_jobs is set to a moderate value (4) so 3 parallel workers don't oversaturate cores
on M4 Pro (~12 cores total, 3 × 4 = 12).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Per-worker cores — keep moderate so 3 workers fit on 12-core M4 Pro
N_JOBS_PER_WORKER = 4
RANDOM_SEED = 42
N_FOLDS = 5

# v4 contract — fail fast if pointed at v3.5 parquets or pre-Stage-1 schema.
TRAIN_PARQUET = "train_features_v4.parquet"
TEST_PARQUET = "test_features_v4.parquet"
EXPECTED_N_FEATURES = 64  # cleaned: 80 - 16 cohort-flip features dropped per D-042


def make_logreg_l2():
    return LogisticRegression(
        C=1.0,
        penalty="l2",
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=200,
        n_jobs=N_JOBS_PER_WORKER,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def make_hist_gbm():
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=8,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def make_lightgbm():
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.05,
        min_child_samples=200,
        class_weight="balanced",
        n_jobs=N_JOBS_PER_WORKER,
        random_state=RANDOM_SEED,
        verbosity=-1,
    )


def make_xgboost():
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=200,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=1.0,
        n_jobs=N_JOBS_PER_WORKER,
        random_state=RANDOM_SEED,
        verbosity=0,
    )


def make_logreg_l1():
    return LogisticRegression(
        C=0.1,
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )


def make_decision_tree():
    return DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=200,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def make_pca_logreg():
    # PCA k=16 from sweep elbow (alex/outputs/sweep_idea1/pca_logreg/pca_selection.json).
    # Pipeline expects scaled input — driver passes scale=True.
    return Pipeline(
        [
            ("pca", PCA(n_components=16, random_state=RANDOM_SEED)),
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
    )


def make_mlp_sklearn():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=4096,
        learning_rate_init=1e-3,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=RANDOM_SEED,
    )


MODELS = {
    "logreg_l2": (make_logreg_l2, True),
    "logreg_l1": (make_logreg_l1, True),
    "decision_tree": (make_decision_tree, False),
    "random_forest": (make_random_forest, False),
    "hist_gbm": (make_hist_gbm, False),
    "lightgbm": (make_lightgbm, False),
    "xgboost": (make_xgboost, False),
    "pca_logreg": (make_pca_logreg, True),
    "mlp_sklearn": (make_mlp_sklearn, True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--out", required=True, type=str)
    args = parser.parse_args()

    factory, scale = MODELS[args.model]
    print(f"[{args.model}] starting (n_jobs={N_JOBS_PER_WORKER})", flush=True)

    # --- v4 data guard ------------------------------------------------------
    train_path = DATA / TRAIN_PARQUET
    test_path = DATA / TEST_PARQUET
    missing = [str(p) for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise SystemExit(
            f"v4 parquet(s) missing: {missing}. Pontus has not delivered, or "
            f"Stage 0 pre-flight was skipped. Run 01_validate_schema.py first."
        )
    fcols = json.loads((DATA / "feature_cols.json").read_text())
    if len(fcols) != EXPECTED_N_FEATURES:
        raise SystemExit(
            f"feature_cols.json has {len(fcols)} features, expected "
            f"{EXPECTED_N_FEATURES}. Run 01_validate_schema.py to update it."
        )

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    X_train = train[fcols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train["bet_correct"].astype(int).values
    g_train = train["market_id"].values
    X_test = test[fcols].fillna(0).replace([np.inf, -np.inf], 0)

    print(f"[{args.model}] CV {N_FOLDS}-fold GroupKFold", flush=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(y_train), dtype=float)
    for fold_idx, (tr, va) in enumerate(gkf.split(X_train, y_train, g_train)):
        if scale:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_train.iloc[tr])
            X_va_s = sc.transform(X_train.iloc[va])
        else:
            X_tr_s = X_train.iloc[tr].values
            X_va_s = X_train.iloc[va].values
        clf = factory()
        clf.fit(X_tr_s, y_train[tr])
        oof[va] = clf.predict_proba(X_va_s)[:, 1]
        print(f"[{args.model}]   fold {fold_idx + 1}/{N_FOLDS} done", flush=True)

    print(f"[{args.model}] final fit on full train", flush=True)
    if scale:
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s = sc.transform(X_test)
    else:
        X_train_s = X_train.values
        X_test_s = X_test.values

    final = factory()
    final.fit(X_train_s, y_train)
    raw = final.predict_proba(X_test_s)[:, 1]

    print(f"[{args.model}] isotonic calibration", flush=True)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof, y_train)
    cal_test = cal.transform(raw)

    # Feature importances if available. For Pipeline (pca_logreg), final.coef_
    # lives in PCA-space — skip attribution rather than emit something misleading.
    if isinstance(final, Pipeline):
        fi = {}
    elif hasattr(final, "feature_importances_"):
        fi = dict(zip(fcols, final.feature_importances_.tolist()))
    elif hasattr(final, "coef_"):
        fi = dict(zip(fcols, final.coef_[0].tolist()))
    else:
        fi = {}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, raw=raw, cal=cal_test, oof=oof)
    fi_path = out.with_suffix(".importance.json")
    fi_path.write_text(json.dumps(fi, indent=2))

    print(f"[{args.model}] DONE: preds → {out}, importance → {fi_path}", flush=True)


if __name__ == "__main__":
    main()
