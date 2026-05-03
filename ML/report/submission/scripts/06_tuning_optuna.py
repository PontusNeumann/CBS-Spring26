"""
06_tuning_optuna.py — Optuna TPE hyperparameter tuning for RF, HistGBM, MLP.

Source-of-truth merge of:
  alex/v5_final_ml_pipeline/scripts/05_optuna_tuning.py    (multi-worker dispatcher)
  alex/v5_final_ml_pipeline/scripts/_optuna_worker.py      (per-model search loop)
  alex/v5_final_ml_pipeline/scripts/_optuna_finalise.py    (refit + comparison)

Teacher-facing simplification: the original Alex pipeline ran one parallel subprocess
per model with a self-refreshing HTML dashboard. Here we run them serially in one
process. Results are identical; total wall-time ~1-2 hours on M4 Pro at default trials.

For each model:
  1. Define a search space.
  2. Maximise mean OOF AUC across GroupKFold (group=market_id, no leakage).
  3. Optuna TPE explores the space for `n_trials` rounds with MedianPruner.
  4. Refit best params on full train, predict on test (raw, no calibration here).
  5. Save best_params.json + comparison_vs_default.json + preds_test_tuned.npz.

Run:
  python 06_tuning_optuna.py --models random_forest hist_gbm --n_trials 30
  python 06_tuning_optuna.py --models mlp_sklearn       --n_trials 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, OUTPUTS_DIR, RANDOM_SEED, N_FOLDS  # noqa: E402

# what: optuna is the only non-stdlib dep specific to this file
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("optuna not installed; run `pip install optuna` then re-run.")
    sys.exit(1)


TARGET = "bet_correct"


# ----------------------------------------------------------------------------
# Search spaces — kept tight so each trial is feasible inside the wall-clock budget
# ----------------------------------------------------------------------------

def rf_search_space(trial) -> dict:
    """Random Forest space."""
    # what: bound the depth and leaf-size to keep per-trial fit time under ~5 min
    # why: an unbounded RF (max_depth=None) on 1.1M rows took 45 min/trial -> infeasible overnight
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10, 12, 15]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 100, 1000, log=True),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
    }


def hgbm_search_space(trial) -> dict:
    """HistGradientBoosting space."""
    # what: standard HGBM tunables; max_iter capped at 500 for wall-time
    return {
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8, 10, "none"]),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 500, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 1.0, log=True),
    }


def mlp_search_space(trial) -> dict:
    """sklearn MLP space — picks one of 6 architectures plus the usual reg/lr knobs."""
    # what: choose the architecture from a discrete set rather than independently per-layer
    # why: independent per-layer search blows up the space; preset arches converge faster
    architectures = [(64,), (128,), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64)]
    return {
        "arch_idx": trial.suggest_int("arch_idx", 0, len(architectures) - 1),
        "_arch_lookup": architectures,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
    }


# ----------------------------------------------------------------------------
# Estimator factories
# ----------------------------------------------------------------------------

def make_rf(p: dict):
    return RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
        n_jobs=-1, class_weight="balanced", random_state=RANDOM_SEED)


def make_hgbm(p: dict):
    md = None if p["max_depth"] == "none" else p["max_depth"]
    return HistGradientBoostingClassifier(
        max_iter=p["max_iter"], learning_rate=p["learning_rate"],
        max_depth=md, max_leaf_nodes=p["max_leaf_nodes"],
        min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2_regularization"],
        class_weight="balanced", random_state=RANDOM_SEED)


def make_mlp(p: dict):
    arch = p["_arch_lookup"][p["arch_idx"]]
    return MLPClassifier(
        hidden_layer_sizes=arch, activation=p["activation"], solver="adam",
        alpha=p["alpha"], batch_size=p["batch_size"],
        learning_rate_init=p["learning_rate_init"], max_iter=50,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
        random_state=RANDOM_SEED)


# ----------------------------------------------------------------------------
# Data loader
# ----------------------------------------------------------------------------

def load_xy() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Load consolidated parquet, split, get the canonical feature list."""
    df = pd.read_parquet(DATA_DIR / "consolidated_modeling_data.parquet")
    feature_cols = json.loads((OUTPUTS_DIR / "data" / "feature_cols.json").read_text())
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)
    g_train = train["market_id"]
    return X_train, y_train, g_train, X_test, y_test, feature_cols


# ----------------------------------------------------------------------------
# Objective + tuning loop per model
# ----------------------------------------------------------------------------

def objective_kfold(make_fn, params: dict, X, y, groups, n_folds: int, scale: bool) -> float:
    """Mean OOF AUC across folds — the value Optuna maximises."""
    # what: same GroupKFold protocol used by 03_train_models so OOF AUC is directly comparable
    gkf = GroupKFold(n_splits=n_folds)
    aucs = []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X.iloc[tr_idx])
            X_va = scaler.transform(X.iloc[va_idx])
        else:
            X_tr, X_va = X.iloc[tr_idx].values, X.iloc[va_idx].values
        clf = make_fn(params).fit(X_tr, y.iloc[tr_idx])
        if hasattr(clf, "predict_proba"):
            preds = clf.predict_proba(X_va)[:, 1]
        else:
            d = clf.decision_function(X_va)
            preds = (d - d.min()) / (d.max() - d.min() + 1e-9)
        aucs.append(roc_auc_score(y.iloc[va_idx], preds))
    return float(np.mean(aucs))


def objective_holdout(make_fn, params: dict, X, y, groups, scale: bool) -> float:
    """Single GroupShuffleSplit holdout AUC — used for slow models like MLP."""
    # what: 80/20 group-aware holdout; 5x faster than full KFold; trade rigour for runtime
    # why: MLP fit takes ~5 min per fold -> 25 min per trial -> 30 trials = 12.5 hr (too slow)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    tr_idx, va_idx = next(splitter.split(X, y, groups))
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X.iloc[tr_idx])
        X_va = scaler.transform(X.iloc[va_idx])
    else:
        X_tr, X_va = X.iloc[tr_idx].values, X.iloc[va_idx].values
    clf = make_fn(params).fit(X_tr, y.iloc[tr_idx])
    preds = clf.predict_proba(X_va)[:, 1] if hasattr(clf, "predict_proba") \
        else clf.decision_function(X_va)
    return float(roc_auc_score(y.iloc[va_idx], preds))


def tune_one_model(model_name: str, n_trials: int) -> None:
    """Optuna TPE loop for one model, then refit best on full train + predict on test."""
    print("\n" + "=" * 60)
    print(f"Tuning {model_name} ({n_trials} trials)")
    print("=" * 60)

    # what: per-model wiring
    if model_name == "random_forest":
        space, make_fn, scale, holdout = rf_search_space, make_rf, False, False
    elif model_name == "hist_gbm":
        space, make_fn, scale, holdout = hgbm_search_space, make_hgbm, False, False
    elif model_name == "mlp_sklearn":
        space, make_fn, scale, holdout = mlp_search_space, make_mlp, True, True
    else:
        raise SystemExit(f"unknown model: {model_name}")

    X_train, y_train, g_train, X_test, y_test, _ = load_xy()
    out_dir = OUTPUTS_DIR / "tuning" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # what: define the optuna objective in closure form so the search space + factory are bound
    def objective(trial):
        params = space(trial)
        if holdout:
            return objective_holdout(make_fn, params, X_train, y_train, g_train, scale=scale)
        return objective_kfold(make_fn, params, X_train, y_train, g_train,
                                n_folds=N_FOLDS, scale=scale)

    # what: TPE sampler with median pruner — standard Optuna recipe for tabular tuning
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                 study_name=model_name)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  done in {(time.time() - t0)/60:.1f} min  best AUC = {study.best_value:.4f}")

    # what: persist study history + best params
    history_rows = [{"trial": t.number, "value": t.value, "state": str(t.state),
                     "duration_sec": t.duration.total_seconds() if t.duration else None,
                     **t.params} for t in study.trials]
    pd.DataFrame(history_rows).to_csv(out_dir / "study_history.csv", index=False)
    (out_dir / "best_params.json").write_text(json.dumps(
        {"best_oof_auc": float(study.best_value), "best_params": study.best_params,
         "n_trials": len(study.trials)}, indent=2))

    # what: refit best params on FULL train, predict on test
    print("  refitting with best params on full train ...")
    best = dict(study.best_params)
    if model_name == "mlp_sklearn":
        best["_arch_lookup"] = mlp_search_space.__defaults__ or [(64,), (128,), (64, 32),
                                                                   (128, 64), (128, 64, 32),
                                                                   (256, 128, 64)]
    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr, X_te = X_train.values, X_test.values
    clf = make_fn(best).fit(X_tr, y_train)
    raw_tuned = clf.predict_proba(X_te)[:, 1]
    np.savez_compressed(out_dir / "preds_test_tuned.npz", raw=raw_tuned.astype("float32"))

    tuned_auc = float(roc_auc_score(y_test, raw_tuned))

    # what: compare against the default model's raw test AUC (already saved by 03_train_models)
    default_path = OUTPUTS_DIR / "models" / model_name / "preds_test.npz"
    delta = None
    if default_path.exists():
        default_raw = np.load(default_path)["raw"]
        default_auc = float(roc_auc_score(y_test, default_raw))
        delta = tuned_auc - default_auc
        print(f"  test AUC: tuned={tuned_auc:.4f}  default={default_auc:.4f}  Δ={delta:+.4f}")
    (out_dir / "comparison_vs_default.json").write_text(json.dumps({
        "model": model_name, "best_oof_auc": float(study.best_value),
        "test_auc_tuned": tuned_auc,
        "test_auc_default": float(default_auc) if default_path.exists() else None,
        "delta_test_auc": delta, "best_params": study.best_params,
        "n_trials": n_trials,
    }, indent=2))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=["random_forest", "hist_gbm", "mlp_sklearn"],
                        default=["random_forest", "hist_gbm"])
    parser.add_argument("--n_trials", type=int, default=30)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Stage 6 — Optuna tuning ({len(args.models)} model(s) × {args.n_trials} trials)")
    print("=" * 60)
    for m in args.models:
        try:
            tune_one_model(m, n_trials=args.n_trials)
        except Exception as e:
            print(f"  [{m}] FAILED: {e}")

    print("\nStage 6 complete. Tuned predictions are in outputs/tuning/<model>/preds_test_tuned.npz.")
    print("Re-run 04_calibration.py and 05_backtest.py if you want the tuned predictions in the headline.")
    return 0


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    sys.exit(main())
