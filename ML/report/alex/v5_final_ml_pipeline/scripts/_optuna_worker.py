"""
_optuna_worker.py — tune ONE model via Optuna TPE, save study + comparison.

Used by 05_optuna_tuning.py for parallel multi-model tuning.

Usage:
    python _optuna_worker.py --model random_forest --n_trials 50

Each worker:
  1. Loads train_features.parquet (test held out)
  2. Defines a search space + objective (5-fold GroupKFold on market_id, mean AUC)
  3. Runs Optuna TPE with MedianPruner over n_trials
  4. Refits best params on full train, predicts on test (raw, no calibration)
  5. Saves: best_params.json, study_history.csv, comparison.json (vs current default)

GroupKFold is mandatory — preserves cross-regime / market-level transfer claim.
Cached default predictions are loaded from .scratch/backtest/preds_*.npz to
compute the default-vs-tuned AUC delta without retraining the default model.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SCRATCH = ROOT / ".scratch" / "backtest"
OUT_BASE = ROOT / "outputs" / "v5" / "rigor" / "optuna"

N_FOLDS = 5
RANDOM_SEED = 42

# v4 contract — fail fast if pointed at v3.5 parquets or pre-Stage-1 schema.
TRAIN_PARQUET = "train_features_v4.parquet"
TEST_PARQUET = "test_features_v4.parquet"
EXPECTED_N_FEATURES = 64  # cleaned: 80 - 16 cohort-flip features dropped per D-042


def make_rf(params, n_jobs=4):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"] if params["max_depth"] != "none" else None,
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        n_jobs=n_jobs,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def make_hgbm(params):
    return HistGradientBoostingClassifier(
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"] if params["max_depth"] != "none" else None,
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def rf_search_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_categorical("max_depth", ["none", 6, 8, 10, 12, 16]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 800, log=True),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
        ),
    }


def hgbm_search_space(trial: optuna.Trial) -> dict:
    return {
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_categorical("max_depth", ["none", 4, 6, 8, 10]),
        "max_leaf_nodes": trial.suggest_categorical(
            "max_leaf_nodes", [15, 31, 63, 127]
        ),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 2.0),
    }


def make_lgbm(params, n_jobs=4):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"] if params["max_depth"] != "none" else -1,
        min_child_samples=params["min_child_samples"],
        reg_lambda=params["reg_lambda"],
        feature_fraction=params["feature_fraction"],
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
        verbosity=-1,
    )


def lgbm_search_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63, 127, 255]),
        "max_depth": trial.suggest_categorical("max_depth", ["none", 4, 6, 8, 10]),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 500, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
    }


_MLP_ARCH_CHOICES = [
    (64,),
    (128,),
    (256,),
    (64, 32),
    (128, 64),
    (256, 128),
    (128, 64, 32),
    (256, 128, 64),
]


def make_mlp(params):
    # Accept either resolved hidden_layer_sizes (tuple/list) or arch_idx into _MLP_ARCH_CHOICES
    if "hidden_layer_sizes" in params:
        arch = params["hidden_layer_sizes"]
    else:
        arch = _MLP_ARCH_CHOICES[params["arch_idx"]]
    return MLPClassifier(
        hidden_layer_sizes=tuple(arch),
        activation=params["activation"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        batch_size=params["batch_size"],
        max_iter=params.get("max_iter", 30),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=4,
        solver="adam",
        random_state=RANDOM_SEED,
    )


def mlp_search_space(trial: optuna.Trial) -> dict:
    return {
        "arch_idx": trial.suggest_int("arch_idx", 0, len(_MLP_ARCH_CHOICES) - 1),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init", 1e-4, 1e-2, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
        "max_iter": 30,
    }


def holdout_auc(X, y, groups, factory, scale: bool, trial=None) -> float:
    """Single GroupShuffleSplit holdout — much faster than 5-fold for MLP.

    Drops one ~20% group-aware slice from train, fits on the rest, scores AUC
    on the held-out slice. The selected config is later refit on full train.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    tr, va = next(splitter.split(X, y, groups))
    if scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X.iloc[tr])
        X_va = sc.transform(X.iloc[va])
    else:
        X_tr = X.iloc[tr].values
        X_va = X.iloc[va].values
    clf = factory()
    clf.fit(X_tr, y.iloc[tr])
    proba = clf.predict_proba(X_va)[:, 1]
    return float(roc_auc_score(y.iloc[va], proba))


def cv_mean_auc(X, y, groups, factory, scale: bool, trial=None) -> float:
    gkf = GroupKFold(n_splits=N_FOLDS)
    aucs = []
    for fold_idx, (tr, va) in enumerate(gkf.split(X, y, groups)):
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X.iloc[tr])
            X_va = sc.transform(X.iloc[va])
        else:
            X_tr = X.iloc[tr].values
            X_va = X.iloc[va].values
        clf = factory()
        clf.fit(X_tr, y.iloc[tr])
        proba = clf.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y.iloc[va], proba)
        aucs.append(auc)
        # Pruner check after each fold
        if trial is not None:
            trial.report(float(np.mean(aucs)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(aucs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["random_forest", "hist_gbm", "lightgbm", "mlp_sklearn"],
    )
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="cores per trial for parallelisable models (RF / LGBM)",
    )
    args = parser.parse_args()

    out_dir = OUT_BASE / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{out_dir / 'study.db'}"

    print(f"[{args.model}] loading data...", flush=True)

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
    y_train = train["bet_correct"].astype(int)
    g_train = train["market_id"].values
    X_test = test[fcols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test["bet_correct"].astype(int).values
    print(
        f"[{args.model}] train {len(X_train):,}×{X_train.shape[1]}, test {len(X_test):,}",
        flush=True,
    )

    eval_mode = "cv"  # default — switched to "holdout" for MLP
    if args.model == "random_forest":
        space_fn = rf_search_space
        factory = lambda p: lambda: make_rf(p, n_jobs=args.n_jobs)
        scale = False
    elif args.model == "hist_gbm":
        space_fn = hgbm_search_space
        factory = lambda p: lambda: make_hgbm(p)
        scale = False
    elif args.model == "lightgbm":
        space_fn = lgbm_search_space
        factory = lambda p: lambda: make_lgbm(p, n_jobs=args.n_jobs)
        scale = False
    elif args.model == "mlp_sklearn":
        space_fn = mlp_search_space
        factory = lambda p: lambda: make_mlp(p)
        scale = True
        eval_mode = "holdout"  # MLP 5-fold is too slow; single 80/20 group split
    else:
        raise SystemExit(f"unknown model: {args.model}")

    def objective(trial):
        params = space_fn(trial)
        if eval_mode == "holdout":
            return holdout_auc(
                X_train, y_train, g_train, factory(params), scale=scale, trial=trial
            )
        return cv_mean_auc(
            X_train, y_train, g_train, factory(params), scale=scale, trial=trial
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        storage=storage,
        study_name=f"{args.model}_tuning",
        load_if_exists=True,
    )
    print(
        f"[{args.model}] starting {args.n_trials} trials (storage={storage})",
        flush=True,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # Save study history
    hist = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "duration")
    )
    hist.to_csv(out_dir / "study_history.csv", index=False)

    # Best params
    best = {
        "best_value_oof_auc": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_pruned": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ),
        "n_completed": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ),
    }
    (out_dir / "best_params.json").write_text(json.dumps(best, indent=2, default=str))
    print(f"[{args.model}] best OOF AUC: {study.best_value:.5f}", flush=True)
    print(f"[{args.model}] best params: {study.best_params}", flush=True)

    # Refit on full train with best params, predict on test
    print(f"[{args.model}] refit best params on full train...", flush=True)
    best_factory = factory(study.best_params)
    if scale:
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train)
        X_te_s = sc.transform(X_test)
    else:
        X_tr_s = X_train.values
        X_te_s = X_test.values
    final = best_factory()
    final.fit(X_tr_s, y_train)
    raw_test_tuned = final.predict_proba(X_te_s)[:, 1]
    test_auc_tuned = float(roc_auc_score(y_test, raw_test_tuned))

    # Compare to default (cached predictions from current pipeline)
    default_path = SCRATCH / f"preds_{args.model}.npz"
    comparison = {
        "model": args.model,
        "tuned": {
            "best_oof_auc": float(study.best_value),
            "test_auc_raw": test_auc_tuned,
            "best_params": study.best_params,
        },
    }
    if default_path.exists():
        d = np.load(default_path)
        raw_default = d["raw"]
        test_auc_default = float(roc_auc_score(y_test, raw_default))
        comparison["default"] = {
            "test_auc_raw": test_auc_default,
            "n_estimators_etc": "see _backtest_worker.py defaults",
        }
        comparison["delta_test_auc"] = float(test_auc_tuned - test_auc_default)
        print(
            f"[{args.model}] tuned test AUC: {test_auc_tuned:.5f} | "
            f"default: {test_auc_default:.5f} | "
            f"delta: {test_auc_tuned - test_auc_default:+.5f}",
            flush=True,
        )

    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str)
    )

    # Save raw tuned predictions for downstream (calibration / backtest reuse)
    np.savez_compressed(out_dir / "preds_test_tuned.npz", raw=raw_test_tuned)

    print(f"[{args.model}] DONE → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
