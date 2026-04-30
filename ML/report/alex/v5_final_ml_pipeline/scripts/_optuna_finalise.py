"""
_optuna_finalise.py — wrap up an Optuna study without running more trials.

Use after killing a long-running tuning job. Loads the existing SQLite study,
picks the best COMPLETE trial, refits on full train, writes the same artefacts
that `_optuna_worker.py` produces at the end of its run:

  - best_params.json
  - study_history.csv
  - comparison.json
  - preds_test_tuned.npz

Usage:
  python _optuna_finalise.py --model random_forest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA  # noqa: E402
from _optuna_worker import (  # noqa: E402
    make_hgbm,
    make_rf,
)

try:
    from _optuna_worker import make_lgbm  # type: ignore
except ImportError:
    make_lgbm = None

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / ".scratch" / "backtest"
OUT = ROOT / "outputs" / "v5" / "rigor" / "optuna"

FACTORIES = {
    "random_forest": make_rf,
    "hist_gbm": make_hgbm,
    "lightgbm": make_lgbm,
}

TRAIN_PARQUET = "train_features_v4.parquet"
TEST_PARQUET = "test_features_v4.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(FACTORIES))
    args = ap.parse_args()

    if FACTORIES[args.model] is None:
        raise SystemExit(
            f"factory for {args.model} not available (lightgbm not installed?)"
        )

    out_dir = OUT / args.model
    storage = f"sqlite:///{out_dir / 'study.db'}"
    study = optuna.load_study(study_name=f"{args.model}_tuning", storage=storage)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise SystemExit(
            f"no COMPLETE trials in study {args.model}_tuning at {storage}"
        )
    print(
        f"[{args.model}] {len(completed)} completed trials, "
        f"{len(study.trials) - len(completed)} other"
    )

    # study.best_value / study.best_params consider only completed trials
    best_value = float(study.best_value)
    best_params = dict(study.best_params)
    print(f"[{args.model}] best OOF AUC: {best_value:.5f}")
    print(f"[{args.model}] best params: {best_params}")

    # Save study history + best_params
    hist = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "duration")
    )
    hist.to_csv(out_dir / "study_history.csv", index=False)

    best = {
        "best_value_oof_auc": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "n_pruned": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ),
        "note": "finalised after early-stop; some scheduled trials were not run",
    }
    (out_dir / "best_params.json").write_text(json.dumps(best, indent=2, default=str))

    # Load data, refit, predict
    print(f"[{args.model}] loading data...", flush=True)
    feature_cols = json.loads((DATA / "feature_cols.json").read_text())
    train = pd.read_parquet(DATA / TRAIN_PARQUET)
    test = pd.read_parquet(DATA / TEST_PARQUET)
    X_train = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train["bet_correct"].astype(int)
    X_test = test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test["bet_correct"].astype(int)
    print(f"[{args.model}] train {X_train.shape}, test {X_test.shape}", flush=True)

    # Tree models don't strictly need scaling, but mirror _optuna_worker's path
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_train)
    X_te_s = sc.transform(X_test)

    print(f"[{args.model}] refit best params on full train...", flush=True)
    factory = FACTORIES[args.model]
    final = factory(best_params, n_jobs=-1)
    final.fit(X_tr_s, y_train)
    raw_test_tuned = final.predict_proba(X_te_s)[:, 1]
    test_auc_tuned = float(roc_auc_score(y_test, raw_test_tuned))
    print(f"[{args.model}] tuned test AUC: {test_auc_tuned:.5f}", flush=True)

    comparison = {
        "model": args.model,
        "tuned": {
            "best_oof_auc": best_value,
            "test_auc_raw": test_auc_tuned,
            "best_params": best_params,
        },
    }
    default_path = SCRATCH / f"preds_{args.model}.npz"
    if default_path.exists():
        d = np.load(default_path)
        raw_default = d["raw"]
        test_auc_default = float(roc_auc_score(y_test, raw_default))
        comparison["default"] = {"test_auc_raw": test_auc_default}
        comparison["delta_test_auc"] = float(test_auc_tuned - test_auc_default)
        print(
            f"[{args.model}] tuned: {test_auc_tuned:.5f}  "
            f"default: {test_auc_default:.5f}  "
            f"delta: {test_auc_tuned - test_auc_default:+.5f}",
            flush=True,
        )

    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str)
    )
    np.savez_compressed(out_dir / "preds_test_tuned.npz", raw=raw_test_tuned)
    print(f"[{args.model}] DONE → {out_dir}")


if __name__ == "__main__":
    main()
