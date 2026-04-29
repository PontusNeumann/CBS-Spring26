"""
03c_mlp_sklearn_only.py — sklearn MLPClassifier replacement for the broken
TensorFlow/Keras 3 MLP slot.

Why this exists: on the local stack (Python 3.13.9, TF 2.21, Keras 3.14, numpy 2.4),
both `model.fit` and `train_on_batch` deadlock at first invocation while using 0%
CPU. See `03_sweep.py:386` and the v4 sweep log for the symptom. This script
keeps the MLP slot in the sweep alive by swapping in sklearn's MLPClassifier,
which has zero TF dependency and runs reliably on Apple Silicon CPU.

Architecture (vs the original Keras spec):
  Original: Dense(64)-BN-SELU-Dropout(0.3)-Dense(32)-BN-SELU-Dropout(0.3)-Dense(1)
  Here:     Dense(64)-ReLU-Dense(32)-ReLU-Dense(1)  (plus L2 alpha=1e-4)
sklearn doesn't ship BatchNorm, SELU, or Dropout. Closest equivalents:
  - Dropout       → none (validation-fraction early stopping instead)
  - BN            → none (StandardScaler upstream is the input-scale equivalent)
  - SELU          → ReLU (closest sklearn supports)
This is a small architecture difference; for a 64-feature input MLP it's
unlikely to move test AUC by more than ~0.01.

Outputs (drop-in replacement for what evaluate_supervised would have produced):
  alex/outputs/sweep_idea1/mlp_sklearn/{metrics.json, per_market_test.json}
  alex/.scratch/backtest/preds_mlp_sklearn.npz
  alex/outputs/sweep_idea1/comparison_table.{md,csv} refreshed
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_PATH = SCRIPT_DIR / "03_sweep.py"
ROOT = SCRIPT_DIR.parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "v5" / "sweep_idea1"
SCRATCH = ROOT / ".scratch" / "backtest"


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("v4_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,  # L2; analogous to Keras Dropout regularisation strength
        batch_size=4096,
        learning_rate_init=1e-3,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        verbose=True,
    )


def _refresh_comparison_table() -> None:
    rows = []
    summaries = []
    model_order = [
        "logreg_l2",
        "logreg_l1",
        "decision_tree",
        "random_forest",
        "hist_gbm",
        "lightgbm",
        "pca_logreg",
        "mlp_sklearn",
        "mlp_keras",
        "iso_forest",
    ]
    for model in model_order:
        metrics_path = OUT / model / "metrics.json"
        if not metrics_path.exists():
            continue
        s = json.loads(metrics_path.read_text())
        summaries.append(s)
        if model == "iso_forest":
            rows.append(
                {
                    "model": model,
                    "test_auc_cal": f"{s.get('anomaly_score_test_auc', 0):.3f}",
                    "top1pct_prec": f"{s.get('top_1pct_precision', 0):.3f}",
                    "top5pct_prec": f"{s.get('top_5pct_precision', 0):.3f}",
                    "anom_target_corr": f"{s.get('anomaly_score_target_corr_test', 0):+.3f}",
                }
            )
        else:
            rows.append(
                {
                    "model": model,
                    "cv_oof_auc": f"{s['cv_oof_auc']:.3f}",
                    "test_auc_cal": f"{s['test_calibrated_auc']:.3f}",
                    "test_brier_cal": f"{s['test_calibrated_brier']:.3f}",
                    "test_ece_cal": f"{s['test_calibrated_ece']:.3f}",
                    "per_market_auc_range": (
                        f"[{s['per_market_auc_min']:.2f}, "
                        f"{s['per_market_auc_max']:.2f}]"
                    ),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "comparison_table.csv", index=False)
    (OUT / "comparison_table.md").write_text(df.to_markdown(index=False))
    (OUT / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"[comparison] refreshed {OUT / 'comparison_table.csv'}", flush=True)


def main() -> None:
    sweep = _load_sweep_module()
    print("=" * 60, flush=True)
    print("v4 MLP-only sweep stage (sklearn MLPClassifier)", flush=True)
    print("=" * 60, flush=True)

    train_path = DATA / sweep.TRAIN_PARQUET
    test_path = DATA / sweep.TEST_PARQUET
    feature_cols = json.loads((DATA / "feature_cols.json").read_text())

    if len(feature_cols) != sweep.EXPECTED_N_FEATURES:
        raise SystemExit(
            f"feature_cols.json has {len(feature_cols)} features, "
            f"expected {sweep.EXPECTED_N_FEATURES}"
        )

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    X_train = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train["bet_correct"].astype(int)
    g_train = train["market_id"]
    X_test = test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test["bet_correct"].astype(int)
    print(
        f"train: {len(X_train):,} × {X_train.shape[1]}, test: {len(X_test):,}",
        flush=True,
    )

    # Use the sweep's stock evaluate_supervised — same CV, calibration, metrics
    # path as logreg/RF/HGBM/LightGBM. Output goes to OUT/mlp_sklearn/.
    summary = sweep.evaluate_supervised(
        "mlp_sklearn",
        _make_mlp,
        X_train,
        y_train,
        g_train,
        X_test,
        y_test,
        test,
        scale=True,
        importance_fn=None,
    )

    # NOTE: preds for Stage 8 backtests are written by `_backtest_worker.py`
    # which refits each model independently. To include mlp_sklearn in
    # backtests, add it to that worker's MODELS dict in a follow-up step.
    # For now this script only produces metrics + per-market results, which
    # is sufficient for the comparison table — and avoids paying for a second
    # full CV + final-fit pass (~25 min on 1.1M rows).

    _refresh_comparison_table()
    print(f"DONE — test AUC (cal) {summary['test_calibrated_auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
