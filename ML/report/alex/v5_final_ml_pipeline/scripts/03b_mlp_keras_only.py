"""
03b_mlp_keras_only.py

Run only the v4 Keras MLP stage and refresh the sweep comparison table from
the model outputs already present on disk. This is useful when the full
03_sweep.py run has completed the classical models but got stuck in the
TensorFlow MLP block.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_PATH = SCRIPT_DIR / "03_sweep.py"
ROOT = SCRIPT_DIR.parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "v5" / "sweep_idea1"


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("v4_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
                    "test_auc_cal": f"{s['anomaly_score_test_auc']:.3f}",
                    "top1pct_prec": f"{s['top_1pct_precision']:.3f}",
                    "top5pct_prec": f"{s['top_5pct_precision']:.3f}",
                    "anom_target_corr": f"{s['anomaly_score_target_corr_test']:+.3f}",
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


def main():
    sweep = _load_sweep_module()
    print("=" * 60, flush=True)
    print("v4 MLP-only sweep stage", flush=True)
    print("=" * 60, flush=True)

    train_path = DATA / sweep.TRAIN_PARQUET
    test_path = DATA / sweep.TEST_PARQUET
    missing = [str(p) for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise SystemExit(f"v4 parquet(s) missing: {missing}")

    feature_cols = json.loads((DATA / "feature_cols.json").read_text())
    if len(feature_cols) != sweep.EXPECTED_N_FEATURES:
        raise SystemExit(
            f"feature_cols.json has {len(feature_cols)} features, expected "
            f"{sweep.EXPECTED_N_FEATURES}"
        )

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    X_train = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train["bet_correct"].astype(int)
    g_train = train["market_id"].values
    X_test = test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test["bet_correct"].astype(int)

    print(
        f"train: {len(X_train):,} x {X_train.shape[1]}, "
        f"test: {len(X_test):,}",
        flush=True,
    )
    sweep.evaluate_keras_mlp(X_train, y_train, g_train, X_test, y_test, test)
    _refresh_comparison_table()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
