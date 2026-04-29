"""
07_rigor_additions.py — Stage 7.2 of the v4 pipeline. Statistical rigor.

Produces four standard ML-paper additions on top of the supervised sweep:

  1. Bootstrap CI on test AUC per model (1000 resamples, 95% percentile CI)
  2. DeLong test for pairwise model AUC differences (with Bonferroni correction)
  3. Permutation importance per model (top-15 features by mean AUC drop)
  4. Learning curves (test AUC vs train fraction in {10, 25, 50, 75, 100}%)

Required by the report's Methodology + Results sections (D-037).

Inputs:
  data/{train,test}_features_v4.parquet
  .scratch/preds/preds_{random_forest,hist_gbm,lightgbm,mlp_keras,...}.npz
    (assumes 03_sweep.py + 06_optuna_tuning.py have already run)

Outputs:
  outputs/rigor/auc_ci.json                     model → {auc, ci_lower, ci_upper}
  outputs/rigor/deLong_pairwise.json            pairwise p-values + corrected
  outputs/rigor/perm_importance_<model>.json    feature → AUC drop
  outputs/rigor/learning_curve.{json,png}       AUC vs train fraction per model

STATUS: skeleton — the four sub-routines below are stubbed out with
TODO markers. Wall time when filled in: ~1.5 hr total.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from _common import DATA, ROOT, SCRATCH

warnings.filterwarnings("ignore")

OUT = ROOT / "outputs" / "rigor"
OUT.mkdir(parents=True, exist_ok=True)

PRED_DIR = SCRATCH / "backtest"  # cached predictions from 03_sweep / 06_optuna_tuning
N_BOOTSTRAP = 1_000
RANDOM_SEED = 42

MODELS_TO_ANALYZE = ["random_forest", "hist_gbm", "lightgbm", "mlp_keras", "logreg_l2"]


# ---------------------------------------------------------------------------
# 1. Bootstrap CI on test AUC
# ---------------------------------------------------------------------------


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_iter: int = N_BOOTSTRAP):
    """Stratified bootstrap. Returns (mean_auc, ci_lower, ci_upper)."""
    # TODO: implement
    # rng = np.random.default_rng(RANDOM_SEED)
    # n = len(y_true)
    # aucs = []
    # for _ in range(n_iter):
    #     idx = rng.integers(0, n, size=n)
    #     aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    # return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
    raise NotImplementedError("bootstrap_auc_ci")


# ---------------------------------------------------------------------------
# 2. DeLong test for paired AUC comparison
# ---------------------------------------------------------------------------


def delong_pairwise(y_true, prob_a, prob_b):
    """Returns p-value for H0: AUC_a == AUC_b. Implements DeLong's test."""
    # TODO: implement (or use scipy.stats.bootstrap as a paired-bootstrap approximation)
    raise NotImplementedError("delong_pairwise")


# ---------------------------------------------------------------------------
# 3. Permutation importance
# ---------------------------------------------------------------------------


def permutation_importance_per_model(model_name: str, X_test, y_test, p_hat_baseline):
    """Per-feature mean AUC drop after shuffling that feature."""
    # TODO: use sklearn.inspection.permutation_importance
    # from sklearn.inspection import permutation_importance
    # result = permutation_importance(model, X_test, y_test, n_repeats=5,
    #                                  scoring='roc_auc', random_state=RANDOM_SEED)
    # return dict(sorted(zip(X_test.columns, result.importances_mean), key=lambda kv: -abs(kv[1])))
    raise NotImplementedError("permutation_importance_per_model")


# ---------------------------------------------------------------------------
# 4. Learning curves
# ---------------------------------------------------------------------------


def learning_curve_per_model(model_factory, X_train, y_train, g_train, X_test, y_test):
    """AUC vs train fraction."""
    # TODO: for each fraction in [0.1, 0.25, 0.5, 0.75, 1.0]:
    #   stratified subsample of train preserving group structure (market_id)
    #   refit, score on test, record AUC + n_train
    # Plot: x=n_train, y=test_auc, one line per model.
    raise NotImplementedError("learning_curve_per_model")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Stage 7.2 — Rigor additions (D-037)")
    print("=" * 60)
    raise NotImplementedError(
        "07_rigor_additions.py is a skeleton. Fill in the four sub-routines "
        "above before running. See the docstrings for sklearn one-liners."
    )


if __name__ == "__main__":
    main()
