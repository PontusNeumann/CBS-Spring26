"""
09_shap_top_picks.py — Stage 7.4 of the v4 pipeline.

Maps to L14 Explainable AI in the CBS MLDP curriculum. Produces SHAP values
for the best model's top-1% picks on test, so the report can answer
"which features drive the model's confidence on its best predictions?"

Inputs:
  data/test_features_v4.parquet
  .scratch/preds/preds_<best_model>.npz   (from 03_sweep.py)

Outputs:
  outputs/rigor/shap/values.parquet         per-pick SHAP values + bet_correct
  outputs/rigor/shap/summary.png            global SHAP summary plot
  outputs/rigor/shap/top_picks.png          per-pick force/waterfall plots (sample)
  outputs/rigor/shap/feature_ranking.json   features sorted by mean(|SHAP|) on top-1%

Why top-1% only: SHAP on 257K test rows is expensive and the report's
"interpretability" claim only needs to defend the model's high-confidence
edge. Top-1% (~2,571 picks) is fast enough to compute exhaustively.

STATUS: skeleton. Wall time when filled in: ~1 hr total (15 min compute
on M4 Pro for tree-based SHAP + plotting).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from _common import DATA, ROOT, SCRATCH

warnings.filterwarnings("ignore")

OUT = ROOT / "outputs" / "rigor" / "shap"
OUT.mkdir(parents=True, exist_ok=True)

PRED_DIR = SCRATCH / "backtest"

# Auto-pick the best model from sweep results, or hardcode here
BEST_MODEL_DEFAULT = "random_forest"  # update to whichever wins on v4


def main():
    print("=" * 60)
    print("Stage 7.4 — SHAP on top-1% picks (L14 Explainable AI)")
    print("=" * 60)

    # TODO: implement
    # import shap
    #
    # test = pd.read_parquet(DATA / "test_features_v4.parquet")
    # fcols = json.loads((DATA / "feature_cols.json").read_text())
    # X_test = test[fcols].fillna(0).replace([np.inf, -np.inf], 0)
    #
    # # Load best model's calibrated preds
    # d = np.load(PRED_DIR / f"preds_{BEST_MODEL_DEFAULT}.npz")
    # p_hat = d["cal"]
    #
    # # Top-1% by p_hat
    # n = len(p_hat)
    # k = max(1, int(n * 0.01))
    # top_idx = np.argsort(p_hat)[-k:]
    #
    # # Refit best model (we need the actual estimator object for SHAP, not just preds)
    # # — or pickle the model in 03_sweep.py for SHAP loading. Either way:
    # model = ...  # load fitted estimator
    #
    # # Tree-based SHAP for RF/HGBM/LightGBM (fast). For MLP use KernelExplainer (slow).
    # if isinstance(model, (RandomForestClassifier, HistGradientBoostingClassifier)):
    #     explainer = shap.TreeExplainer(model)
    # else:
    #     explainer = shap.KernelExplainer(model.predict_proba, X_test.sample(100, random_state=42))
    #
    # shap_values = explainer.shap_values(X_test.iloc[top_idx])
    # # Save per-pick values, ranking, summary plot, force plots for sample
    # ...

    raise NotImplementedError(
        "09_shap_top_picks.py is a skeleton. Need to (1) pickle the best model "
        "in 03_sweep.py first so we can load it here, then (2) implement the "
        "SHAP computation + plots above. Tree SHAP is fast; KernelSHAP for MLP."
    )


if __name__ == "__main__":
    main()
