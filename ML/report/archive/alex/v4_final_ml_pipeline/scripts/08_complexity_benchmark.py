"""
08_complexity_benchmark.py — Stage 7.3 of the v4 pipeline.

Required by the exam guidelines (`memories/uni/.../exam.md:47`):
  "Model complexity analysis (such as running time) compared to baseline model."

Measures, per model, three complexity dimensions:

  1. Fit time (wall-clock seconds for one fit on the full train set, n=1.11M)
  2. Predict time (seconds per 1,000 test rows, averaged over 5 runs)
  3. Parameter / size proxy:
       - LogReg / NaiveBayes: number of coefficients
       - Tree models: total leaf count across all trees
       - MLP: number of trainable Keras parameters

Outputs:
  outputs/rigor/complexity_benchmark.csv    one row per model
  outputs/rigor/complexity_benchmark.png    bar chart (fit time vs predict time)
  outputs/rigor/complexity_summary.json     pretty-printed summary

Methodology section in the report cites this table to compare RF vs HGBM vs
MLP wall-clock + memory tradeoffs.

STATUS: skeleton. Wall time when filled in: ~30 min total (most of it is
measuring fit times for 5+ models on M4 Pro).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from _common import DATA, ROOT

warnings.filterwarnings("ignore")

OUT = ROOT / "outputs" / "rigor"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
PREDICT_REPEATS = 5


def measure_one(name: str, model_factory, X_train, y_train, X_test):
    """Returns dict with fit_time_sec, predict_time_per_1k_sec, n_params_or_leaves."""
    # TODO: implement
    # t0 = time.time()
    # model = model_factory()
    # model.fit(X_train, y_train)
    # fit_time = time.time() - t0
    #
    # predict_times = []
    # for _ in range(PREDICT_REPEATS):
    #     t0 = time.time()
    #     model.predict_proba(X_test)
    #     predict_times.append((time.time() - t0) * 1000 / len(X_test))  # per 1k rows
    # predict_per_1k = float(np.median(predict_times))
    #
    # # Parameter proxy depends on model type:
    # n_params = ...
    #
    # return {"name": name, "fit_time_sec": fit_time, "predict_time_per_1k_sec": predict_per_1k, "n_params": n_params}
    raise NotImplementedError("measure_one")


def main():
    print("=" * 60)
    print("Stage 7.3 — Complexity benchmark (exam.md:47)")
    print("=" * 60)
    raise NotImplementedError(
        "08_complexity_benchmark.py is a skeleton. Fill in measure_one() "
        "above and a model factory dict in main(), then run."
    )


if __name__ == "__main__":
    main()
