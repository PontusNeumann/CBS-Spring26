"""
04_calibration_vm.py — VM-parallel sibling of 04_calibration.py.

Same outputs, same metrics, same plots. Only difference: the two bootstrap
loops (per-model AUC CI, paired AUC differences) run in parallel via joblib.

Reuses the canonical script's helpers via importlib (the leading digit in
04_calibration.py forbids a normal `import`). Reimplements only:
  - bootstrap_auc_ci_parallel
  - paired_bootstrap_auc_diff_parallel
  - main()  (writes to outputs_vm/, calls the parallel bootstraps)

Determinism: the canonical script uses one shared rng per bootstrap call
(sequential). Workers can't share an rng, so we derive a per-iteration
seed from SeedSequence(RANDOM_SEED).spawn(idx). The result is reproducible
(same seed -> same numbers) but NOT bit-identical to serial. Test parity
contract: CI bounds within +-0.02 (well inside Monte-Carlo noise at 200-500
resamples).

Run:
  python 04_calibration_vm.py [--n-workers 64] [--reference-mode]

  --n-workers N      override joblib n_jobs (default: auto from CPU count)
  --reference-mode   force serial (n_jobs=1) for the parity baseline run
"""

from __future__ import annotations

# CRITICAL: cap BLAS BEFORE numpy / sklearn import. Otherwise the driver
# process spawns 256 OpenBLAS threads and they fight the worker pool.
from _vm_utils import cap_blas_threads

cap_blas_threads(1)

import argparse
import importlib.util as _importutil
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _vm_utils import (  # noqa: E402
    derive_seed,
    n_workers_default,
    vm_paths,
    wall_clock_log,
    worker_init,
)
from config import RANDOM_SEED  # noqa: E402

TARGET = "bet_correct"
N_BOOTSTRAP_AUC = 500  # per-model AUC CI iterations (canonical default)
N_BOOTSTRAP_PAIRED = 200  # paired-difference iterations (canonical default)


# ----------------------------------------------------------------------------
# Load canonical helpers (calibrate_one, plots, permutation_importance, shap)
# ----------------------------------------------------------------------------


def _load_canonical():
    """Dynamically load 04_calibration.py because of its digit-prefixed name."""
    canonical_path = Path(__file__).resolve().parent / "04_calibration.py"
    if not canonical_path.exists():
        raise FileNotFoundError(f"canonical script missing: {canonical_path}")
    spec = _importutil.spec_from_file_location("calibration_canonical", canonical_path)
    mod = _importutil.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# Worker functions (must be top-level for pickling)
# ----------------------------------------------------------------------------


def _auc_one_resample(
    idx: int, base_seed: int, y_true: np.ndarray, y_prob: np.ndarray
) -> float:
    """One bootstrap resample of test AUC. Returns nan on degenerate samples."""
    rng = np.random.default_rng(derive_seed(base_seed, idx))
    n = len(y_true)
    sample = rng.integers(0, n, size=n)
    if len(np.unique(y_true[sample])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[sample], y_prob[sample]))


def _paired_auc_diff_one_resample(
    idx: int, base_seed: int, y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray
) -> float:
    """One paired bootstrap resample of AUC(a) - AUC(b). NaN on degenerate."""
    rng = np.random.default_rng(derive_seed(base_seed, idx))
    n = len(y_true)
    sample = rng.integers(0, n, size=n)
    y_s = y_true[sample]
    if len(np.unique(y_s)) < 2:
        return float("nan")
    return float(roc_auc_score(y_s, p_a[sample]) - roc_auc_score(y_s, p_b[sample]))


# ----------------------------------------------------------------------------
# Parallel bootstrap driver
# ----------------------------------------------------------------------------


def bootstrap_auc_ci_parallel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iter: int = N_BOOTSTRAP_AUC,
    base_seed: int = RANDOM_SEED,
    n_jobs: int = 1,
) -> tuple[float, float, float]:
    """Parallel per-model AUC bootstrap CI. Returns (mean, ci_lo, ci_hi)."""
    if n_jobs == 1:
        # serial fallback (parity-test path)
        results = [
            _auc_one_resample(i, base_seed, y_true, y_prob) for i in range(n_iter)
        ]
    else:
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            prefer="processes",
            initializer=worker_init,
            initargs=(1,),
        )(
            delayed(_auc_one_resample)(i, base_seed, y_true, y_prob)
            for i in range(n_iter)
        )
    arr = np.array([r for r in results if not np.isnan(r)])
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


def paired_bootstrap_auc_diff_parallel(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_iter: int = N_BOOTSTRAP_PAIRED,
    base_seed: int = RANDOM_SEED,
    n_jobs: int = 1,
) -> dict:
    """Parallel paired-bootstrap AUC diff. Returns same dict shape as canonical."""
    if n_jobs == 1:
        diffs = [
            _paired_auc_diff_one_resample(i, base_seed, y_true, p_a, p_b)
            for i in range(n_iter)
        ]
    else:
        diffs = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            prefer="processes",
            initializer=worker_init,
            initargs=(1,),
        )(
            delayed(_paired_auc_diff_one_resample)(i, base_seed, y_true, p_a, p_b)
            for i in range(n_iter)
        )
    arr = np.array([d for d in diffs if not np.isnan(d)])
    return {
        "mean_diff": float(np.mean(arr)),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
        "p_value": float(2.0 * min((arr <= 0).mean(), (arr >= 0).mean())),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main(n_workers: int, reference_mode: bool) -> int:
    canon = _load_canonical()
    data_dir, outputs_vm = vm_paths()
    metrics_dir = outputs_vm / "metrics"
    models_dir = outputs_vm / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Use the canonical-pipeline outputs for predictions if outputs_vm/models
    # is empty (i.e. 03 hasn't been parallelized yet). The user can copy or
    # symlink the canonical outputs/models into outputs_vm/models for an
    # isolated run; for now fall through to the canonical location.
    if not models_dir.exists() or not any(models_dir.iterdir()):
        canonical_models = canon.OUTPUTS_DIR / "models"
        if canonical_models.exists():
            print(f"  using canonical model preds from {canonical_models}")
            models_dir = canonical_models
        else:
            print(f"  ERROR: no model predictions found. Run 03_train_models first.")
            return 1

    n_jobs = 1 if reference_mode else n_workers
    print("=" * 60)
    print(f"Stage 4 VM — Calibration + parallel bootstrap (n_jobs={n_jobs})")
    print("=" * 60)

    # Load labels
    df = pd.read_parquet(data_dir / "consolidated_modeling_data.parquet")
    y_train = df[df["split"] == "train"][TARGET].astype(int).reset_index(drop=True)
    y_test = df[df["split"] == "test"][TARGET].astype(int).reset_index(drop=True)

    model_names = sorted([p.name for p in models_dir.iterdir() if p.is_dir()])
    print(f"  found {len(model_names)} models: {model_names}")

    # ------- isotonic calibration (cheap; serial via canonical helper) -------
    cal_rows: list[dict] = []
    cal_preds: dict[str, np.ndarray] = {}
    with wall_clock_log("calibration", outputs_vm / "wall_clock.json"):
        for name in model_names:
            row = canon.calibrate_one(name, y_train, y_test, models_dir)
            if row is None:
                continue
            cal_rows.append(row)
            cal = np.load(models_dir / name / "preds_test_cal.npz")["cal"]
            raw = np.load(models_dir / name / "preds_test.npz")["raw"]
            canon.plot_reliability(
                y_test.values, raw, cal, name, metrics_dir / f"reliability_{name}.png"
            )
            cal_preds[name] = cal
        if cal_preds:
            canon.plot_reliability_combined(
                y_test.values, cal_preds, metrics_dir / "reliability_combined.png"
            )
    cal_df = pd.DataFrame(cal_rows).sort_values("test_auc_cal", ascending=False)
    cal_df.to_csv(metrics_dir / "calibration_summary.csv", index=False)

    # ------- per-model AUC bootstrap CI (parallel) -------
    with wall_clock_log("bootstrap_auc_ci", outputs_vm / "wall_clock.json"):
        ci_rows = []
        for name in model_names:
            cal_path = models_dir / name / "preds_test_cal.npz"
            if not cal_path.exists():
                continue
            cal = np.load(cal_path)["cal"]
            mean_auc, lo, hi = bootstrap_auc_ci_parallel(
                y_test.values,
                cal,
                n_iter=N_BOOTSTRAP_AUC,
                base_seed=RANDOM_SEED,
                n_jobs=n_jobs,
            )
            ci_rows.append(
                {
                    "model": name,
                    "test_auc_cal_mean": mean_auc,
                    "ci_lower_2_5": lo,
                    "ci_upper_97_5": hi,
                    "ci_width": hi - lo,
                }
            )
            print(f"  {name:14s}  AUC = {mean_auc:.4f}  CI = [{lo:.4f}, {hi:.4f}]")
        pd.DataFrame(ci_rows).to_csv(metrics_dir / "auc_bootstrap_ci.csv", index=False)

    # ------- paired bootstrap (parallel; biggest win) -------
    with wall_clock_log("paired_bootstrap", outputs_vm / "wall_clock.json"):
        cal_map: dict[str, np.ndarray] = {}
        for name in model_names:
            cp = models_dir / name / "preds_test_cal.npz"
            if cp.exists():
                cal_map[name] = np.load(cp)["cal"]
        ordered = sorted(cal_map)
        pairs = [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1 :]]
        n_pairs = len(pairs)
        pair_rows = []
        for m_a, m_b in pairs:
            res = paired_bootstrap_auc_diff_parallel(
                y_test.values,
                cal_map[m_a],
                cal_map[m_b],
                n_iter=N_BOOTSTRAP_PAIRED,
                base_seed=RANDOM_SEED,
                n_jobs=n_jobs,
            )
            bonf = min(res["p_value"] * n_pairs, 1.0)
            pair_rows.append(
                {
                    "model_a": m_a,
                    "model_b": m_b,
                    "mean_auc_diff": res["mean_diff"],
                    "ci_lower": res["ci_lower"],
                    "ci_upper": res["ci_upper"],
                    "p_value": res["p_value"],
                    "p_value_bonferroni": bonf,
                }
            )
        if pair_rows:
            pair_df = pd.DataFrame(pair_rows)
            pair_df.to_csv(metrics_dir / "auc_pairwise.csv", index=False)
            top5 = (
                pair_df.assign(abs_diff=pair_df["mean_auc_diff"].abs())
                .sort_values("abs_diff", ascending=False)
                .head(5)
                .drop(columns="abs_diff")
            )
            print("\nTop-5 pairs by |mean AUC diff|:")
            print(top5.to_string(index=False))

    # ------- permutation importance + SHAP (cheap; canonical serial) -------
    with wall_clock_log("permutation_and_shap", outputs_vm / "wall_clock.json"):
        if not cal_df.empty:
            best = cal_df.iloc[0]["model"]
            print(f"\nBest model by calibrated AUC: {best}")
            imp = canon.permutation_importance_top_k(best)
            if imp is not None:
                imp.to_csv(
                    metrics_dir / f"permutation_importance_{best}.csv", index=False
                )
            canon.shap_on_top_picks(best)

    # Write a small summary the parity test can read
    parity_summary = {
        "n_models": len(model_names),
        "n_jobs": n_jobs,
        "reference_mode": reference_mode,
    }
    if not cal_df.empty:
        parity_summary["best_model"] = str(cal_df.iloc[0]["model"])
        parity_summary["best_test_auc_cal"] = float(cal_df.iloc[0]["test_auc_cal"])
    (outputs_vm / "metrics" / "parity_04.json").write_text(
        json.dumps(parity_summary, indent=2)
    )

    print(
        f"\nStage 4 VM complete. Outputs in {outputs_vm.relative_to(outputs_vm.parent.parent)}/."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=n_workers_default("cell"),
        help="joblib n_jobs for the bootstrap loops",
    )
    parser.add_argument(
        "--reference-mode",
        action="store_true",
        help="serial run (n_jobs=1) for parity baseline",
    )
    args = parser.parse_args()
    np.random.seed(RANDOM_SEED)
    sys.exit(main(args.n_workers, args.reference_mode))
