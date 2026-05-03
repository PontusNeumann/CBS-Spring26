"""
05_optuna_tuning.py — Optuna TPE tuning for RF + HistGBM + LightGBM (D-037 rigor).

Spawns parallel workers (one per model). Each runs Optuna TPE with
MedianPruner over N trials, evaluating mean AUC over 5-fold GroupKFold
on market_id (preserves cross-regime transfer claim). Best params refit
on full train + predicted on test for tuned-vs-default comparison.

Outputs:
  alex/outputs/rigor/optuna/
    random_forest/
      study.db, study_history.csv, best_params.json, comparison.json,
      preds_test_tuned.npz
    hist_gbm/
      (same)
    lightgbm/
      (same — only if `lightgbm` is installed)

Wall-time on M4 Pro: ~1.5-2 hours, dominated by RF. Adding LightGBM costs
~30 min on top.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import time as _time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "rigor" / "optuna"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["random_forest", "hist_gbm", "lightgbm"],
        choices=["random_forest", "hist_gbm", "lightgbm"],
    )
    args = parser.parse_args()

    print("=" * 60)
    print(
        f"Optuna tuning — {len(args.models)} models × {args.n_trials} trials parallelised"
    )
    print("=" * 60)

    procs = {}
    log_handles = {}
    t0 = _time()
    for name in args.models:
        log_path = OUT / f"worker_{name}.log"
        log_handle = open(log_path, "w")
        log_handles[name] = log_handle
        p = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).parent / "_optuna_worker.py"),
                "--model",
                name,
                "--n_trials",
                str(args.n_trials),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        procs[name] = p
        print(f"  spawned {name} (PID {p.pid}, log: {log_path.name})")

    failed = []
    for name, p in procs.items():
        rc = p.wait()
        log_handles[name].close()
        if rc != 0:
            failed.append(name)
            print(f"  ✗ {name} failed (exit {rc}); see {OUT / f'worker_{name}.log'}")
        else:
            print(f"  ✓ {name} done")

    if failed:
        raise SystemExit(f"workers failed: {failed}")
    print(f"\n[done] all {len(args.models)} workers in {(_time() - t0) / 60:.1f} min")
    print(f"\noutputs in {OUT}")


if __name__ == "__main__":
    main()
