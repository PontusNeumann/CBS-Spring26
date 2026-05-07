"""
_vm_utils.py — Shared infrastructure for the VM-parallel sibling scripts.

Imported by 04_calibration_vm.py, 05_backtest_vm.py, 03_train_models_vm.py,
06_tuning_optuna_vm.py. The canonical (serial) scripts NEVER import this
module so they keep running unchanged on the laptop.

Why a separate module:
  - One place for BLAS thread caps (avoids 256x256 thread storms on the VM).
  - One place for deterministic per-work-unit seeds (joblib worker order
    is nondeterministic; we derive every random_state from a SeedSequence).
  - One place for the outputs_vm/ path so VM and laptop runs never collide.

Critical timing: cap_blas_threads() MUST be called before any numpy /
sklearn / lightgbm import in the calling process. Once those libs load,
they read OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS once
and never re-check. We set env vars in the joblib `initializer=` callback
which runs before the first task in each worker, and at the top of every
*_vm.py launcher before its own numpy import.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Default cap inside workers. Override in launcher scripts that need more
# (e.g. 03_train_models_vm uses 32 BLAS threads per outer worker).
DEFAULT_BLAS_PER_WORKER = 1


def cap_blas_threads(n: int = DEFAULT_BLAS_PER_WORKER) -> None:
    """Cap every common BLAS / OpenMP thread pool to n threads.

    Must be called before numpy / sklearn / lightgbm imports in this
    process. After those libs load, the env vars are ignored.
    """
    val = str(int(n))
    # what: every BLAS / OpenMP variant we might encounter
    # why: a single one being unset is enough to spawn 256 helper threads
    #      per worker on a 256-core box; that crushes the scheduler
    for var in (
        "OMP_NUM_THREADS",  # OpenMP (HistGBM, lightgbm core)
        "MKL_NUM_THREADS",  # Intel MKL
        "OPENBLAS_NUM_THREADS",  # OpenBLAS
        "BLIS_NUM_THREADS",  # BLIS
        "NUMEXPR_NUM_THREADS",  # numexpr
        "VECLIB_MAXIMUM_THREADS",  # macOS Accelerate (laptop only)
    ):
        os.environ[var] = val


def worker_init(blas_threads: int = DEFAULT_BLAS_PER_WORKER) -> None:
    """joblib `initializer=` callback. Runs once per worker, before any task."""
    cap_blas_threads(blas_threads)
    # cheap sanity check that the env vars actually stuck
    assert os.environ.get("OMP_NUM_THREADS") == str(blas_threads), (
        f"BLAS cap not set in worker (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})"
    )


def derive_seed(base_seed: int, *idx: int) -> int:
    """Deterministic 32-bit seed from (base_seed, idx_tuple).

    Use this everywhere a worker needs randomness. Workers run in
    nondeterministic order, so seeding from job index (not from a
    shared global rng) is the only way to keep results reproducible.
    """
    import numpy as np  # local import: keep module top numpy-free

    ss = np.random.SeedSequence(base_seed)
    if not idx:
        return int(ss.generate_state(1)[0])
    # spawn one child per index level; final state is deterministic
    children = ss.spawn(1)
    for i in idx:
        children = children[0].spawn(int(i) + 1)
    return int(children[-1].generate_state(1)[0])


def submission_root() -> Path:
    """Return the submission/ root, derived from this file's location."""
    return Path(__file__).resolve().parent.parent


def vm_paths() -> tuple[Path, Path]:
    """(DATA_DIR, OUTPUTS_VM_DIR). Mirrors config.py but with _vm suffix.

    OUTPUTS_VM_DIR is created if missing so VM runs never collide with
    the laptop reference outputs/.
    """
    sub = submission_root()
    data = sub / "data"
    out_vm = sub / "outputs_vm"
    out_vm.mkdir(parents=True, exist_ok=True)
    return data, out_vm


@contextmanager
def wall_clock_log(stage: str, log_path: Path | None = None):
    """Print and (optionally) record wall-clock seconds for a pipeline stage."""
    t0 = time.time()
    print(f"[{stage}] start")
    try:
        yield
    finally:
        elapsed = time.time() - t0
        print(f"[{stage}] done in {elapsed:.1f}s")
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            entries: dict[str, Any] = {}
            if log_path.exists():
                try:
                    entries = json.loads(log_path.read_text())
                except json.JSONDecodeError:
                    pass
            entries[stage] = round(elapsed, 1)
            log_path.write_text(json.dumps(entries, indent=2))


def detect_oversubscription(threshold_load_factor: float = 1.5) -> dict:
    """Heuristic load-average check. Returns warn flag, doesn't raise."""
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return {"warn": False, "reason": "psutil not installed"}
    cpus = psutil.cpu_count(logical=False) or 1
    load1 = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    return {
        "physical_cpus": cpus,
        "load_avg_1min": round(load1, 1),
        "warn": bool(load1 > threshold_load_factor * cpus),
    }


def assert_close(serial: dict, vm: dict, tol: dict[str, float]) -> list[str]:
    """Compare flat metric dicts. Returns list of failed keys (empty = pass).

    Used by tests/test_vm_parity.py to enforce the parity contract:
        AUC tol 1e-3, ROI tol 5e-3, paired-bootstrap CI tol 2e-2.
    """
    failures: list[str] = []
    for key, allowed in tol.items():
        sv = serial.get(key)
        vv = vm.get(key)
        if sv is None or vv is None:
            failures.append(f"{key}: missing (serial={sv}, vm={vv})")
            continue
        diff = abs(float(sv) - float(vv))
        if diff > allowed:
            failures.append(f"{key}: |{sv} - {vv}| = {diff:.4f} > tol {allowed}")
    return failures


def n_workers_default(unit: str = "cell") -> int:
    """Default n_jobs based on the parallel unit and machine size.

    'cell':  fine-grained, no inner BLAS  -> physical_cpus // 4 (cap 64)
    'model': coarse, 32 BLAS per worker  -> 8
    """
    try:
        import psutil  # noqa: PLC0415

        cpus = psutil.cpu_count(logical=False) or 8
    except ImportError:
        cpus = (os.cpu_count() or 8) // 2
    if unit == "cell":
        return min(64, max(1, cpus // 4))
    if unit == "model":
        return min(8, max(1, cpus // 32))
    raise ValueError(f"unknown unit: {unit!r}")
