"""
05_backtest_vm.py — VM-parallel sibling of 05_backtest.py.

Same outputs (sensitivity.csv, overview.png, falsification.json, diagnostics).
Only the 1,620-cell sensitivity grid is parallelized:

    9 models x 10 strategies x 3 capitals x 3 bet_pcts x 2 liquidity = 1,620 cells

Each cell runs the canonical realistic_backtest() unchanged. The function is
pure given inputs (no RNG), so parallel execution is bit-identical to serial.

Reuses the canonical script's helpers (compute_cost_and_edge, strategy_masks,
realistic_backtest, attach_backtest_context, all the diagnostics writers and
chart renderers) via importlib.

Run:
  python 05_backtest_vm.py [--n-workers 64] [--reference-mode]
"""

from __future__ import annotations

# CRITICAL: cap BLAS BEFORE numpy import in this driver process.
from _vm_utils import cap_blas_threads

cap_blas_threads(1)

import argparse
import importlib.util as _importutil
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _vm_utils import (  # noqa: E402
    cap_blas_threads as _cap_blas_threads,
    n_workers_default,
    vm_paths,
    wall_clock_log,
)
from config import RANDOM_SEED  # noqa: E402

TARGET = "bet_correct"


# ----------------------------------------------------------------------------
# Load canonical helpers
# ----------------------------------------------------------------------------


def _load_canonical():
    canonical_path = Path(__file__).resolve().parent / "05_backtest.py"
    if not canonical_path.exists():
        raise FileNotFoundError(f"canonical script missing: {canonical_path}")
    spec = _importutil.spec_from_file_location("backtest_canonical", canonical_path)
    mod = _importutil.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# Worker globals (populated by initializer; avoids re-pickling per task)
# ----------------------------------------------------------------------------

_WORKER_CACHE: dict = {}


def _init_backtest_worker(blas_threads: int, payload: dict) -> None:
    """joblib initializer: cap BLAS, stash shared arrays in module globals.

    Called once per worker process at pool creation. Subsequent task calls
    read from _WORKER_CACHE and only need to pickle the small cell tuple.
    """
    _cap_blas_threads(blas_threads)
    global _WORKER_CACHE
    _WORKER_CACHE.clear()
    _WORKER_CACHE.update(payload)
    # Lazy-load the canonical module here too — workers need realistic_backtest
    _WORKER_CACHE["canon"] = _load_canonical()


def _backtest_one(cell: tuple) -> dict:
    """Worker: run one (model, strategy, capital, bet_pct, ls) cell.

    realistic_backtest is deterministic given inputs, so no seed plumbing.
    """
    model_name, strategy, capital, bet_pct, ls = cell
    mask = _WORKER_CACHE["masks"][(model_name, strategy)]
    cost = _WORKER_CACHE["costs"][model_name]
    canon = _WORKER_CACHE["canon"]
    out = canon.realistic_backtest(
        mask,
        cost,
        _WORKER_CACHE["bet_correct"],
        _WORKER_CACHE["timestamps"],
        _WORKER_CACHE["market_ids"],
        _WORKER_CACHE["usd_amount"],
        _WORKER_CACHE["market_res_times"],
        initial_capital=capital,
        max_bet_pct_capital=bet_pct,
        liquidity_scaler=ls,
    )
    return {
        "model": model_name,
        "strategy": strategy,
        "initial_capital": capital,
        "max_bet_pct": bet_pct,
        "liquidity_scaler": ls,
        **out,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main(n_workers: int, reference_mode: bool) -> int:
    canon = _load_canonical()
    data_dir, outputs_vm = vm_paths()
    out_dir = outputs_vm / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use canonical model preds if outputs_vm/models is empty (Phase 3 not done yet)
    models_dir = outputs_vm / "models"
    if not models_dir.exists() or not any(models_dir.iterdir()):
        canonical_models = canon.OUTPUTS_DIR / "models"
        if canonical_models.exists():
            print(f"  using canonical model preds from {canonical_models}")
            models_dir = canonical_models
        else:
            print("  ERROR: no model predictions found. Run 03 + 04 first.")
            return 1

    n_jobs = 1 if reference_mode else n_workers
    print("=" * 60)
    print(f"Stage 5 VM — Parallel backtest sweep (n_jobs={n_jobs})")
    print("=" * 60)

    # what: load test cohort + predictions (matches canonical main())
    print("  loading test data and predictions ...")
    df = pd.read_parquet(data_dir / "consolidated_modeling_data.parquet")
    test = df[df["split"] == "test"].reset_index(drop=True).copy()
    test["market_id"] = test["market_id"].astype(str)
    test = canon.attach_backtest_context(test)

    # what: per-market resolve_ts (same logic as canonical)
    test["recovered_deadline"] = (
        test["timestamp"] + np.exp(test["log_time_to_deadline_hours"]) * 3600
    )
    per_market_resolve = (
        test.groupby("market_id")["recovered_deadline"]
        .median()
        .fillna(canon.CEASEFIRE_EVENT_UTC)
        .clip(upper=canon.CEASEFIRE_EVENT_UTC)
    )
    test["time_to_deadline"] = (
        test["market_id"].map(per_market_resolve).values
        - test["timestamp"].astype(float).values
    )
    market_res_times = {str(mid): int(ts) for mid, ts in per_market_resolve.items()}

    # what: gather calibrated predictions
    model_preds: dict[str, np.ndarray] = {}
    for d in sorted(models_dir.iterdir()):
        cal_path = d / "preds_test_cal.npz"
        if cal_path.exists():
            model_preds[d.name] = np.load(cal_path)["cal"]
    model_preds["naive_consensus"] = canon.naive_consensus_phat(test)
    print(f"  models in this run: {list(model_preds)}")

    # what: claim diagnostics (cheap; serial, same as canonical)
    canon.write_residual_edge_diagnostics(test, model_preds, out_dir)
    canon.write_consensus_diagnostics(test, model_preds, out_dir)
    canon.write_sell_semantics_diagnostics(test, model_preds, out_dir)
    print("  wrote residual-edge, consensus, and SELL-semantics diagnostics")

    # what: parameter grid (same as canonical)
    capitals = [1_000, 10_000, 100_000]
    bet_pcts = [0.01, 0.05, 0.10]
    liquidity_scalers = [1.0, 0.10]

    # ------- precompute per-(model, strategy) masks and per-model costs -------
    bet_correct = test[TARGET].astype(int).values
    timestamps = test["timestamp"].astype(float).values
    market_ids = test["market_id"].values
    usd_amount = test["usd_amount"].astype(float).values
    time_to_deadline = test["time_to_deadline"].values

    masks_dict: dict[tuple[str, str], np.ndarray] = {}
    costs_dict: dict[str, np.ndarray] = {}
    strategy_names: list[str] = []
    for model_name, p_hat in model_preds.items():
        cost, edge = canon.compute_cost_and_edge(test, p_hat)
        costs_dict[model_name] = cost
        masks = canon.strategy_masks(p_hat, edge, cost, time_to_deadline)
        if not strategy_names:
            strategy_names = list(masks.keys())
        for strat_name, mask in masks.items():
            masks_dict[(model_name, strat_name)] = mask.astype(bool)

    # ------- build cell list and run in parallel -------
    cells: list[tuple] = []
    for model_name in model_preds:
        for strat_name in strategy_names:
            for capital in capitals:
                for bet_pct in bet_pcts:
                    for ls in liquidity_scalers:
                        cells.append((model_name, strat_name, capital, bet_pct, ls))
    print(
        f"  {len(cells)} cells to evaluate "
        f"({len(model_preds)} models x {len(strategy_names)} strategies "
        f"x {len(capitals) * len(bet_pcts) * len(liquidity_scalers)} scenarios)"
    )

    payload = {
        "masks": masks_dict,
        "costs": costs_dict,
        "bet_correct": bet_correct,
        "timestamps": timestamps,
        "market_ids": market_ids,
        "usd_amount": usd_amount,
        "market_res_times": market_res_times,
    }

    with wall_clock_log(f"backtest_grid_n{n_jobs}", outputs_vm / "wall_clock.json"):
        if n_jobs == 1:
            # serial fallback (parity-test path) — populate cache locally
            _init_backtest_worker(1, payload)
            rows = [_backtest_one(c) for c in cells]
        else:
            rows = Parallel(
                n_jobs=n_jobs,
                backend="loky",
                prefer="processes",
                initializer=_init_backtest_worker,
                initargs=(1, payload),
                batch_size="auto",
            )(delayed(_backtest_one)(c) for c in cells)

    sens = pd.DataFrame(rows)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    print(f"  wrote {len(sens)} sensitivity rows -> sensitivity.csv")

    # ------- post-processing (cheap; serial via canonical helpers) -------
    canon.render_overview(sens, out_dir / "overview.png")
    print("  wrote overview.png")

    headline = sens[
        (sens["initial_capital"] == 10_000)
        & (sens["max_bet_pct"] == 0.05)
        & (sens["liquidity_scaler"] == 1.0)
    ]
    falsification: dict = {}
    for strat in headline["strategy"].unique():
        sub = headline[headline["strategy"] == strat]
        if not (sub["model"] == "naive_consensus").any():
            continue
        naive_roi = float(sub[sub["model"] == "naive_consensus"]["roi"].iloc[0])
        ml = sub[sub["model"] != "naive_consensus"].sort_values("roi", ascending=False)
        if ml.empty:
            continue
        best = ml.iloc[0]
        falsification[strat] = {
            "naive_roi": naive_roi,
            "best_model": best["model"],
            "best_model_roi": float(best["roi"]),
            "ml_beats_naive": bool(best["roi"] > naive_roi),
        }
    (out_dir / "falsification.json").write_text(json.dumps(falsification, indent=2))

    # ------- per-market PnL + edge buckets diagnostics (canonical helpers) -------
    ml_only = headline[headline["model"] != "naive_consensus"].sort_values(
        "roi", ascending=False
    )
    if not ml_only.empty:
        best_model = ml_only.iloc[0]["model"]
        p_hat_best = model_preds[best_model]
        cost_b, edge_b = canon.compute_cost_and_edge(test, p_hat_best)
        masks_b = canon.strategy_masks(p_hat_best, edge_b, cost_b, time_to_deadline)
        canon.per_market_pnl_breakdown(
            best_model,
            p_hat_best,
            edge_b,
            cost_b,
            bet_correct,
            market_ids,
            masks_b["general_ev"],
            out_dir / f"per_market_pnl_{best_model}.csv",
        )
        canon.plot_edge_distribution(
            edge_b,
            bet_correct,
            masks_b["top1pct_phat"],
            out_dir / f"edge_distribution_{best_model}.png",
        )

    # ------- parity summary -------
    parity_summary = {
        "n_jobs": n_jobs,
        "reference_mode": reference_mode,
        "n_cells": len(rows),
        "n_models": len(model_preds),
        "n_strategies": len(strategy_names),
        "headline_best_model_roi": (
            float(ml_only.iloc[0]["roi"]) if not ml_only.empty else None
        ),
    }
    (outputs_vm / "metrics" / "parity_05.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (outputs_vm / "metrics" / "parity_05.json").write_text(
        json.dumps(parity_summary, indent=2)
    )

    print(f"\nStage 5 VM complete. Outputs in {out_dir}.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=n_workers_default("cell"),
        help="joblib n_jobs for the cell sweep",
    )
    parser.add_argument(
        "--reference-mode",
        action="store_true",
        help="serial run (n_jobs=1) for parity baseline",
    )
    args = parser.parse_args()
    np.random.seed(RANDOM_SEED)
    sys.exit(main(args.n_workers, args.reference_mode))
