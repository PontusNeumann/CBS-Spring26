"""
10_backtest.py

Economic evaluation of v3.5 models. Answers the actual research question:
"Can we gain an advantage by capturing asymmetric information in trades?"

Tier 1 (must-have):
  1. PnL backtest — general +EV (edge > 0.02) and home-run (edge > 0.20 + late + cheap)
  2. Edge-based top-k ranking (sort by |edge|, not by p_hat)
  3. Baseline comparisons (4 baselines)

Tier 2 (strong secondary):
  4. Magamyman explicit filter
  5. Cutoff-date sweep
  6. Contrarian-only PnL
  7. (Wallet-level PnL — basic version with wallet IDs available)

Tier 3 (quick wins):
  8. Calibration diagram
  9. Per-market PnL
  10. Threshold sweep

PnL math:
  cost_per_token = the price the trader paid to bet on their chosen side
    BUY YES at p_yes:    cost = p_yes
    BUY NO at (1-p_yes): cost = 1 - p_yes
    SELL YES at p_yes:   cost = 1 - p_yes  (equivalent to BUY NO)
    SELL NO at (1-p_yes):cost = p_yes      (equivalent to BUY YES)
  edge = p_hat - cost_per_token  (model's view minus market-implied prob)
  PnL per $1 staked: + (1-cost)/cost if bet_correct, -1 otherwise.

Outputs:
  alex/outputs/backtest/
    summary.json
    pnl_curves.png
    edge_distribution.png
    calibration.png
    per_market_pnl.json
    baselines_comparison.csv
    home_run_picks.parquet
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from _common import (
    DATA,
    ROOT,
    compute_cost_and_edge,
    compute_pre_yes_price_corrected,
    general_ev_rule,
    home_run_rule,
)

warnings.filterwarnings("ignore")

OUT = ROOT / "outputs" / "backtest"
OUT.mkdir(parents=True, exist_ok=True)

STAKE = 100.0
RANDOM_SEED = 42
N_FOLDS_CALIBRATION = 5
COST_FLOOR_RAW = 0.001  # raw economic math; realism uses 0.05 (see _common)


# ---------------------------------------------------------------------------
# PnL math (raw, no realism)
# ---------------------------------------------------------------------------


def pnl_per_trade(
    cost: np.ndarray, bet_correct: np.ndarray, stake: float = STAKE
) -> np.ndarray:
    """PnL per trade with $stake bet."""
    return np.where(bet_correct == 1, stake * (1 - cost) / cost, -stake)


# ---------------------------------------------------------------------------
# 10-specific filter — kept local because it's only used here
# ---------------------------------------------------------------------------


def magamyman_filter(
    cost: np.ndarray,
    p_hat: np.ndarray,
    side_buy: np.ndarray,
    outcome_yes: np.ndarray,
    time_to_deadline_sec: np.ndarray,
) -> np.ndarray:
    """BUY YES at low price + high model confidence + late."""
    return (
        (cost < 0.30)
        & (p_hat > 0.7)
        & (side_buy == 1)
        & (outcome_yes == 1)
        & (time_to_deadline_sec < 6 * 3600)
    )


# ---------------------------------------------------------------------------
# Train + calibrate models
# ---------------------------------------------------------------------------


def cv_oof_for_calibration(make_estimator, X, y, groups, scale=True):
    gkf = GroupKFold(n_splits=N_FOLDS_CALIBRATION)
    oof = np.zeros(len(y), dtype=float)
    for tr, va in gkf.split(X, y, groups):
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X.iloc[tr])
            X_va = sc.transform(X.iloc[va])
        else:
            X_tr = X.iloc[tr].values
            X_va = X.iloc[va].values
        clf = make_estimator()
        clf.fit(X_tr, y.iloc[tr])
        oof[va] = clf.predict_proba(X_va)[:, 1]
    return oof


def train_and_predict(X_train, y_train, g_train, X_test, factory, scale=True):
    """Returns calibrated test predictions + isotonic calibrator."""
    oof = cv_oof_for_calibration(factory, X_train, y_train, g_train, scale=scale)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof, y_train)

    if scale:
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train)
        X_te_s = sc.transform(X_test)
    else:
        X_tr_s = X_train.values
        X_te_s = X_test.values
    final = factory()
    final.fit(X_tr_s, y_train)
    raw_test = final.predict_proba(X_te_s)[:, 1]
    cal_test = cal.transform(raw_test)
    return final, cal, raw_test, cal_test, oof


# ---------------------------------------------------------------------------
# Per-market and per-bucket PnL
# ---------------------------------------------------------------------------


def evaluate_strategy(
    name: str,
    mask: np.ndarray,
    cost: np.ndarray,
    bet_correct: np.ndarray,
    market_id: pd.Series,
    edge: np.ndarray,
) -> dict:
    if mask.sum() == 0:
        return {
            "name": name,
            "n_trades": 0,
            "hit_rate": None,
            "total_pnl": 0.0,
            "mean_pnl_per_trade": 0.0,
            "sharpe": None,
            "max_drawdown": 0.0,
            "n_markets": 0,
            "mean_edge": None,
            "median_cost": None,
        }
    pnl = pnl_per_trade(cost[mask], bet_correct[mask])
    cum = np.cumsum(pnl)
    drawdown = (np.maximum.accumulate(cum) - cum).max()
    return {
        "name": name,
        "n_trades": int(mask.sum()),
        "hit_rate": float(bet_correct[mask].mean()),
        "total_pnl": float(pnl.sum()),
        "mean_pnl_per_trade": float(pnl.mean()),
        "sharpe": float(pnl.mean() / pnl.std()) if pnl.std() > 0 else None,
        "max_drawdown": float(drawdown),
        "n_markets": int(market_id[mask].nunique()),
        "mean_edge": float(edge[mask].mean()),
        "median_cost": float(np.median(cost[mask])),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def random_baseline(n: int, n_select: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    idx = rng.choice(n, size=n_select, replace=False)
    mask[idx] = True
    return mask


def follow_crowd_mask(
    side_buy: np.ndarray, outcome_yes: np.ndarray, pre_trade_price: np.ndarray
) -> np.ndarray:
    """Bet only when trader's bet matches what market consensus suggests."""
    trader_side_wins_yes = side_buy * outcome_yes + (1 - side_buy) * (1 - outcome_yes)
    market_thinks_yes_wins = pre_trade_price > 0.5
    return trader_side_wins_yes == market_thinks_yes_wins.astype(int)


def contrarian_mask(
    side_buy: np.ndarray, outcome_yes: np.ndarray, pre_trade_price: np.ndarray
) -> np.ndarray:
    """Trader bets against market consensus."""
    return ~follow_crowd_mask(side_buy, outcome_yes, pre_trade_price)


def buy_majority_mask(
    pre_trade_price: np.ndarray, outcome_yes: np.ndarray, side_buy: np.ndarray
) -> np.ndarray:
    """Always bet on the side market favors. Trader's trade matches majority side AND it's a buy."""
    trader_side_wins_yes = side_buy * outcome_yes + (1 - side_buy) * (1 - outcome_yes)
    market_thinks_yes_wins = pre_trade_price > 0.5
    return trader_side_wins_yes == market_thinks_yes_wins.astype(int)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_pnl_curves(strategies: dict, path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in strategies.items():
        if data["pnl_series"] is None:
            continue
        ax.plot(
            data["pnl_series"],
            label=f"{name} (n={data['n_trades']}, total ${data['total_pnl']:.0f})",
        )
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Trade index (chronological)")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Cumulative PnL by strategy — test cohort")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_edge_distribution(
    edge: np.ndarray, bet_correct: np.ndarray, mask_top_k: np.ndarray, path: Path
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Left: full edge distribution with top-k overlay
    axes[0].hist(edge, bins=80, alpha=0.5, label="all test trades", color="grey")
    axes[0].hist(
        edge[mask_top_k], bins=80, alpha=0.7, label="top-1% by p_hat", color="C1"
    )
    axes[0].axvline(0, color="black", lw=0.5)
    axes[0].axvline(
        0.02, color="green", lw=0.5, ls="--", label="general +EV threshold (0.02)"
    )
    axes[0].axvline(
        0.20, color="red", lw=0.5, ls="--", label="home-run threshold (0.20)"
    )
    axes[0].set_xlabel("edge = p_hat − pre_trade_price")
    axes[0].set_ylabel("trade count")
    axes[0].set_title("Edge distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # Right: hit rate by edge bucket
    bucket_edges = [-1, -0.2, -0.05, 0, 0.02, 0.05, 0.10, 0.20, 0.50, 1]
    labels = [f"{a:.2f}-{b:.2f}" for a, b in zip(bucket_edges[:-1], bucket_edges[1:])]
    bucket = np.digitize(edge, bucket_edges[1:-1], right=False)
    rates = []
    counts = []
    for b in range(len(labels)):
        mask = bucket == b
        rates.append(bet_correct[mask].mean() if mask.any() else 0)
        counts.append(mask.sum())
    bars = axes[1].bar(
        labels, rates, color=["red" if r < 0.5 else "green" for r in rates]
    )
    axes[1].axhline(0.5, color="black", ls="--", lw=0.5)
    axes[1].set_xlabel("edge bucket")
    axes[1].set_ylabel("empirical hit rate")
    axes[1].set_title("Hit rate by edge bucket — does higher edge = more wins?")
    for bar, c in zip(bars, counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            0.02,
            f"n={c}",
            ha="center",
            fontsize=8,
            rotation=90,
        )
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_calibration(p_hat: np.ndarray, bet_correct: np.ndarray, path: Path, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p_hat, bins[1:-1], right=False)
    confs, accs, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            confs.append(p_hat[m].mean())
            accs.append(bet_correct[m].mean())
            ns.append(m.sum())
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    sizes = np.array(ns) / max(ns) * 200 + 20
    ax.scatter(confs, accs, s=sizes, alpha=0.6)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical hit rate")
    ax.set_title("Calibration diagram (test) — bubble size = trade count")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("v3.5 economic backtest")
    print("=" * 60)
    train = pd.read_parquet(DATA / "train_features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    fcols = json.loads((DATA / "feature_cols.json").read_text())

    # Sort test by (market_id, timestamp) and remember original row indices so
    # we can reorder cached predictions (workers save in the on-disk file order).
    test["market_id"] = test["market_id"].astype(str)
    test["_orig_idx"] = np.arange(len(test))
    test = test.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    sort_key = test[
        "_orig_idx"
    ].values  # sort_key[i] = original idx now at sorted pos i

    # Compute corrected pre_yes_price (per-token-price bug fix). Same algorithm
    # as 11_realistic_backtest.py::compute_pre_yes_price_corrected. Aligns to the
    # sorted `test` order because we sort test_raw the same way inside the fn.
    test_raw = pd.read_parquet(DATA / "test.parquet")
    test["pre_yes_price_corrected"] = compute_pre_yes_price_corrected(test_raw)
    print(
        f"[fix] corrected pre_yes mean {test['pre_yes_price_corrected'].mean():.3f} "
        f"vs raw pre_trade_price mean {test['pre_trade_price'].mean():.3f}"
    )

    X_train = train[fcols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train["bet_correct"].astype(int)
    g_train = train["market_id"]
    X_test = test[fcols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test["bet_correct"].astype(int)
    n_test = len(X_test)

    print(f"train: {len(X_train):,} × {X_train.shape[1]}, test: {n_test:,}")
    print(f"test base rate: {y_test.mean():.4f}\n")

    # --- precompute test-side fields used everywhere
    # Use corrected pre_yes for baselines so consensus/contrarian masks reflect
    # actual YES probability, not whatever per-token price preceded the trade.
    pre_trade_price = test["pre_yes_price_corrected"].values
    side_buy = test["side_buy"].values
    outcome_yes = test["outcome_yes"].values
    time_to_deadline_sec = (
        np.exp(test["log_time_to_deadline_hours"].values) - 1
    )  # hours, but log_time
    time_to_deadline_sec = time_to_deadline_sec * 3600
    market_id = test["market_id"]
    bet_correct = y_test.values

    # ---- Train models in parallel via subprocess pool ------------------
    # Each worker (alex/scripts/_backtest_worker.py) loads parquets independently,
    # trains one model with 5-fold CV + final fit + isotonic, saves preds as npz.
    # n_jobs=4 per worker × 3 workers = 12 cores (M4 Pro friendly).
    import subprocess
    import sys
    from time import time as _time

    SCRATCH = ROOT / ".scratch" / "backtest"
    SCRATCH.mkdir(parents=True, exist_ok=True)

    model_names = ["logreg_l2", "random_forest", "hist_gbm"]
    pred_paths = {name: SCRATCH / f"preds_{name}.npz" for name in model_names}

    # Workers are deterministic (random_state=42). If all preds exist, skip
    # retraining unless RETRAIN=1 forces a fresh run.
    import os

    force_retrain = os.environ.get("RETRAIN") == "1"
    all_cached = all(p.exists() for p in pred_paths.values())
    if all_cached and not force_retrain:
        print(
            "\n[cache] all 3 preds_*.npz present — skipping worker training "
            "(set RETRAIN=1 to force)."
        )
    else:
        print(f"\n[parallel] spawning {len(model_names)} workers...")
        t0 = _time()
        procs = {}
        for name in model_names:
            out_path = pred_paths[name]
            log_path = SCRATCH / f"worker_{name}.log"
            log_handle = open(log_path, "w")
            p = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).parent / "_backtest_worker.py"),
                    "--model",
                    name,
                    "--out",
                    str(out_path),
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            procs[name] = (p, log_handle)
            print(f"  spawned {name} (PID {p.pid}, log: {log_path.name})")

        failed = []
        for name, (p, log_handle) in procs.items():
            rc = p.wait()
            log_handle.close()
            if rc != 0:
                failed.append(name)
                print(f"  ✗ {name} failed (exit {rc}); see {SCRATCH}/worker_{name}.log")
            else:
                print(f"  ✓ {name} done")
        if failed:
            raise RuntimeError(f"workers failed: {failed}")
        print(f"[parallel] all 3 workers done in {_time() - t0:.0f}s\n")

    # ---- Load predictions from disk ------------------------------------
    # Workers save preds in the on-disk test_features.parquet row order. We
    # sorted `test` by (market_id, timestamp) above, so reorder cached preds
    # via sort_key to keep everything aligned. `oof` lives on the train rows
    # and doesn't need reordering.
    model_preds = {}
    for name in model_names:
        d = np.load(pred_paths[name])
        raw, cal, oof = d["raw"][sort_key], d["cal"][sort_key], d["oof"]
        cost, edge, _ = compute_cost_and_edge(test, cal, cost_floor=COST_FLOOR_RAW)
        model_preds[name] = {
            "p_hat_raw": raw,
            "p_hat_cal": cal,
            "edge": edge,
            "cost": cost,
            "oof": oof,
        }
        print(
            f"  [{name}] mean p_hat (cal): {cal.mean():.3f}, "
            f"acc@0.5: {((cal > 0.5) == bet_correct).mean():.3f}"
        )

    # ===================================================================
    # Strategy evaluation per model
    # ===================================================================
    all_results = {}
    for model_name, preds in model_preds.items():
        edge = preds["edge"]
        cost = preds["cost"]
        p_hat = preds["p_hat_cal"]

        strategies = {}

        # Tier 1: trading rules
        m_general = general_ev_rule(edge, edge_min=0.02)
        m_home_run = home_run_rule(edge, cost, time_to_deadline_sec, edge_min=0.20)
        m_top_1pct_phat = np.zeros(n_test, dtype=bool)
        m_top_1pct_phat[np.argsort(p_hat)[-int(n_test * 0.01) :]] = True
        m_top_1pct_edge = np.zeros(n_test, dtype=bool)
        m_top_1pct_edge[np.argsort(edge)[-int(n_test * 0.01) :]] = True
        m_magamyman = magamyman_filter(
            cost, p_hat, side_buy, outcome_yes, time_to_deadline_sec
        )
        m_contrarian_passing = m_general & contrarian_mask(
            side_buy, outcome_yes, pre_trade_price
        )

        strategies["general_ev_edge>0.02"] = m_general
        strategies["home_run_edge>0.20_late_cheap"] = m_home_run
        strategies["top_1pct_by_p_hat"] = m_top_1pct_phat
        strategies["top_1pct_by_edge"] = m_top_1pct_edge
        strategies["magamyman_filter"] = m_magamyman
        strategies["contrarian_passing_general"] = m_contrarian_passing

        # Tier 1: baselines
        strategies["baseline_random_1pct"] = random_baseline(n_test, int(n_test * 0.01))
        strategies["baseline_follow_crowd_all"] = follow_crowd_mask(
            side_buy, outcome_yes, pre_trade_price
        )
        strategies["baseline_buy_every_trade"] = np.ones(n_test, dtype=bool)
        strategies["baseline_contrarian_all"] = contrarian_mask(
            side_buy, outcome_yes, pre_trade_price
        )

        # Evaluate
        results = {}
        for s_name, mask in strategies.items():
            r = evaluate_strategy(s_name, mask, cost, bet_correct, market_id, edge)
            # Add cumulative pnl series for the plot
            if mask.sum() > 0:
                pnl = pnl_per_trade(cost[mask], bet_correct[mask])
                # Order by timestamp for cumulative curve
                ts_order = np.argsort(test.loc[mask, "timestamp"].values)
                r["pnl_series"] = np.cumsum(pnl[ts_order]).tolist()
            else:
                r["pnl_series"] = None
            results[s_name] = r
        all_results[model_name] = results

        # Print compact table for this model
        print(f"\n=== {model_name} strategies ===")
        print(
            f"{'strategy':<36} {'n':>7} {'mkts':>5} {'hit':>6} {'total $':>10} {'mean $':>8} {'mean edge':>9}"
        )
        for s_name, r in results.items():
            hit = f"{r['hit_rate']:.3f}" if r["hit_rate"] is not None else "—"
            edge_str = f"{r['mean_edge']:+.3f}" if r["mean_edge"] is not None else "—"
            print(
                f"  {s_name:<34} {r['n_trades']:>7} {r['n_markets']:>5} {hit:>6} ${r['total_pnl']:>+9.0f} ${r['mean_pnl_per_trade']:>+7.2f} {edge_str:>9}"
            )

    # ===================================================================
    # Tier 2: cutoff-date sweep on home-run for best model
    # ===================================================================
    best_model = "random_forest"  # the winner from sweep
    print(f"\n=== cutoff-date sweep — home-run on {best_model} ===")
    preds = model_preds[best_model]
    edge = preds["edge"]
    cost = preds["cost"]
    cutoff_results = []
    for n_days in [14, 7, 3, 1]:
        time_max_sec = n_days * 24 * 3600
        m = (edge > 0.20) & (cost < 0.30) & (time_to_deadline_sec < time_max_sec)
        r = evaluate_strategy(
            f"cutoff_{n_days}d", m, cost, bet_correct, market_id, edge
        )
        cutoff_results.append({"n_days": n_days, **r})
        print(
            f"  ≤{n_days}d: n={r['n_trades']:>5}, hit={r['hit_rate']}, total ${r['total_pnl']:+.0f}, mean edge {r['mean_edge']}"
        )

    # ===================================================================
    # Tier 3: per-market PnL
    # ===================================================================
    print(f"\n=== per-market PnL — {best_model}, general +EV rule ===")
    edge = preds["edge"]
    cost = preds["cost"]
    m_general = general_ev_rule(edge)
    pm_data = []
    for mid in test["market_id"].unique():
        mid_mask = (test["market_id"] == mid).values & m_general
        if mid_mask.sum() == 0:
            continue
        pnl = pnl_per_trade(cost[mid_mask], bet_correct[mid_mask])
        pm_data.append(
            {
                "market_id": str(mid),
                "n_trades": int(mid_mask.sum()),
                "hit_rate": float(bet_correct[mid_mask].mean()),
                "total_pnl": float(pnl.sum()),
                "mean_edge": float(edge[mid_mask].mean()),
            }
        )
    pm_df = pd.DataFrame(pm_data).sort_values("total_pnl", ascending=False)
    print(pm_df.to_string(index=False))
    pm_df.to_csv(OUT / "per_market_pnl.csv", index=False)

    # ===================================================================
    # Tier 3: threshold sweep on best model
    # ===================================================================
    print(f"\n=== threshold sweep — {best_model}, p_hat threshold ===")
    p_hat = preds["p_hat_cal"]
    cost = preds["cost"]
    thresh_results = []
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        m = p_hat > thr
        r = evaluate_strategy(
            f"p_hat>{thr}", m, cost, bet_correct, market_id, preds["edge"]
        )
        thresh_results.append({"threshold": thr, **r})
        print(
            f"  p_hat>{thr}: n={r['n_trades']:>6}, hit={r['hit_rate']}, total ${r['total_pnl']:+.0f}, mean edge {r['mean_edge']}"
        )

    # ===================================================================
    # Plots
    # ===================================================================
    print(f"\n=== generating plots ===")
    # PnL curves for the best model's key strategies
    plot_data = {
        f"general_ev (n={all_results[best_model]['general_ev_edge>0.02']['n_trades']})": all_results[
            best_model
        ]["general_ev_edge>0.02"],
        f"home_run (n={all_results[best_model]['home_run_edge>0.20_late_cheap']['n_trades']})": all_results[
            best_model
        ]["home_run_edge>0.20_late_cheap"],
        f"top1%_p_hat": all_results[best_model]["top_1pct_by_p_hat"],
        f"top1%_edge": all_results[best_model]["top_1pct_by_edge"],
        f"baseline_buy_all": all_results[best_model]["baseline_buy_every_trade"],
        f"baseline_random_1pct": all_results[best_model]["baseline_random_1pct"],
    }
    plot_pnl_curves(plot_data, OUT / "pnl_curves.png")
    print(f"  -> {OUT / 'pnl_curves.png'}")

    # Edge distribution
    edge = preds["edge"]
    top_1pct_phat_mask = np.zeros(n_test, dtype=bool)
    top_1pct_phat_mask[np.argsort(preds["p_hat_cal"])[-int(n_test * 0.01) :]] = True
    plot_edge_distribution(
        edge, bet_correct, top_1pct_phat_mask, OUT / "edge_distribution.png"
    )
    print(f"  -> {OUT / 'edge_distribution.png'}")

    # Calibration
    plot_calibration(preds["p_hat_cal"], bet_correct, OUT / "calibration.png")
    print(f"  -> {OUT / 'calibration.png'}")

    # ===================================================================
    # Save summary
    # ===================================================================
    summary = {
        "models": list(model_preds.keys()),
        "best_model": best_model,
        "test_n": int(n_test),
        "test_base_rate": float(bet_correct.mean()),
        "stake_per_signal_usd": STAKE,
        "all_results": {
            mn: {
                sn: {k: v for k, v in r.items() if k != "pnl_series"}
                for sn, r in res.items()
            }
            for mn, res in all_results.items()
        },
        "cutoff_sweep": cutoff_results,
        "threshold_sweep": thresh_results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Save home-run picks for inspection
    edge = preds["edge"]
    cost = preds["cost"]
    p_hat = preds["p_hat_cal"]
    hr_mask = home_run_rule(edge, cost, time_to_deadline_sec)
    if hr_mask.sum() > 0:
        hr_picks = test[hr_mask].copy()
        hr_picks["p_hat"] = p_hat[hr_mask]
        hr_picks["edge"] = edge[hr_mask]
        hr_picks["cost"] = cost[hr_mask]
        hr_picks["pnl"] = pnl_per_trade(cost[hr_mask], bet_correct[hr_mask])
        hr_picks.to_parquet(OUT / "home_run_picks.parquet", index=False)
        print(f"  -> {OUT / 'home_run_picks.parquet'} ({hr_mask.sum()} picks)")

    print("\n" + "=" * 60)
    print(f"DONE — outputs in {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
