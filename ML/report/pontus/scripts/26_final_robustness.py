"""
26_final_robustness.py

Two final robustness deliverables that round out the Option-2 writeup:

  A. **Slippage-adjusted backtest.** Re-runs the home-run and
     general +EV trading rules on the embedding-MLP's test predictions
     with a configurable fill-slippage haircut applied to
     `market_implied_prob` on both the follow and inverse sides.
     Reports PnL at 0 %, 2 %, 3 %, and 5 % slippage.

  B. **Per-market temporal split.** For each of the 74 markets, split
     its §4-filtered trades into first-70 % / middle-15 % / last-15 %
     by `timestamp`, pool all per-market first-70 % slices into a
     combined within-market train, fit LogReg + embedding MLP, test on
     the combined last-15 % slice. Reports per-market test ROC plus
     the cross-market distribution.

Outputs → `pontus/outputs/v2_final_robustness/`:
  slippage/backtest.json
  per_market_temporal/summary.json, per_market_rocs.csv
  summary.json

Framework: tf.keras + sklearn (CBS MLDP rubric).

Usage:
  caffeinate -i python pontus/scripts/26_final_robustness.py \\
      2>&1 | tee pontus/outputs/v2_final_robustness/run.log
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[2]

# Import V1 (preprocess helpers) and V2-embedding model builder
V1_PATH = ROOT / "pontus" / "scripts" / "21_full_pipeline.py"
_s1 = importlib.util.spec_from_file_location("pipeline_v1", V1_PATH)
V1 = importlib.util.module_from_spec(_s1)
sys.modules["pipeline_v1"] = V1
_s1.loader.exec_module(V1)

V25_PATH = ROOT / "pontus" / "scripts" / "25_v2_market_embedding.py"
_s25 = importlib.util.spec_from_file_location("v2emb", V25_PATH)
V25 = importlib.util.module_from_spec(_s25)
sys.modules["v2emb"] = V25
_s25.loader.exec_module(V25)

OUT_DIR = ROOT / "pontus" / "outputs" / "v2_final_robustness"
(OUT_DIR / "slippage").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "per_market_temporal").mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# A. Slippage-adjusted backtest
# ---------------------------------------------------------------------------
def slippage_backtest(
    df_te: pd.DataFrame,
    p_te: np.ndarray,
    mip_te: np.ndarray,
    y_te: np.ndarray,
    *,
    edge_threshold: float,
    ttd_max_sec: float | None = None,
    mip_max: float | None = None,
    stake_usd: float = 100.0,
    slippage_frac: float = 0.0,
    mip_floor: float = 0.05,
) -> dict:
    """Flat-stake backtest with per-fill slippage.

    Slippage model: when buying shares at implied price p, actual fill is
    at p * (1 + slippage_frac); payoff ratio therefore becomes
    1 / (p * (1 + slippage_frac)) rather than 1 / p. Same on the inverse
    side with (1 - p).

    `mip_floor` clips the effective price away from zero to prevent
    long-shot payoffs from blowing up. Matches V1.backtest_rule's
    mip-clipped convention.
    """
    edge = p_te - mip_te
    deadline = pd.to_datetime(df_te[V1.DEADLINE_COL], utc=True, errors="coerce")
    ts = pd.to_datetime(df_te[V1.TIME_COL], utc=True, errors="coerce")
    ttd = (deadline - ts).dt.total_seconds().to_numpy()

    gate_follow = edge > edge_threshold
    gate_inverse = edge < -edge_threshold
    gate = gate_follow | gate_inverse
    if ttd_max_sec is not None:
        gate &= ttd < ttd_max_sec
    if mip_max is not None:
        gate &= mip_te < mip_max

    eff_mip = np.clip(mip_te * (1 + slippage_frac), mip_floor, 1 - mip_floor)
    pnl = np.zeros(len(df_te), dtype=np.float64)

    # Follow
    m = gate_follow & gate
    if m.any():
        pnl[m] = stake_usd * (y_te[m] / eff_mip[m] - 1.0)
    # Inverse side: pay (1 - mip)*(1 + slippage)
    eff_inv = np.clip((1 - mip_te) * (1 + slippage_frac), mip_floor, 1 - mip_floor)
    m = gate_inverse & gate
    if m.any():
        pnl[m] = stake_usd * ((1 - y_te[m]) / eff_inv[m] - 1.0)

    n_trig = int(gate.sum())
    wins = int(((pnl > 0) & gate).sum())
    total_pnl = float(pnl.sum())

    cum = pnl.cumsum()
    if len(cum) and np.isfinite(cum).all():
        running_max = np.maximum.accumulate(cum)
        max_dd = float((cum - running_max).min())
    else:
        max_dd = 0.0

    rets = pnl[gate] / stake_usd if n_trig else np.array([0.0])
    sharpe_trade = (
        float(rets.mean() / (rets.std() + 1e-12)) if n_trig else 0.0
    )

    return {
        "edge_threshold": edge_threshold,
        "ttd_max_sec": ttd_max_sec,
        "mip_max": mip_max,
        "slippage_frac": slippage_frac,
        "stake_usd": stake_usd,
        "mip_floor": mip_floor,
        "triggers": n_trig,
        "wins": wins,
        "hit_rate": float(wins / max(1, n_trig)),
        "total_pnl_usd": total_pnl,
        "mean_pnl_per_trigger": float(total_pnl / max(1, n_trig)),
        "sharpe_per_trade": sharpe_trade,
        "max_drawdown_usd": max_dd,
    }


def run_slippage_battery(
    cohorts_test: pd.DataFrame,
    p_te: np.ndarray,
    mip_te: np.ndarray,
    y_te: np.ndarray,
) -> dict:
    strategies = [
        ("general_ev", {"edge_threshold": 0.02}),
        (
            "home_run",
            {
                "edge_threshold": 0.20,
                "ttd_max_sec": 6 * 3600,
                "mip_max": 0.30,
            },
        ),
    ]
    slippages = [0.0, 0.02, 0.03, 0.05]
    out: dict = {}
    for s_name, s_cfg in strategies:
        out[s_name] = {}
        for s_frac in slippages:
            r = slippage_backtest(
                cohorts_test, p_te, mip_te, y_te,
                **s_cfg,
                slippage_frac=s_frac,
            )
            out[s_name][f"slippage_{int(s_frac * 100):02d}pct"] = r
            print(
                f"[slippage {s_frac:.0%} {s_name:<11s}] n_trig={r['triggers']:>5} "
                f"hit={r['hit_rate']:.3f} PnL=${r['total_pnl_usd']:>12,.0f} "
                f"mean/trig=${r['mean_pnl_per_trigger']:>6,.0f} "
                f"Sharpe={r['sharpe_per_trade']:.3f}"
            )
    return out


# ---------------------------------------------------------------------------
# B. Per-market temporal split (within-market first-70 / last-15 split)
# ---------------------------------------------------------------------------
def per_market_temporal_split() -> dict:
    """Pool within-market early-trade slices into a combined training set
    and within-market late-trade slices into a combined test set. One
    model trained; per-market test ROC reported."""
    t0 = time.time()
    CSV = ROOT / "data" / "03_consolidated_dataset.csv"
    df = pd.read_csv(CSV, low_memory=False)
    print(f"[pmt] loaded full CSV: {len(df):,} rows")
    df = df[pd.to_numeric(df["settlement_minus_trade_sec"], errors="coerce") > 0].copy()
    print(f"[pmt] after §4 filter: {len(df):,} rows")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["condition_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )

    # Assign per-market temporal fold via cumcount / n
    df["_rank"] = df.groupby("condition_id").cumcount()
    df["_n_market"] = df.groupby("condition_id")["condition_id"].transform("count")
    df["_pct"] = df["_rank"] / df["_n_market"]
    df["_fold"] = pd.cut(
        df["_pct"], bins=[-0.001, 0.70, 0.85, 1.0],
        labels=["train", "val", "test"],
    )

    feats = [c for c in df.columns if c not in V1.NON_FEATURE_COLS]
    tr = df[df["_fold"] == "train"].copy()
    va = df[df["_fold"] == "val"].copy()
    te = df[df["_fold"] == "test"].copy()
    print(f"[pmt] fold sizes:  train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    Xtr = tr[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    Xva = va[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    Xte = te[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    ytr = tr[V1.TARGET].to_numpy().astype(np.int64)
    yva = va[V1.TARGET].to_numpy().astype(np.int64)
    yte = te[V1.TARGET].to_numpy().astype(np.int64)

    # Winsorise on train
    for c in V1.WINSORISE_COLS:
        if c in feats:
            j = feats.index(c)
            lo, hi = np.quantile(Xtr[:, j], [0.01, 0.99])
            Xtr[:, j] = np.clip(Xtr[:, j], lo, hi)
            Xva[:, j] = np.clip(Xva[:, j], lo, hi)
            Xte[:, j] = np.clip(Xte[:, j], lo, hi)

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(Xtr)).astype(np.float32)
    Xva = sc.transform(imp.transform(Xva)).astype(np.float32)
    Xte = sc.transform(imp.transform(Xte)).astype(np.float32)

    print("[pmt] fitting LogReg...")
    lr = LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000, solver="lbfgs",
        class_weight="balanced", random_state=RANDOM_SEED,
    )
    lr.fit(Xtr, ytr)
    p_te_lr = lr.predict_proba(Xte)[:, 1]
    p_va_lr = lr.predict_proba(Xva)[:, 1]

    print("[pmt] fitting MLP (vanilla, no embedding)...")
    mlp, hist = V1.fit_mlp(Xtr, ytr, Xva, yva)
    p_te_mlp = mlp.predict(Xte, batch_size=V1.MLP_BATCH, verbose=0).ravel()
    p_va_mlp = mlp.predict(Xva, batch_size=V1.MLP_BATCH, verbose=0).ravel()

    def m(y, p):
        if len(set(y)) < 2:
            return None
        return float(roc_auc_score(y, p))

    print(
        f"[pmt] overall val  ROC  logreg={m(yva, p_va_lr):.4f}  mlp={m(yva, p_va_mlp):.4f}"
    )
    print(
        f"[pmt] overall test ROC  logreg={m(yte, p_te_lr):.4f}  mlp={m(yte, p_te_mlp):.4f}"
    )

    # Per-market test ROC
    te_reset = te.reset_index(drop=True)
    per_market = []
    for cid, g in te_reset.groupby("condition_id"):
        positions = te_reset.index.get_indexer(g.index.to_numpy())
        y_m = yte[positions]
        if len(set(y_m)) < 2:
            continue
        per_market.append({
            "condition_id": cid,
            "question": g["question"].iloc[0],
            "is_yes": int(g["is_yes"].iloc[0]) if "is_yes" in g.columns else None,
            "n_test": int(len(g)),
            "logreg_roc": float(roc_auc_score(y_m, p_te_lr[positions])),
            "mlp_roc": float(roc_auc_score(y_m, p_te_mlp[positions])),
        })

    pm_df = pd.DataFrame(per_market).sort_values("mlp_roc", ascending=False)
    pm_df.to_csv(OUT_DIR / "per_market_temporal" / "per_market_rocs.csv", index=False)

    # Histogram plot — translucent fills, contrasting edges, mean markers
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        pm_df["mlp_roc"], bins=20, alpha=0.45, color="steelblue",
        edgecolor="steelblue", linewidth=1.0, label="MLP",
    )
    ax.hist(
        pm_df["logreg_roc"], bins=20, alpha=0.45, color="darkorange",
        edgecolor="darkorange", linewidth=1.0, label="LogReg",
    )
    ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8, label="chance")
    ax.axvline(
        pm_df["mlp_roc"].mean(), color="steelblue", linestyle=":",
        linewidth=1.3, label=f"MLP mean = {pm_df['mlp_roc'].mean():.3f}",
    )
    ax.axvline(
        pm_df["logreg_roc"].mean(), color="darkorange", linestyle=":",
        linewidth=1.3, label=f"LogReg mean = {pm_df['logreg_roc'].mean():.3f}",
    )
    ax.set_xlabel("test ROC-AUC (last 15% of each market)")
    ax.set_ylabel("number of markets")
    ax.set_title(f"Per-market temporal ROC distribution (n={len(pm_df)} / 74 markets)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_market_temporal" / "roc_histogram.png", dpi=120)
    plt.close(fig)

    summary = {
        "n_markets_evaluated": int(len(pm_df)),
        "logreg": {
            "overall_test_roc": m(yte, p_te_lr),
            "per_market_mean_roc": float(pm_df["logreg_roc"].mean()),
            "per_market_median_roc": float(pm_df["logreg_roc"].median()),
            "per_market_p5_roc": float(pm_df["logreg_roc"].quantile(0.05)),
            "per_market_p95_roc": float(pm_df["logreg_roc"].quantile(0.95)),
            "markets_above_0_55": int((pm_df["logreg_roc"] > 0.55).sum()),
        },
        "mlp": {
            "overall_test_roc": m(yte, p_te_mlp),
            "per_market_mean_roc": float(pm_df["mlp_roc"].mean()),
            "per_market_median_roc": float(pm_df["mlp_roc"].median()),
            "per_market_p5_roc": float(pm_df["mlp_roc"].quantile(0.05)),
            "per_market_p95_roc": float(pm_df["mlp_roc"].quantile(0.95)),
            "markets_above_0_55": int((pm_df["mlp_roc"] > 0.55).sum()),
        },
        "top_5_markets_by_mlp_roc": pm_df.head(5).to_dict(orient="records"),
        "bottom_5_markets_by_mlp_roc": pm_df.tail(5).to_dict(orient="records"),
        "runtime_sec": float(time.time() - t0),
    }
    with open(OUT_DIR / "per_market_temporal" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(
        f"[pmt] mean per-market ROC: logreg={summary['logreg']['per_market_mean_roc']:.4f}  "
        f"mlp={summary['mlp']['per_market_mean_roc']:.4f}"
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()

    # A. Slippage-adjusted backtest on Option 2 embedding-MLP test predictions
    print("\n=== A. Slippage-adjusted backtest (embedding-MLP test predictions) ===")
    cohorts = V1.load_cohorts()
    # Load embedding-MLP test predictions
    emb_preds = pd.read_parquet(
        ROOT / "pontus" / "outputs" / "v2_market_embedding" / "modelling" / "mlp_emb"
        / "predictions_test_maskB.parquet"
    )
    # Join by transactionHash to align with cohort
    te = cohorts.test.merge(
        emb_preds[["transactionHash", "p_hat"]], on="transactionHash", how="left"
    )
    assert te["p_hat"].notna().all(), "prediction join failed"
    p_te = te["p_hat"].to_numpy()
    mip_te = te[V1.BENCHMARK].to_numpy().astype(np.float64)
    y_te = te[V1.TARGET].to_numpy().astype(np.int64)

    slippage_results = run_slippage_battery(te, p_te, mip_te, y_te)
    with open(OUT_DIR / "slippage" / "backtest.json", "w") as f:
        json.dump(slippage_results, f, indent=2, default=str)

    # B. Per-market temporal split
    print("\n=== B. Per-market temporal split ===")
    pmt = per_market_temporal_split()

    # Write top-level summary
    summary = {
        "slippage_home_run_at_3pct": slippage_results["home_run"]["slippage_03pct"]["total_pnl_usd"],
        "slippage_home_run_at_5pct": slippage_results["home_run"]["slippage_05pct"]["total_pnl_usd"],
        "per_market_temporal": {
            "n_markets": pmt["n_markets_evaluated"],
            "mlp_mean_roc": pmt["mlp"]["per_market_mean_roc"],
            "mlp_median_roc": pmt["mlp"]["per_market_median_roc"],
            "logreg_mean_roc": pmt["logreg"]["per_market_mean_roc"],
        },
        "runtime_min": round((time.time() - t0) / 60, 2),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[done] {summary['runtime_min']:.1f} min total")


if __name__ == "__main__":
    main()
