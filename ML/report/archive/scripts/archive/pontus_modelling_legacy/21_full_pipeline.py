"""
21_full_pipeline.py

Full §5 modelling pipeline for the Iran Polymarkets mispricing study.
Implements the spec in `pontus/pontus_adventure.md` §5.1 – §5.7 end-to-end.

Steps
-----
  1. Load market-cohort parquets (train / val / test, §4).
  2. Preprocess (winsorise + median-impute + standardise, train-fold fit only).
  3. Train baselines — LogReg L2, Random Forest (heavily regularised).
  4. Train MLP (tf.keras, SELU + BatchNorm + Dropout; isotonic calibration on val).
  5. Train autoencoder (tf.keras, undercomplete, SELU, MSE).
  6. Train Isolation Forest (sklearn).
  7. Probability metrics — ROC-AUC, PR-AUC, Brier, 15-bin ECE.
  8. Residual-edge analysis (RQ1b proper test — partial-corr of
     `edge - a*market_implied_prob - b` with `bet_correct`).
  9. Trading-rule backtest — general +EV (`edge > 0.02`) and home-run
     (`edge > 0.20` ∧ `time_to_deadline < 6h` ∧ `market_implied_prob < 0.30`).
  10. Anomaly-overlap test — top-decile `|edge|` trades ∩ top-decile
      reconstruction-error trades, compared against a random-overlap null.

Outputs under `pontus/outputs/`:
  modelling/{logreg,rf,mlp,autoencoder,isolation_forest}/
    metrics.json, predictions.parquet, feature_list.json,
    loss_curve.png (tf.keras models), calibration.png (calibrated models).
  residual_edge/metrics.json, scatter.png
  backtest/{general_ev,home_run}.json, pnl_curve.png
  overlap/metrics.json

Framework constraint (CBS MLDP): the MLP and the autoencoder MUST be
`tf.keras`. sklearn is fine for baselines + Isolation Forest.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# tf.keras is heavy — only import at module top so the script fails fast if
# the env is misconfigured. The CBS rubric requires it for the NN models.
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]           # .../ML/report
DATA_DIR = ROOT / "data" / "experiments"
OUT_DIR = ROOT / "pontus" / "outputs"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 18 non-feature columns kept in the finalised 54-col CSV (and in the cohort
# parquets) as IDs / labels / filter / benchmark. Everything else is treated
# as a model feature.
NON_FEATURE_COLS = {
    # IDs / metadata
    "proxyWallet", "asset", "transactionHash", "condition_id",
    "source", "question", "end_date", "resolution_ts", "deadline_ts",
    "winning_outcome_index", "resolved", "is_yes",
    # Raw cols superseded by derived features
    "size", "price", "timestamp",
    # Filter / label / benchmark
    "settlement_minus_trade_sec", "bet_correct", "market_implied_prob",
}

CATEGORICAL_COLS: set[str] = set()  # everything numeric after the finalise step
WINSORISE_COLS = ["trade_value_usd", "wallet_prior_volume_usd"]

TARGET = "bet_correct"
BENCHMARK = "market_implied_prob"
TIME_COL = "timestamp"
DEADLINE_COL = "deadline_ts"

# Trading-rule gates from pontus_adventure.md §5.2
GENERAL_EV_EDGE = 0.02
HOME_RUN_EDGE = 0.20
HOME_RUN_TTD_SEC = 6 * 3600
HOME_RUN_MIP_MAX = 0.30

# MLP hyper-parameters (§5.1 starting point)
MLP_HIDDEN = [256, 128, 64]
MLP_DROPOUT = 0.3
MLP_LR = 1e-3
MLP_BATCH = 4096
MLP_EPOCHS = 40
MLP_PATIENCE = 8
MLP_LR_PATIENCE = 3

# Autoencoder (undercomplete stacked, §5.3)
AE_HIDDEN = [24, 16, 8, 16, 24]
AE_DROPOUT = 0.1
AE_LR = 1e-3
AE_BATCH = 4096
AE_EPOCHS = 40
AE_PATIENCE = 6

# Random Forest — heavily regularised to prevent Layer-6 wallet memorisation
RF_PARAMS = dict(
    n_estimators=400,
    min_samples_leaf=200,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_SEED,
)

# Logistic regression — L2 baseline
LOGREG_PARAMS = dict(
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    solver="lbfgs",
    random_state=RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# 1. Cohort loading
# ---------------------------------------------------------------------------
@dataclass
class Cohorts:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    features: list[str]


def _load_parquet(name: str) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        raise SystemExit(
            f"missing {p} — regenerate via "
            "`python scripts/14_build_experiment_splits.py`"
        )
    return pd.read_parquet(p)


def load_cohorts() -> Cohorts:
    print("[load] train / val / test parquets")
    tr = _load_parquet("train")
    va = _load_parquet("val")
    te = _load_parquet("test")

    # Feature list = numeric cols not in NON_FEATURE_COLS. Verify the three
    # frames agree.
    feats_tr = [c for c in tr.columns if c not in NON_FEATURE_COLS]
    feats_va = [c for c in va.columns if c not in NON_FEATURE_COLS]
    feats_te = [c for c in te.columns if c not in NON_FEATURE_COLS]
    if set(feats_tr) != set(feats_va) or set(feats_tr) != set(feats_te):
        missing_va = set(feats_tr) - set(feats_va)
        missing_te = set(feats_tr) - set(feats_te)
        raise SystemExit(
            f"cohort schema mismatch:"
            f"\n  train-only cols: {missing_va} / {missing_te}"
        )

    # Coerce feature columns to numeric — parquet preserves dtype, but the
    # rare string-typed column would silently break StandardScaler.
    for df in (tr, va, te):
        for c in feats_tr:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    print(
        f"  train={len(tr):,}  val={len(va):,}  test={len(te):,}  "
        f"features={len(feats_tr)}"
    )
    return Cohorts(tr, va, te, sorted(feats_tr))


# ---------------------------------------------------------------------------
# 2. Preprocessing — fit on train, transform val+test
# ---------------------------------------------------------------------------
def winsorise_train_fit(
    cohorts: Cohorts, cols: list[str], q_lo: float = 0.01, q_hi: float = 0.99
) -> None:
    """In-place winsorisation. Quantiles computed on the TRAIN fold only, then
    applied (clipped) to val and test. Modifies frames in place.
    """
    cols = [c for c in cols if c in cohorts.features]
    if not cols:
        return
    bounds = {}
    for c in cols:
        lo, hi = cohorts.train[c].quantile([q_lo, q_hi])
        bounds[c] = (lo, hi)
    for df in (cohorts.train, cohorts.val, cohorts.test):
        for c in cols:
            lo, hi = bounds[c]
            df[c] = df[c].clip(lower=lo, upper=hi)
    print(f"[preproc] winsorised {len(cols)} columns on [{q_lo:.2f}, {q_hi:.2f}]")


def impute_and_scale(
    cohorts: Cohorts,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-impute (fit on train) + StandardScaler (fit on train). Returns
    (X_train, X_val, X_test) as dense float32 arrays ready for tf.keras.
    """
    X_tr = cohorts.train[cohorts.features].to_numpy(dtype=np.float64)
    X_va = cohorts.val[cohorts.features].to_numpy(dtype=np.float64)
    X_te = cohorts.test[cohorts.features].to_numpy(dtype=np.float64)

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr = imp.fit_transform(X_tr)
    X_tr = sc.fit_transform(X_tr)
    X_va = sc.transform(imp.transform(X_va))
    X_te = sc.transform(imp.transform(X_te))
    return X_tr.astype(np.float32), X_va.astype(np.float32), X_te.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------
def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece_15bin": expected_calibration_error(y_true, y_prob),
    }


def per_market_breakdown(
    df: pd.DataFrame, p: np.ndarray, y: np.ndarray
) -> list[dict]:
    rows = []
    key = "question" if "question" in df.columns else "condition_id"
    for q, g in df.groupby(key):
        positions = df.index.get_indexer(g.index.to_numpy())
        p_m, y_m = p[positions], y[positions]
        roc = float(roc_auc_score(y_m, p_m)) if len(set(y_m)) > 1 else None
        rows.append({
            "market": q,
            "n": int(len(g)),
            "bc_rate": float(y_m.mean()),
            "p_hat_mean": float(p_m.mean()),
            "roc_auc": roc,
        })
    return rows


# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------
def fit_logreg(X_tr: np.ndarray, y_tr: np.ndarray) -> LogisticRegression:
    t0 = time.time()
    m = LogisticRegression(**LOGREG_PARAMS).fit(X_tr, y_tr)
    print(f"[fit] logreg done in {time.time() - t0:.1f}s")
    return m


def fit_rf(X_tr: np.ndarray, y_tr: np.ndarray) -> RandomForestClassifier:
    t0 = time.time()
    m = RandomForestClassifier(**RF_PARAMS).fit(X_tr, y_tr)
    print(f"[fit] rf done in {time.time() - t0:.1f}s")
    return m


def build_mlp(input_dim: int) -> keras.Model:
    keras.utils.set_random_seed(RANDOM_SEED)
    inp = keras.Input(shape=(input_dim,), dtype="float32")
    x = inp
    for units in MLP_HIDDEN:
        x = keras.layers.Dense(
            units, activation="selu",
            kernel_initializer="lecun_normal",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(MLP_DROPOUT)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out, name="mlp_phat")
    m.compile(
        optimizer=keras.optimizers.Adam(MLP_LR),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return m


def fit_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray
) -> tuple[keras.Model, dict]:
    t0 = time.time()
    model = build_mlp(X_tr.shape[1])

    # Class weight (mild — bet_correct ≈ 0.52)
    from sklearn.utils.class_weight import compute_class_weight
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weight = {0: float(w[0]), 1: float(w[1])}

    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=MLP_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=MLP_LR_PATIENCE, min_lr=1e-5
        ),
    ]
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH,
        class_weight=class_weight,
        callbacks=cb,
        verbose=2,
    )
    print(f"[fit] mlp done in {time.time() - t0:.1f}s ({len(hist.history['loss'])} epochs)")
    return model, hist.history


def build_autoencoder(input_dim: int) -> keras.Model:
    keras.utils.set_random_seed(RANDOM_SEED)
    inp = keras.Input(shape=(input_dim,), dtype="float32")
    x = inp
    for units in AE_HIDDEN:
        x = keras.layers.Dense(
            units, activation="selu", kernel_initializer="lecun_normal"
        )(x)
        if AE_DROPOUT > 0:
            x = keras.layers.Dropout(AE_DROPOUT)(x)
    out = keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    m = keras.Model(inp, out, name="autoencoder")
    m.compile(optimizer=keras.optimizers.Adam(AE_LR), loss="mse")
    return m


def fit_autoencoder(
    X_tr: np.ndarray, X_va: np.ndarray
) -> tuple[keras.Model, dict]:
    t0 = time.time()
    m = build_autoencoder(X_tr.shape[1])
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=AE_PATIENCE, restore_best_weights=True
        )
    ]
    hist = m.fit(
        X_tr, X_tr,
        validation_data=(X_va, X_va),
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH,
        callbacks=cb,
        verbose=2,
    )
    print(f"[fit] autoencoder done in {time.time() - t0:.1f}s "
          f"({len(hist.history['loss'])} epochs)")
    return m, hist.history


def fit_isolation_forest(X_tr: np.ndarray) -> IsolationForest:
    t0 = time.time()
    m = IsolationForest(
        n_estimators=200,
        contamination="auto",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    ).fit(X_tr)
    print(f"[fit] isolation forest done in {time.time() - t0:.1f}s")
    return m


# ---------------------------------------------------------------------------
# 5. Calibration
# ---------------------------------------------------------------------------
def calibrate_isotonic(
    p_va: np.ndarray, y_va: np.ndarray, p_te: np.ndarray
) -> tuple[np.ndarray, IsotonicRegression]:
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va)
    return iso.transform(p_te), iso


def plot_calibration(
    p: np.ndarray, y: np.ndarray, title: str, out_path: Path, n_bins: int = 15
) -> None:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=False)
    centres, means = [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        centres.append((bins[b] + bins[b + 1]) / 2)
        means.append(y[mask].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    ax.plot(centres, means, "o-", label="model")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_loss_curve(hist: dict, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["loss"], label="train")
    if "val_loss" in hist:
        ax.plot(hist["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Residual-edge (§5.7 item 2)
# ---------------------------------------------------------------------------
def residual_edge_analysis(
    p_hat_tr: np.ndarray, mip_tr: np.ndarray, y_tr: np.ndarray,
    p_hat_va: np.ndarray, mip_va: np.ndarray, y_va: np.ndarray,
    p_hat_te: np.ndarray, mip_te: np.ndarray, y_te: np.ndarray,
) -> dict:
    """Fit `edge = a * mip + b` on train, residualise, score the residual
    against bet_correct. Per Alex's investigation this is the honest gap
    test — aggregate ROC of `edge` is contaminated by the naive-market
    Simpson's paradox (naive baseline hits 0.63 test ROC without signal).
    """
    edge_tr = p_hat_tr - mip_tr
    edge_va = p_hat_va - mip_va
    edge_te = p_hat_te - mip_te

    a, b = np.polyfit(mip_tr, edge_tr, deg=1)
    res_tr = edge_tr - (a * mip_tr + b)
    res_va = edge_va - (a * mip_va + b)
    res_te = edge_te - (a * mip_te + b)

    def _r(y, r):
        return float(roc_auc_score(y, r)) if len(set(y)) > 1 else None

    def _partial_corr(y, r):
        y_c = y - y.mean()
        r_c = r - r.mean()
        denom = (np.sqrt((y_c ** 2).sum()) * np.sqrt((r_c ** 2).sum())) or 1e-12
        return float((y_c * r_c).sum() / denom)

    return {
        "slope_a": float(a), "intercept_b": float(b),
        "train": {
            "residual_roc": _r(y_tr, res_tr),
            "partial_corr_bc_vs_residual": _partial_corr(y_tr, res_tr),
            "edge_mean": float(edge_tr.mean()),
            "edge_std": float(edge_tr.std()),
        },
        "val": {
            "residual_roc": _r(y_va, res_va),
            "partial_corr_bc_vs_residual": _partial_corr(y_va, res_va),
        },
        "test": {
            "residual_roc": _r(y_te, res_te),
            "partial_corr_bc_vs_residual": _partial_corr(y_te, res_te),
            "edge_mean": float(edge_te.mean()),
            "edge_std": float(edge_te.std()),
        },
    }


# ---------------------------------------------------------------------------
# 7. Trading-rule backtest (§5.2)
# ---------------------------------------------------------------------------
def _pnl_per_dollar(bet_correct: np.ndarray, mip: np.ndarray, follow: np.ndarray) -> np.ndarray:
    """PnL per $1 staked.

    `follow=True` → we take the same direction as the wallet (bet on
    bet_correct=1); payoff 1/mip if bet_correct=1 else 0, minus $1 stake.
    `follow=False` → we take the opposite direction (bet on
    bet_correct=0); payoff 1/(1 - mip) if bet_correct=0 else 0, minus $1.
    """
    pnl = np.zeros_like(bet_correct, dtype=np.float64)
    # Follow trades
    m = follow
    if m.any():
        mip_m = np.clip(mip[m], 1e-4, 1 - 1e-4)
        pnl[m] = bet_correct[m] / mip_m - 1.0
    # Inverse trades
    inv = ~follow
    if inv.any():
        mip_inv = np.clip(mip[inv], 1e-4, 1 - 1e-4)
        pnl[inv] = (1 - bet_correct[inv]) / (1 - mip_inv) - 1.0
    return pnl


def backtest_rule(
    df_te: pd.DataFrame,
    p_hat_te: np.ndarray,
    mip_te: np.ndarray,
    y_te: np.ndarray,
    *,
    edge_threshold: float,
    ttd_max_sec: float | None = None,
    mip_max: float | None = None,
    stake_usd: float = 100.0,
) -> dict:
    """Gate -> stake -> PnL. Returns per-trade frame + aggregate stats.

    The gate fires both ways (follow when `edge > +thresh`, inverse when
    `edge < -thresh`), filtered by the optional time-to-deadline and
    max-mip conditions from the home-run spec.
    """
    edge = p_hat_te - mip_te

    # Time-to-deadline at trading-rule time (not a model feature, recomputed
    # here from deadline_ts − timestamp).
    deadline = pd.to_datetime(df_te[DEADLINE_COL], utc=True, errors="coerce")
    ts = pd.to_datetime(df_te[TIME_COL], utc=True, errors="coerce")
    ttd = (deadline - ts).dt.total_seconds().to_numpy()

    gate_follow = edge > edge_threshold
    gate_inverse = edge < -edge_threshold
    gate = gate_follow | gate_inverse
    if ttd_max_sec is not None:
        gate &= ttd < ttd_max_sec
    if mip_max is not None:
        gate &= mip_te < mip_max

    follow = gate_follow & gate

    # PnL per $1, then scaled by flat stake for the triggers.
    pnl_pd = _pnl_per_dollar(y_te, mip_te, follow)
    stake = np.where(gate, stake_usd, 0.0)
    trade_pnl = stake * pnl_pd * gate

    n_trig = int(gate.sum())
    wins = int(((trade_pnl > 0) & gate).sum())
    total_pnl = float(trade_pnl.sum())

    # Sharpe on per-trade returns (not scaled to annualised — this is a
    # fixed-window backtest, so reporting Sharpe_trade; the report can
    # compute an annualised version if useful).
    rets = trade_pnl[gate] / stake_usd if n_trig else np.array([0.0])
    sharpe_trade = float(rets.mean() / (rets.std() + 1e-12)) if n_trig else 0.0
    hit_rate = float(wins / n_trig) if n_trig else 0.0

    # Max drawdown on the cumulative curve
    cum = trade_pnl.cumsum()
    if len(cum):
        running_max = np.maximum.accumulate(cum)
        dd = cum - running_max
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    return {
        "config": {
            "edge_threshold": edge_threshold,
            "ttd_max_sec": ttd_max_sec,
            "mip_max": mip_max,
            "stake_usd": stake_usd,
        },
        "triggers": n_trig,
        "follow_triggers": int(follow.sum()),
        "inverse_triggers": int(n_trig - follow.sum()),
        "wins": wins,
        "hit_rate": hit_rate,
        "total_pnl_usd": total_pnl,
        "mean_pnl_per_trigger": float(total_pnl / max(1, n_trig)),
        "sharpe_per_trade": sharpe_trade,
        "max_drawdown_usd": max_dd,
        "pnl_curve": cum.tolist(),
    }


def plot_pnl_curves(backtests: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, r in backtests.items():
        cum = np.asarray(r["pnl_curve"])
        if len(cum):
            ax.plot(cum, label=f"{name}  n_trig={r['triggers']}  pnl=${r['total_pnl_usd']:.0f}")
    ax.set_xlabel("trade index (test cohort)")
    ax.set_ylabel("cumulative PnL, USD")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_title("Trading-rule backtest — cumulative PnL on test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Anomaly overlap (§5.3 + §5.7)
# ---------------------------------------------------------------------------
def anomaly_overlap(
    edge_te: np.ndarray, recon_err_te: np.ndarray, iso_score_te: np.ndarray,
    *, top_frac: float = 0.1,
) -> dict:
    """Overlap at top-decile of |edge| vs top-decile of reconstruction error
    and Isolation Forest anomaly score. Reports the overlap count and the
    random-overlap null (expected count when the two sets are independent).
    """
    n = len(edge_te)
    k = int(np.ceil(n * top_frac))
    null_overlap = (k * k) / n   # expectation under independence

    top_edge = np.argpartition(-np.abs(edge_te), k - 1)[:k]
    top_recon = np.argpartition(-recon_err_te, k - 1)[:k]
    top_iso = np.argpartition(iso_score_te, k - 1)[:k]  # lower score = more anomalous

    ov_edge_recon = int(len(np.intersect1d(top_edge, top_recon)))
    ov_edge_iso = int(len(np.intersect1d(top_edge, top_iso)))
    ov_recon_iso = int(len(np.intersect1d(top_recon, top_iso)))

    return {
        "n_test": n,
        "top_frac": top_frac,
        "k_top": k,
        "random_null_overlap": float(null_overlap),
        "overlap_edge_vs_autoencoder": ov_edge_recon,
        "overlap_edge_vs_isolation_forest": ov_edge_iso,
        "overlap_autoencoder_vs_isolation_forest": ov_recon_iso,
        "uplift_edge_vs_autoencoder_x": float(ov_edge_recon / max(1.0, null_overlap)),
        "uplift_edge_vs_isolation_forest_x": float(ov_edge_iso / max(1.0, null_overlap)),
    }


# ---------------------------------------------------------------------------
# 9. I/O helpers
# ---------------------------------------------------------------------------
def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def save_predictions(
    cohorts: Cohorts,
    predictions: dict[str, np.ndarray],
    model_dir: Path,
) -> None:
    """Persist per-fold predictions alongside the cohort identifiers so
    downstream analyses (feature-importance, Magamyman, PR replay) can run
    without re-training. `predictions` maps fold name → p_hat array.
    """
    for fold_name, df_fold in [
        ("train", cohorts.train),
        ("val", cohorts.val),
        ("test", cohorts.test),
    ]:
        id_cols = [c for c in ("proxyWallet", "condition_id", "transactionHash",
                               "timestamp", "bet_correct", "market_implied_prob")
                   if c in df_fold.columns]
        out = df_fold[id_cols].copy()
        out["p_hat"] = predictions[fold_name]
        out.to_parquet(model_dir / f"predictions_{fold_name}.parquet", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cohorts = load_cohorts()
    winsorise_train_fit(cohorts, WINSORISE_COLS)
    X_tr, X_va, X_te = impute_and_scale(cohorts)
    y_tr = cohorts.train[TARGET].to_numpy().astype(np.int64)
    y_va = cohorts.val[TARGET].to_numpy().astype(np.int64)
    y_te = cohorts.test[TARGET].to_numpy().astype(np.int64)
    mip_tr = cohorts.train[BENCHMARK].to_numpy().astype(np.float64)
    mip_va = cohorts.val[BENCHMARK].to_numpy().astype(np.float64)
    mip_te = cohorts.test[BENCHMARK].to_numpy().astype(np.float64)

    print(f"[shape] X_tr={X_tr.shape} X_va={X_va.shape} X_te={X_te.shape}")
    print(f"[balance] bc rate  train={y_tr.mean():.3f}  val={y_va.mean():.3f}  test={y_te.mean():.3f}")

    # --- feature list dump (shared across models) -----------------------
    (OUT_DIR / "modelling").mkdir(parents=True, exist_ok=True)
    save_json(cohorts.features, OUT_DIR / "modelling" / "feature_list.json")

    # =====================================================================
    # Baselines
    # =====================================================================
    models: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}

    # LogReg
    m = fit_logreg(X_tr, y_tr)
    models["logreg"] = m
    p_tr = m.predict_proba(X_tr)[:, 1]
    p_va = m.predict_proba(X_va)[:, 1]
    p_te = m.predict_proba(X_te)[:, 1]
    predictions["logreg"] = {"train": p_tr, "val": p_va, "test": p_te}

    # Random forest
    m = fit_rf(X_tr, y_tr)
    models["rf"] = m
    predictions["rf"] = {
        "train": m.predict_proba(X_tr)[:, 1],
        "val":   m.predict_proba(X_va)[:, 1],
        "test":  m.predict_proba(X_te)[:, 1],
    }

    # MLP + isotonic calibration
    mlp, mlp_hist = fit_mlp(X_tr, y_tr, X_va, y_va)
    models["mlp"] = mlp
    p_tr_mlp = mlp.predict(X_tr, batch_size=MLP_BATCH, verbose=0).ravel()
    p_va_mlp = mlp.predict(X_va, batch_size=MLP_BATCH, verbose=0).ravel()
    p_te_mlp = mlp.predict(X_te, batch_size=MLP_BATCH, verbose=0).ravel()
    p_te_mlp_cal, iso = calibrate_isotonic(p_va_mlp, y_va, p_te_mlp)
    p_va_mlp_cal = iso.transform(p_va_mlp)
    predictions["mlp"] = {"train": p_tr_mlp, "val": p_va_mlp_cal, "test": p_te_mlp_cal}

    # Plots
    loss_dir = OUT_DIR / "modelling" / "mlp"
    loss_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(mlp_hist, "MLP BCE loss", loss_dir / "loss_curve.png")
    plot_calibration(p_te_mlp_cal, y_te, "MLP (isotonic) — test", loss_dir / "calibration.png")

    # =====================================================================
    # Probability metrics + per-market
    # =====================================================================
    for name, preds in predictions.items():
        mdir = OUT_DIR / "modelling" / name
        mdir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "train": probability_metrics(y_tr, preds["train"]),
            "val":   probability_metrics(y_va, preds["val"]),
            "test":  probability_metrics(y_te, preds["test"]),
            "per_market_test": per_market_breakdown(cohorts.test, preds["test"], y_te),
        }
        save_json(metrics, mdir / "metrics.json")
        save_predictions(cohorts, {
            "train": preds["train"], "val": preds["val"], "test": preds["test"],
        }, mdir)
        print(f"[metrics {name}]  test roc={metrics['test']['roc_auc']}  "
              f"brier={metrics['test']['brier']:.4f}  "
              f"ece={metrics['test']['ece_15bin']:.4f}")

    # =====================================================================
    # Naive-market baseline (p_hat = market_implied_prob)
    # =====================================================================
    naive_dir = OUT_DIR / "modelling" / "naive_market"
    naive_dir.mkdir(parents=True, exist_ok=True)
    save_json({
        "train": probability_metrics(y_tr, mip_tr),
        "val":   probability_metrics(y_va, mip_va),
        "test":  probability_metrics(y_te, mip_te),
        "note": (
            "Aggregate ROC of the naive market baseline is not 0.5 in this "
            "dataset — Simpson's paradox via (side × outcomeIndex) makes it "
            "~0.63 on test. The efficient-market null implies calibration, "
            "not ROC 0.5. See residual_edge/metrics.json for the honest "
            "gap-signal test."
        ),
    }, naive_dir / "metrics.json")

    # =====================================================================
    # Residual-edge analysis (RQ1b proper test)
    # =====================================================================
    r_edge = residual_edge_analysis(
        predictions["mlp"]["train"], mip_tr, y_tr,
        predictions["mlp"]["val"],   mip_va, y_va,
        predictions["mlp"]["test"],  mip_te, y_te,
    )
    (OUT_DIR / "residual_edge").mkdir(parents=True, exist_ok=True)
    save_json(r_edge, OUT_DIR / "residual_edge" / "metrics.json")
    print(f"[residual-edge] test partial-corr(bc, residual) = "
          f"{r_edge['test']['partial_corr_bc_vs_residual']:+.4f}")

    # =====================================================================
    # Trading-rule backtest (test cohort only)
    # =====================================================================
    bt_general = backtest_rule(
        cohorts.test, predictions["mlp"]["test"], mip_te, y_te,
        edge_threshold=GENERAL_EV_EDGE,
    )
    bt_home_run = backtest_rule(
        cohorts.test, predictions["mlp"]["test"], mip_te, y_te,
        edge_threshold=HOME_RUN_EDGE,
        ttd_max_sec=HOME_RUN_TTD_SEC, mip_max=HOME_RUN_MIP_MAX,
    )
    bt_dir = OUT_DIR / "backtest"
    bt_dir.mkdir(parents=True, exist_ok=True)
    save_json(bt_general, bt_dir / "general_ev.json")
    save_json(bt_home_run, bt_dir / "home_run.json")
    plot_pnl_curves(
        {"general_ev": bt_general, "home_run": bt_home_run},
        bt_dir / "pnl_curve.png",
    )
    print(f"[backtest] general_ev: n_trig={bt_general['triggers']}  "
          f"pnl=${bt_general['total_pnl_usd']:,.0f}  hit={bt_general['hit_rate']:.3f}")
    print(f"[backtest] home_run:   n_trig={bt_home_run['triggers']}  "
          f"pnl=${bt_home_run['total_pnl_usd']:,.0f}  hit={bt_home_run['hit_rate']:.3f}")

    # =====================================================================
    # Autoencoder + Isolation Forest + anomaly overlap
    # =====================================================================
    ae, ae_hist = fit_autoencoder(X_tr, X_va)
    ae_dir = OUT_DIR / "modelling" / "autoencoder"
    ae_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(ae_hist, "Autoencoder MSE loss", ae_dir / "loss_curve.png")
    recon_tr = ae.predict(X_tr, batch_size=AE_BATCH, verbose=0)
    recon_va = ae.predict(X_va, batch_size=AE_BATCH, verbose=0)
    recon_te = ae.predict(X_te, batch_size=AE_BATCH, verbose=0)
    err_tr = ((X_tr - recon_tr) ** 2).mean(axis=1)
    err_va = ((X_va - recon_va) ** 2).mean(axis=1)
    err_te = ((X_te - recon_te) ** 2).mean(axis=1)
    save_json(
        {
            "train_mse_mean": float(err_tr.mean()),
            "val_mse_mean":   float(err_va.mean()),
            "test_mse_mean":  float(err_te.mean()),
            "test_mse_p95":   float(np.quantile(err_te, 0.95)),
            "test_mse_p99":   float(np.quantile(err_te, 0.99)),
            "epochs_trained": len(ae_hist["loss"]),
        },
        ae_dir / "metrics.json",
    )

    iso_f = fit_isolation_forest(X_tr)
    iso_score_te = iso_f.score_samples(X_te)   # higher = less anomalous
    if_dir = OUT_DIR / "modelling" / "isolation_forest"
    if_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "mean_score_train": float(iso_f.score_samples(X_tr).mean()),
            "mean_score_test":  float(iso_score_te.mean()),
            "test_score_p5":    float(np.quantile(iso_score_te, 0.05)),
        },
        if_dir / "metrics.json",
    )

    edge_te = predictions["mlp"]["test"] - mip_te
    overlap = anomaly_overlap(edge_te, err_te, iso_score_te, top_frac=0.10)
    (OUT_DIR / "overlap").mkdir(parents=True, exist_ok=True)
    save_json(overlap, OUT_DIR / "overlap" / "metrics.json")
    print(f"[overlap] |edge| ∩ recon at top-10%  = "
          f"{overlap['overlap_edge_vs_autoencoder']} "
          f"(null={overlap['random_null_overlap']:.0f}, "
          f"uplift×{overlap['uplift_edge_vs_autoencoder_x']:.2f})")

    print("\n[done]")


if __name__ == "__main__":
    main()
