"""
22_v2_pipeline.py — rigorous ensemble modelling for the Iran Polymarkets study.

Builds on `21_full_pipeline.py` (imported as a library for the shared
preprocessing + metric helpers) and adds course-inspired techniques to beat
Alex's baseline sweep (LogReg L2 test ROC ≈ 0.532 on the 32-feature
direction-dropped set).

Technique stack
---------------
- **GroupKFold(proxyWallet) out-of-fold stacking** (Wolpert 1992, course
  Lecture 7 ensembles). Base learners — LogReg (L2), Random Forest, MLP —
  are fit on the first four folds and predict the fifth, in rotation, so
  each training row gets an out-of-fold prediction from each base. A meta
  logistic regression then fits `y_tr` on the three OOF columns. The bases
  are refit on the full training fold, and the meta applies at inference
  time. Wallet-grouped splits prevent Layer-6 wallet-fingerprint memorisation
  from biasing the meta-training signal.
- **Dual calibration** (course Lecture 12). Fit both isotonic regression
  and Platt scaling on the val fold; pick whichever has the lower 15-bin
  ECE on val. Apply the chosen calibrator to test.
- **Permutation feature importance** (course Lecture 14 XAI). Shuffle each
  feature in the val fold and measure ROC-AUC decay. Reports a ranked list
  — the headline evidence for the Discussion.
- **Kelly-sized trading rule** (Kelly 1956). Stake a fraction `f = (b*p − q)/b`
  of current bankroll per trigger, where `b` is the payoff ratio, clipped
  to [0, 0.25] so no single bet exceeds a quarter of the bank. Reports
  alongside flat-stake and liquidity-bounded variants.
- **Wallet-level bootstrap 95 % CIs** on test ROC, Brier, PnL (pontus_
  adventure §5.7 layer 4c). Re-samples the proxyWallet index with
  replacement to preserve within-wallet correlation.
- **Magamyman sanity check** (pontus_adventure §5.7 layer 5; Mitts & Ofir
  2026). Attempts to locate the Magamyman-style wallet in the test cohort
  using the documented signature (new wallet, >$500k cumulative volume,
  low-mip entry in the final hour before deadline). Reports the rank of
  those trades under our p_hat. Graceful if no candidate is present in
  the cohort.

Outputs (all under `pontus/outputs/v2/`)
  modelling/{logreg,rf,mlp,stack_isotonic,stack_platt,stack_chosen}/
    metrics.json, predictions_{train,val,test}.parquet
  modelling/stack_chosen/oof_predictions.parquet
  permutation_importance/{val_roc_decay.json, top20.png}
  residual_edge/metrics.json
  backtest/{flat.json, kelly.json, clipped.json, pnl_curve.png}
  bootstrap/{roc_ci.json, brier_ci.json, pnl_ci.json}
  overlap/metrics.json
  autoencoder/metrics.json, loss_curve.png
  isolation_forest/metrics.json
  magamyman/check.json
  run_summary.json

Framework constraint (CBS MLDP): MLP + autoencoder MUST be tf.keras.

Approximate runtime on CPU: 70–100 min end-to-end (OOF stacking dominates —
five MLP fits with early stopping are the long pole). Run with `caffeinate`
to prevent sleep:
    caffeinate -i python pontus/scripts/22_v2_pipeline.py \
      2>&1 | tee pontus/outputs/v2/run.log
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Import v1 helpers as a library. The `sys.modules` dance is required so
# @dataclass inside 21_full_pipeline.py resolves cls.__module__.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / "pontus" / "scripts" / "21_full_pipeline.py"
_spec = importlib.util.spec_from_file_location("pipeline_v1", V1_PATH)
V1 = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_v1"] = V1
_spec.loader.exec_module(V1)

DATA_DIR = ROOT / "data" / "experiments"
OUT_DIR = ROOT / "pontus" / "outputs" / "v2"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Silence sklearn deprecation noise so the log is readable
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants re-exported for readability
# ---------------------------------------------------------------------------
NON_FEATURE_COLS = V1.NON_FEATURE_COLS
WINSORISE_COLS = V1.WINSORISE_COLS
TARGET = V1.TARGET
BENCHMARK = V1.BENCHMARK
TIME_COL = V1.TIME_COL
DEADLINE_COL = V1.DEADLINE_COL

MLP_HIDDEN = V1.MLP_HIDDEN
MLP_DROPOUT = V1.MLP_DROPOUT
MLP_LR = V1.MLP_LR
MLP_BATCH = V1.MLP_BATCH
MLP_EPOCHS = V1.MLP_EPOCHS
MLP_PATIENCE = V1.MLP_PATIENCE
MLP_LR_PATIENCE = V1.MLP_LR_PATIENCE

OOF_FOLDS = 5              # GroupKFold splits for OOF stacking
BOOTSTRAP_ROUNDS = 300     # wallet-level resample count
KELLY_FRACTION_CAP = 0.25  # safety cap on Kelly stake fraction

# Trading-rule gates — from pontus_adventure.md §5.2
GENERAL_EV_EDGE = 0.02
HOME_RUN_EDGE = 0.20
HOME_RUN_TTD_SEC = 6 * 3600
HOME_RUN_MIP_MAX = 0.30


# ---------------------------------------------------------------------------
# Base-learner wrappers (use v1 where possible; thin shims where not)
# ---------------------------------------------------------------------------
def fit_logreg(X_tr, y_tr):
    return V1.fit_logreg(X_tr, y_tr)


def fit_rf(X_tr, y_tr):
    return V1.fit_rf(X_tr, y_tr)


def fit_rf_calibrated(X_tr: np.ndarray, y_tr: np.ndarray) -> CalibratedClassifierCV:
    """RandomForest wrapped in internal 3-fold isotonic calibration.

    Addresses the v2-run-1 finding that the meta learner over-weighted the
    uncalibrated RF (train ROC 0.82 → test 0.55, implying heavy
    wallet-memorisation the meta propagated with coefficient +8.2).
    Isotonic calibration on held-out folds of train flattens the RF's
    confident-but-wrong scores before the meta sees them.
    """
    base = RandomForestClassifier(**V1.RF_PARAMS)
    cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    t0 = time.time()
    cal.fit(X_tr, y_tr)
    print(f"[fit] rf-calibrated done in {time.time() - t0:.1f}s")
    return cal


def fit_mlp(X_tr, y_tr, X_va, y_va):
    return V1.fit_mlp(X_tr, y_tr, X_va, y_va)


# ---------------------------------------------------------------------------
# 1. GroupKFold out-of-fold stacking
# ---------------------------------------------------------------------------
def oof_predictions(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    *,
    n_splits: int = OOF_FOLDS,
    rf_calibrated: bool = True,
) -> dict[str, np.ndarray]:
    """GroupKFold OOF predictions for three base learners.

    Returns a dict {"logreg": oof, "rf": oof, "mlp": oof} where each oof is
    shape (len(X_tr),) with the fold-out prediction for every training row.

    Layer-6 wallet memorisation bias is mitigated by grouping on proxyWallet:
    a wallet's trades never span both sides of the split. When
    `rf_calibrated=True` the RF is wrapped in isotonic calibration inside
    each OOF fold (step 3 of the v2-run-2 follow-ups).
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof = {k: np.zeros(len(X_tr), dtype=np.float32) for k in ("logreg", "rf", "mlp")}
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_tr, groups_tr)):
        print(f"  [oof fold {fold + 1}/{n_splits}] train={len(tr_idx):,} out={len(va_idx):,}")
        t0 = time.time()

        Xf_tr, Xf_va = X_tr[tr_idx], X_tr[va_idx]
        yf_tr, yf_va = y_tr[tr_idx], y_tr[va_idx]

        m_lr = fit_logreg(Xf_tr, yf_tr)
        oof["logreg"][va_idx] = m_lr.predict_proba(Xf_va)[:, 1].astype(np.float32)

        if rf_calibrated:
            m_rf = fit_rf_calibrated(Xf_tr, yf_tr)
        else:
            m_rf = fit_rf(Xf_tr, yf_tr)
        oof["rf"][va_idx] = m_rf.predict_proba(Xf_va)[:, 1].astype(np.float32)

        m_mlp, _ = fit_mlp(Xf_tr, yf_tr, Xf_va, yf_va)
        oof["mlp"][va_idx] = m_mlp.predict(Xf_va, batch_size=MLP_BATCH, verbose=0).ravel().astype(
            np.float32
        )
        print(f"    fold ok in {time.time() - t0:.0f}s")

    return oof


def fit_meta(
    oof: dict[str, np.ndarray], y_tr: np.ndarray,
    keys: tuple[str, ...] = ("logreg", "rf", "mlp"),
) -> tuple[LogisticRegression, tuple[str, ...]]:
    """Fit the stacking meta-learner — LogReg on OOF base predictions → y_tr.
    `keys` selects which base learners feed the meta; the default uses all
    three, the ("logreg", "mlp") variant skips RF (step 2 of the v2-run-2
    follow-ups — v2-run-1 had the meta load +8.2 on a poorly-calibrated RF).
    """
    X_meta = np.column_stack([oof[k] for k in keys]).astype(np.float64)
    meta = LogisticRegression(
        penalty="l2", C=10.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_SEED
    )
    meta.fit(X_meta, y_tr)
    coef_str = "  ".join(
        f"{k}={meta.coef_[0, i]:+.3f}" for i, k in enumerate(keys)
    )
    print(f"  meta LogReg coefs ({','.join(keys)}): {coef_str}  "
          f"intercept={meta.intercept_[0]:+.3f}")
    return meta, keys


def apply_meta(
    meta: LogisticRegression, base_preds: dict[str, np.ndarray],
    keys: tuple[str, ...],
) -> np.ndarray:
    X = np.column_stack([base_preds[k] for k in keys]).astype(np.float64)
    return meta.predict_proba(X)[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Dual calibration — isotonic + Platt (sigmoid), pick by val ECE
# ---------------------------------------------------------------------------
class PlattCalibrator:
    """Classical Platt scaling: fit a logistic regression on raw p_hat → y."""

    def __init__(self):
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(p.reshape(-1, 1), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(p.reshape(-1, 1))[:, 1].astype(np.float32)


def pick_calibrator_kfold(
    p_va: np.ndarray, y_va: np.ndarray, p_te: np.ndarray,
    val_df: pd.DataFrame, *, n_splits: int = 3,
) -> tuple[str, np.ndarray, np.ndarray, float, float]:
    """Choose isotonic vs Platt by **GroupKFold cross-validated** val ECE,
    then refit the chosen calibrator on the full val fold and apply to test.

    Step 4 of the v2-run-2 follow-ups. v2-run-1's picker fit the calibrator
    on val and evaluated its ECE on the same val — isotonic always won by
    construction (ECE ≈ 1.5e-8 on the training fold). The k-fold split
    gives both calibrators a held-out ECE they weren't fit on, so the
    choice is informative.
    """
    groups = val_df["proxyWallet"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    iso_eces, platt_eces = [], []
    for tr_idx, va_idx in gkf.split(p_va, y_va, groups):
        iso_tmp = IsotonicRegression(out_of_bounds="clip").fit(
            p_va[tr_idx], y_va[tr_idx]
        )
        platt_tmp = PlattCalibrator().fit(p_va[tr_idx], y_va[tr_idx])
        iso_eces.append(
            V1.expected_calibration_error(
                y_va[va_idx], iso_tmp.transform(p_va[va_idx])
            )
        )
        platt_eces.append(
            V1.expected_calibration_error(
                y_va[va_idx], platt_tmp.transform(p_va[va_idx])
            )
        )
    ece_iso = float(np.mean(iso_eces))
    ece_platt = float(np.mean(platt_eces))

    if ece_iso <= ece_platt:
        iso_final = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va)
        return "isotonic", iso_final.transform(p_va), iso_final.transform(p_te), ece_iso, ece_platt
    else:
        platt_final = PlattCalibrator().fit(p_va, y_va)
        return "platt", platt_final.transform(p_va), platt_final.transform(p_te), ece_iso, ece_platt


# ---------------------------------------------------------------------------
# 3. Permutation importance (Lecture 14 XAI)
# ---------------------------------------------------------------------------
def permutation_importance(
    score_fn, X: np.ndarray, y: np.ndarray, features: list[str],
    *, n_repeats: int = 5, seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Feature permutation ROC-AUC decay on a held-out fold.

    `score_fn(X) -> p_hat` is the post-calibration stacked prediction pipeline.
    Shuffles each feature `n_repeats` times and averages the ROC drop vs the
    baseline. Returns a DataFrame sorted by mean drop, descending.
    """
    rng = np.random.default_rng(seed)
    baseline = roc_auc_score(y, score_fn(X))
    rows = []
    for j, col in enumerate(features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, j])
            roc_perm = roc_auc_score(y, score_fn(X_perm))
            drops.append(baseline - roc_perm)
        rows.append({
            "feature": col,
            "mean_drop": float(np.mean(drops)),
            "std_drop": float(np.std(drops)),
            "baseline_roc": float(baseline),
        })
    return pd.DataFrame(rows).sort_values("mean_drop", ascending=False).reset_index(drop=True)


def plot_importance(imp: pd.DataFrame, top_k: int, out_path: Path) -> None:
    sub = imp.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * top_k)))
    ax.barh(sub["feature"], sub["mean_drop"], xerr=sub["std_drop"], color="steelblue")
    ax.set_xlabel("mean ROC-AUC drop when shuffled")
    ax.set_title(f"Permutation importance (top {top_k}, val fold)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Kelly-sized backtest (alongside flat + clipped variants)
# ---------------------------------------------------------------------------
def backtest_kelly(
    df_te: pd.DataFrame,
    p_te: np.ndarray,
    mip_te: np.ndarray,
    y_te: np.ndarray,
    *,
    edge_threshold: float,
    ttd_max_sec: float | None = None,
    mip_max: float | None = None,
    starting_bankroll: float = 10_000.0,
    fraction_cap: float = KELLY_FRACTION_CAP,
) -> dict:
    """Kelly-sized streaming backtest.

    For each triggered trade we compute the Kelly fraction on the two-outcome
    binary option: stake `f = (b*p - q)/b` of current bankroll, where
      follow side: b = 1/mip - 1, p = p_hat, q = 1 - p_hat
      inverse side: b = 1/(1 - mip) - 1, p = 1 - p_hat, q = p_hat
    Clipped to [0, `fraction_cap`] so no single trade dominates the bank.

    Capital is compounding — wins feed future stakes.
    """
    # Same gate logic as v1.backtest_rule.
    edge = p_te - mip_te
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

    bankroll = starting_bankroll
    cum = np.zeros(len(df_te), dtype=np.float64)
    stake_record = np.zeros(len(df_te), dtype=np.float64)
    pnl_record = np.zeros(len(df_te), dtype=np.float64)
    n_trig = 0
    wins = 0
    min_bank = bankroll

    for i in range(len(df_te)):
        if not gate[i] or bankroll <= 0:
            cum[i] = bankroll - starting_bankroll
            continue
        # Guard against NaN in p_te / mip_te that would propagate through the
        # bankroll state and make downstream stats NaN (step 1 of v2-run-2).
        if not (np.isfinite(p_te[i]) and np.isfinite(mip_te[i])
                and np.isfinite(y_te[i])):
            cum[i] = bankroll - starting_bankroll
            continue
        n_trig += 1
        mip = float(np.clip(mip_te[i], 1e-3, 1 - 1e-3))
        if gate_follow[i]:
            p = float(p_te[i])
            b = 1 / mip - 1
        else:
            p = float(1 - p_te[i])
            b = 1 / (1 - mip) - 1
        q = 1 - p
        f = (b * p - q) / b if (b > 0 and np.isfinite(b)) else 0.0
        if not np.isfinite(f):
            f = 0.0
        f = max(0.0, min(f, fraction_cap))
        stake = f * bankroll
        if stake <= 0 or not np.isfinite(stake):
            cum[i] = bankroll - starting_bankroll
            continue

        # Realise PnL
        if gate_follow[i]:
            pnl = stake * (y_te[i] / mip - 1.0)
        else:
            pnl = stake * ((1 - y_te[i]) / (1 - mip) - 1.0)
        if not np.isfinite(pnl):
            cum[i] = bankroll - starting_bankroll
            continue
        bankroll += pnl
        min_bank = min(min_bank, bankroll)
        stake_record[i] = stake
        pnl_record[i] = pnl
        wins += int(pnl > 0)
        cum[i] = bankroll - starting_bankroll

    # Only sum stakes on rows that actually placed a trade (stake_record > 0).
    placed = stake_record > 0
    mean_stake = float(stake_record[placed].mean()) if placed.any() else 0.0
    return {
        "config": {
            "edge_threshold": edge_threshold,
            "ttd_max_sec": ttd_max_sec,
            "mip_max": mip_max,
            "starting_bankroll": starting_bankroll,
            "fraction_cap": fraction_cap,
        },
        "triggers": n_trig,
        "wins": wins,
        "hit_rate": float(wins / max(1, n_trig)),
        "final_bankroll": float(bankroll),
        "total_pnl_usd": float(bankroll - starting_bankroll),
        "return_pct": float((bankroll / starting_bankroll - 1) * 100),
        "min_bankroll": float(min_bank),
        "max_drawdown_usd": float(min_bank - starting_bankroll),
        "mean_stake_when_placed": mean_stake,
        "placed_trades": int(placed.sum()),
        "pnl_curve": cum.tolist(),
    }


def backtest_clipped(
    df_te: pd.DataFrame,
    p_te: np.ndarray,
    mip_te: np.ndarray,
    y_te: np.ndarray,
    *,
    edge_threshold: float,
    ttd_max_sec: float | None = None,
    mip_max: float | None = None,
    stake_usd: float = 100.0,
    mip_floor: float = 0.05,
) -> dict:
    """Flat-stake backtest with `mip` clipped at `mip_floor` on the
    payoff-ratio side. Bounds the unrealistic long-shot PnL blow-up from
    v1's raw binary-option math.
    """
    mip_clip = np.clip(mip_te, mip_floor, 1 - mip_floor)
    return V1.backtest_rule(
        df_te, p_te, mip_clip, y_te,
        edge_threshold=edge_threshold,
        ttd_max_sec=ttd_max_sec, mip_max=mip_max,
        stake_usd=stake_usd,
    )


# ---------------------------------------------------------------------------
# 5. Wallet-level bootstrap 95 % CIs
# ---------------------------------------------------------------------------
def wallet_bootstrap_ci(
    df: pd.DataFrame,
    p: np.ndarray,
    y: np.ndarray,
    metric_name: str,
    metric_fn,
    *,
    n_boot: int = BOOTSTRAP_ROUNDS,
    seed: int = RANDOM_SEED,
) -> dict:
    """Resample proxyWallet with replacement; pool the resampled rows; apply
    `metric_fn`. Returns mean + 2.5 / 97.5 percentile band.
    """
    rng = np.random.default_rng(seed)
    wallets = df["proxyWallet"].to_numpy()
    unique = np.asarray(list(set(wallets.tolist())))
    wallet_to_rows: dict[str, np.ndarray] = {
        w: np.where(wallets == w)[0] for w in unique
    }

    base = float(metric_fn(y, p))
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        choice = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([wallet_to_rows[w] for w in choice])
        try:
            samples[b] = float(metric_fn(y[idx], p[idx]))
        except ValueError:
            samples[b] = np.nan
    samples = samples[~np.isnan(samples)]
    lo, hi = float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
    return {
        "metric": metric_name,
        "point": base,
        "bootstrap_mean": float(samples.mean()),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "n_boot_ok": int(len(samples)),
        "n_wallets": int(len(unique)),
    }


# ---------------------------------------------------------------------------
# 6. Magamyman sanity lookup
# ---------------------------------------------------------------------------
def magamyman_signature_check(
    cohorts: V1.Cohorts, p_te: np.ndarray
) -> dict:
    """Identify trades that match the Mitts & Ofir Magamyman signature on
    the test cohort:
      * wallet is newly funded (low prior_trades)
      * trade enters close to the deadline (ttd < 6h)
      * low market_implied_prob at entry (< 0.30)
      * large notional relative to the wallet's prior volume
    and report our model's p_hat rank for those rows.
    """
    te = cohorts.test.copy().reset_index(drop=True)
    te["p_hat"] = p_te

    te["_ttd_sec"] = (
        pd.to_datetime(te[DEADLINE_COL], utc=True, errors="coerce")
        - pd.to_datetime(te[TIME_COL], utc=True, errors="coerce")
    ).dt.total_seconds()

    mask = (
        (te["_ttd_sec"].fillna(0) < HOME_RUN_TTD_SEC)
        & (te[BENCHMARK].fillna(0.5) < HOME_RUN_MIP_MAX)
        & (te["wallet_prior_trades"].fillna(0) < 20)
        & (te["trade_value_usd"].fillna(0) > 1000)
    )
    hits = te[mask].copy()
    if len(hits) == 0:
        return {
            "flagged_rows": 0,
            "note": "no test-cohort rows match the Magamyman signature",
        }

    # p_hat percentile rank among the full test cohort
    te_sorted = te["p_hat"].rank(pct=True)
    hits["p_hat_pct_rank"] = te_sorted.loc[hits.index]
    return {
        "flagged_rows": int(len(hits)),
        "p_hat_mean_at_hits": float(hits["p_hat"].mean()),
        "p_hat_pct_rank_mean": float(hits["p_hat_pct_rank"].mean()),
        "p_hat_pct_rank_min": float(hits["p_hat_pct_rank"].min()),
        "p_hat_pct_rank_max": float(hits["p_hat_pct_rank"].max()),
        "bc_rate_at_hits": float(hits["bet_correct"].mean()),
        "top5_by_p_hat": hits.nlargest(5, "p_hat")[
            ["proxyWallet", "p_hat", "bet_correct", BENCHMARK, "_ttd_sec"]
        ].to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 7. Main orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # --- Load + preprocess -------------------------------------------------
    cohorts = V1.load_cohorts()
    V1.winsorise_train_fit(cohorts, WINSORISE_COLS)
    X_tr, X_va, X_te = V1.impute_and_scale(cohorts)
    y_tr = cohorts.train[TARGET].to_numpy().astype(np.int64)
    y_va = cohorts.val[TARGET].to_numpy().astype(np.int64)
    y_te = cohorts.test[TARGET].to_numpy().astype(np.int64)
    mip_tr = cohorts.train[BENCHMARK].to_numpy().astype(np.float64)
    mip_va = cohorts.val[BENCHMARK].to_numpy().astype(np.float64)
    mip_te = cohorts.test[BENCHMARK].to_numpy().astype(np.float64)
    groups_tr = cohorts.train["proxyWallet"].to_numpy()

    print(
        f"[shape] X_tr={X_tr.shape}  X_va={X_va.shape}  X_te={X_te.shape}  "
        f"n_wallets_train={len(set(groups_tr.tolist()))}"
    )

    V1.save_json(cohorts.features, OUT_DIR / "feature_list.json")

    # --- OOF stacking ------------------------------------------------------
    print(f"\n[stacking] GroupKFold OOF base predictions, {OOF_FOLDS} folds...")
    t0 = time.time()
    oof = oof_predictions(X_tr, y_tr, groups_tr, n_splits=OOF_FOLDS)
    print(f"[stacking] OOF done in {time.time() - t0:.0f}s")

    # Meta learners on OOF → y_tr: (a) all three bases, (b) logreg + mlp only
    # (skip RF after v2-run-1 showed the meta over-weighted it).
    meta_all, keys_all = fit_meta(oof, y_tr, keys=("logreg", "rf", "mlp"))
    meta_no_rf, keys_no_rf = fit_meta(oof, y_tr, keys=("logreg", "mlp"))

    # Refit base learners on FULL train, predict val + test. Use calibrated
    # RF to match the OOF training regime.
    print("[bases] refitting on full train...")
    m_lr = fit_logreg(X_tr, y_tr)
    m_rf = fit_rf_calibrated(X_tr, y_tr)
    m_mlp, mlp_hist = fit_mlp(X_tr, y_tr, X_va, y_va)

    p_tr_lr = m_lr.predict_proba(X_tr)[:, 1]
    p_va_lr = m_lr.predict_proba(X_va)[:, 1]
    p_te_lr = m_lr.predict_proba(X_te)[:, 1]

    p_tr_rf = m_rf.predict_proba(X_tr)[:, 1]
    p_va_rf = m_rf.predict_proba(X_va)[:, 1]
    p_te_rf = m_rf.predict_proba(X_te)[:, 1]

    p_tr_mlp = m_mlp.predict(X_tr, batch_size=MLP_BATCH, verbose=0).ravel()
    p_va_mlp = m_mlp.predict(X_va, batch_size=MLP_BATCH, verbose=0).ravel()
    p_te_mlp = m_mlp.predict(X_te, batch_size=MLP_BATCH, verbose=0).ravel()

    # Stacked predictions — both flavours
    base_tr = {"logreg": p_tr_lr, "rf": p_tr_rf, "mlp": p_tr_mlp}
    base_va = {"logreg": p_va_lr, "rf": p_va_rf, "mlp": p_va_mlp}
    base_te = {"logreg": p_te_lr, "rf": p_te_rf, "mlp": p_te_mlp}

    p_tr_stack_all = apply_meta(meta_all, base_tr, keys_all)
    p_va_stack_all = apply_meta(meta_all, base_va, keys_all)
    p_te_stack_all = apply_meta(meta_all, base_te, keys_all)

    p_tr_stack_no_rf = apply_meta(meta_no_rf, base_tr, keys_no_rf)
    p_va_stack_no_rf = apply_meta(meta_no_rf, base_va, keys_no_rf)
    p_te_stack_no_rf = apply_meta(meta_no_rf, base_te, keys_no_rf)

    # --- Dual calibration (isotonic vs Platt) via K-fold val ECE ---------
    # Applied separately to the two stacks so each gets its own calibrator
    # choice based on its own score distribution.
    cal_all_name, p_va_cal_all, p_te_cal_all, ece_iso_all, ece_platt_all = pick_calibrator_kfold(
        p_va_stack_all, y_va, p_te_stack_all, cohorts.val
    )
    cal_no_rf_name, p_va_cal_no_rf, p_te_cal_no_rf, ece_iso_no_rf, ece_platt_no_rf = pick_calibrator_kfold(
        p_va_stack_no_rf, y_va, p_te_stack_no_rf, cohorts.val
    )
    print(
        f"[calibration stack_all]    picked {cal_all_name}  "
        f"val_cv_ece_iso={ece_iso_all:.4f}  val_cv_ece_platt={ece_platt_all:.4f}"
    )
    print(
        f"[calibration stack_no_rf]  picked {cal_no_rf_name}  "
        f"val_cv_ece_iso={ece_iso_no_rf:.4f}  val_cv_ece_platt={ece_platt_no_rf:.4f}"
    )

    # Pick the "winning" stack by val ROC (NOT by train/test to avoid leak).
    roc_all_val = roc_auc_score(y_va, p_va_cal_all)
    roc_no_rf_val = roc_auc_score(y_va, p_va_cal_no_rf)
    if roc_no_rf_val > roc_all_val:
        winner = "stack_no_rf"
        p_tr_win = p_tr_stack_no_rf; p_va_win = p_va_cal_no_rf; p_te_win = p_te_cal_no_rf
        winner_cal_name = cal_no_rf_name
    else:
        winner = "stack_all"
        p_tr_win = p_tr_stack_all; p_va_win = p_va_cal_all; p_te_win = p_te_cal_all
        winner_cal_name = cal_all_name
    print(
        f"[calibration] winner by val ROC: {winner}  "
        f"(stack_all={roc_all_val:.4f}  stack_no_rf={roc_no_rf_val:.4f})"
    )
    # Keep this variable name for downstream code that expects it.
    cal_name = winner_cal_name
    p_va_stack = p_va_stack_all if winner == "stack_all" else p_va_stack_no_rf
    p_te_stack = p_te_stack_all if winner == "stack_all" else p_te_stack_no_rf
    p_va_cal = p_va_win; p_te_cal = p_te_win
    picked_ece = ece_iso_all if winner == "stack_all" else ece_iso_no_rf
    ece_iso = picked_ece

    # --- Metrics for every model -----------------------------------------
    all_preds = {
        "logreg":          {"train": p_tr_lr,         "val": p_va_lr,         "test": p_te_lr},
        "rf":              {"train": p_tr_rf,         "val": p_va_rf,         "test": p_te_rf},
        "mlp":             {"train": p_tr_mlp,        "val": p_va_mlp,        "test": p_te_mlp},
        "stack_all_raw":   {"train": p_tr_stack_all,  "val": p_va_stack_all,  "test": p_te_stack_all},
        "stack_no_rf_raw": {"train": p_tr_stack_no_rf,"val": p_va_stack_no_rf,"test": p_te_stack_no_rf},
        "stack_all_cal":   {"train": p_tr_stack_all,  "val": p_va_cal_all,    "test": p_te_cal_all},
        "stack_no_rf_cal": {"train": p_tr_stack_no_rf,"val": p_va_cal_no_rf,  "test": p_te_cal_no_rf},
        "stack_chosen":    {"train": p_tr_win,        "val": p_va_win,        "test": p_te_win},
    }

    mdir_root = OUT_DIR / "modelling"
    for name, preds in all_preds.items():
        mdir = mdir_root / name
        mdir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "train": V1.probability_metrics(y_tr, preds["train"]),
            "val":   V1.probability_metrics(y_va, preds["val"]),
            "test":  V1.probability_metrics(y_te, preds["test"]),
            "per_market_test": V1.per_market_breakdown(cohorts.test, preds["test"], y_te),
        }
        V1.save_json(metrics, mdir / "metrics.json")
        V1.save_predictions(cohorts, preds, mdir)
        t = metrics["test"]
        print(f"[metrics {name:<14s}]  test roc={t['roc_auc']}  "
              f"brier={t['brier']:.4f}  ece={t['ece_15bin']:.4f}")

    # Save OOF block for audit
    oof_df = pd.DataFrame(
        {"y_train": y_tr, **{f"oof_{k}": v for k, v in oof.items()}}
    )
    oof_df.to_parquet(mdir_root / "stack_chosen" / "oof_predictions.parquet", index=False)
    V1.save_json(
        {
            "winner": winner,
            "calibrator": cal_name,
            "meta_all_coefs": {
                keys_all[i]: float(meta_all.coef_[0, i]) for i in range(len(keys_all))
            } | {"intercept": float(meta_all.intercept_[0])},
            "meta_no_rf_coefs": {
                keys_no_rf[i]: float(meta_no_rf.coef_[0, i]) for i in range(len(keys_no_rf))
            } | {"intercept": float(meta_no_rf.intercept_[0])},
            "val_roc_stack_all": float(roc_all_val),
            "val_roc_stack_no_rf": float(roc_no_rf_val),
            "val_cv_ece_iso_stack_all": ece_iso_all,
            "val_cv_ece_platt_stack_all": ece_platt_all,
            "val_cv_ece_iso_stack_no_rf": ece_iso_no_rf,
            "val_cv_ece_platt_stack_no_rf": ece_platt_no_rf,
        },
        mdir_root / "stack_chosen" / "meta.json",
    )

    # Plots for the chosen model
    V1.plot_loss_curve(mlp_hist, "Base MLP BCE loss", mdir_root / "mlp" / "loss_curve.png")
    V1.plot_calibration(
        p_te_cal, y_te, f"Stacked + {cal_name} — test",
        mdir_root / "stack_chosen" / "calibration.png",
    )

    # --- Permutation importance (Lecture 14 XAI) --------------------------
    print("\n[xai] permutation importance on val (5 repeats/feature)...")
    t0 = time.time()

    # Cache the chosen calibrator so the importance loop doesn't refit on
    # every permutation.
    if cal_name == "isotonic":
        _cal = IsotonicRegression(out_of_bounds="clip").fit(p_va_stack, y_va)
        _apply_cal = _cal.transform
    else:
        _cal = PlattCalibrator().fit(p_va_stack, y_va)
        _apply_cal = _cal.transform

    _winner_meta = meta_all if winner == "stack_all" else meta_no_rf
    _winner_keys = keys_all if winner == "stack_all" else keys_no_rf

    def stacked_score(X):
        p_lr = m_lr.predict_proba(X)[:, 1]
        p_rf = m_rf.predict_proba(X)[:, 1]
        p_mlp = m_mlp.predict(X, batch_size=MLP_BATCH, verbose=0).ravel()
        base = {"logreg": p_lr, "rf": p_rf, "mlp": p_mlp}
        p_raw = apply_meta(_winner_meta, base, _winner_keys)
        return _apply_cal(p_raw)

    imp = permutation_importance(
        stacked_score, X_va, y_va, cohorts.features, n_repeats=5
    )
    perm_dir = OUT_DIR / "permutation_importance"
    perm_dir.mkdir(parents=True, exist_ok=True)
    imp.to_csv(perm_dir / "val_roc_decay.csv", index=False)
    V1.save_json(imp.head(20).to_dict(orient="records"), perm_dir / "top20.json")
    plot_importance(imp, top_k=20, out_path=perm_dir / "top20.png")
    print(f"[xai] done in {time.time() - t0:.0f}s  top feature: "
          f"{imp.iloc[0]['feature']} (Δroc={imp.iloc[0]['mean_drop']:+.4f})")

    # --- Residual-edge on the stacked model -------------------------------
    r_edge = V1.residual_edge_analysis(
        all_preds["stack_chosen"]["train"], mip_tr, y_tr,
        all_preds["stack_chosen"]["val"],   mip_va, y_va,
        all_preds["stack_chosen"]["test"],  mip_te, y_te,
    )
    (OUT_DIR / "residual_edge").mkdir(parents=True, exist_ok=True)
    V1.save_json(r_edge, OUT_DIR / "residual_edge" / "metrics.json")
    print(f"[residual-edge] test partial-corr = "
          f"{r_edge['test']['partial_corr_bc_vs_residual']:+.4f}  "
          f"(alex baseline: +0.06)")

    # --- Trading-rule backtest: 3 variants ------------------------------
    bt_dir = OUT_DIR / "backtest"
    bt_dir.mkdir(parents=True, exist_ok=True)

    bt_flat_general = V1.backtest_rule(
        cohorts.test, p_te_cal, mip_te, y_te, edge_threshold=GENERAL_EV_EDGE
    )
    bt_flat_homerun = V1.backtest_rule(
        cohorts.test, p_te_cal, mip_te, y_te,
        edge_threshold=HOME_RUN_EDGE,
        ttd_max_sec=HOME_RUN_TTD_SEC, mip_max=HOME_RUN_MIP_MAX,
    )
    bt_clip_general = backtest_clipped(
        cohorts.test, p_te_cal, mip_te, y_te, edge_threshold=GENERAL_EV_EDGE,
    )
    bt_clip_homerun = backtest_clipped(
        cohorts.test, p_te_cal, mip_te, y_te,
        edge_threshold=HOME_RUN_EDGE,
        ttd_max_sec=HOME_RUN_TTD_SEC, mip_max=HOME_RUN_MIP_MAX,
    )
    bt_kelly_general = backtest_kelly(
        cohorts.test, p_te_cal, mip_te, y_te, edge_threshold=GENERAL_EV_EDGE,
    )
    bt_kelly_homerun = backtest_kelly(
        cohorts.test, p_te_cal, mip_te, y_te,
        edge_threshold=HOME_RUN_EDGE,
        ttd_max_sec=HOME_RUN_TTD_SEC, mip_max=HOME_RUN_MIP_MAX,
    )

    V1.save_json(bt_flat_general, bt_dir / "flat_general_ev.json")
    V1.save_json(bt_flat_homerun, bt_dir / "flat_home_run.json")
    V1.save_json(bt_clip_general, bt_dir / "clipped_general_ev.json")
    V1.save_json(bt_clip_homerun, bt_dir / "clipped_home_run.json")
    V1.save_json(bt_kelly_general, bt_dir / "kelly_general_ev.json")
    V1.save_json(bt_kelly_homerun, bt_dir / "kelly_home_run.json")
    V1.plot_pnl_curves(
        {
            "kelly_general": bt_kelly_general,
            "kelly_home_run": bt_kelly_homerun,
            "clipped_general": bt_clip_general,
            "clipped_home_run": bt_clip_homerun,
        },
        bt_dir / "pnl_curve.png",
    )
    print(
        f"[backtest kelly]  general PnL ${bt_kelly_general['total_pnl_usd']:,.0f}  "
        f"({bt_kelly_general['return_pct']:+.1f}%)   "
        f"home_run PnL ${bt_kelly_homerun['total_pnl_usd']:,.0f}  "
        f"({bt_kelly_homerun['return_pct']:+.1f}%)"
    )
    print(
        f"[backtest clipped] general PnL ${bt_clip_general['total_pnl_usd']:,.0f}   "
        f"home_run PnL ${bt_clip_homerun['total_pnl_usd']:,.0f}"
    )

    # --- Wallet-level bootstrap CIs on test ROC / Brier -----------------
    print("\n[bootstrap] wallet-level 95% CI on test metrics...")
    ci_roc = wallet_bootstrap_ci(
        cohorts.test, p_te_cal, y_te, "roc_auc", roc_auc_score
    )
    ci_brier = wallet_bootstrap_ci(
        cohorts.test, p_te_cal, y_te, "brier", brier_score_loss
    )
    bdir = OUT_DIR / "bootstrap"
    bdir.mkdir(parents=True, exist_ok=True)
    V1.save_json(ci_roc, bdir / "roc_ci.json")
    V1.save_json(ci_brier, bdir / "brier_ci.json")
    print(
        f"[bootstrap] test ROC = {ci_roc['point']:.4f}  "
        f"95% CI [{ci_roc['ci95_lo']:.4f}, {ci_roc['ci95_hi']:.4f}] "
        f"across {ci_roc['n_wallets']:,} wallets"
    )

    # --- Autoencoder + Isolation Forest + anomaly overlap ----------------
    ae, ae_hist = V1.fit_autoencoder(X_tr, X_va)
    recon_te = ae.predict(X_te, batch_size=V1.AE_BATCH, verbose=0)
    err_te = ((X_te - recon_te) ** 2).mean(axis=1)
    ae_dir = OUT_DIR / "autoencoder"
    ae_dir.mkdir(parents=True, exist_ok=True)
    V1.save_json(
        {
            "test_mse_mean": float(err_te.mean()),
            "test_mse_p95": float(np.quantile(err_te, 0.95)),
            "test_mse_p99": float(np.quantile(err_te, 0.99)),
            "epochs_trained": len(ae_hist["loss"]),
        },
        ae_dir / "metrics.json",
    )
    V1.plot_loss_curve(ae_hist, "Autoencoder MSE", ae_dir / "loss_curve.png")

    iso_f = V1.fit_isolation_forest(X_tr)
    iso_score_te = iso_f.score_samples(X_te)
    if_dir = OUT_DIR / "isolation_forest"
    if_dir.mkdir(parents=True, exist_ok=True)
    V1.save_json(
        {"mean_score_test": float(iso_score_te.mean())},
        if_dir / "metrics.json",
    )

    edge_te = p_te_cal - mip_te
    overlap = V1.anomaly_overlap(edge_te, err_te, iso_score_te, top_frac=0.10)
    (OUT_DIR / "overlap").mkdir(parents=True, exist_ok=True)
    V1.save_json(overlap, OUT_DIR / "overlap" / "metrics.json")
    print(
        f"[overlap] |edge|∩recon at top-10% = "
        f"{overlap['overlap_edge_vs_autoencoder']} vs null "
        f"{overlap['random_null_overlap']:.0f} "
        f"(×{overlap['uplift_edge_vs_autoencoder_x']:.2f})"
    )

    # --- Magamyman sanity check ------------------------------------------
    mm = magamyman_signature_check(cohorts, p_te_cal)
    (OUT_DIR / "magamyman").mkdir(parents=True, exist_ok=True)
    V1.save_json(mm, OUT_DIR / "magamyman" / "check.json")
    print(f"[magamyman] flagged_rows={mm['flagged_rows']}"
          + (f"  p_hat_pct_rank_mean={mm.get('p_hat_pct_rank_mean', 0):.3f}"
             if mm["flagged_rows"] else ""))

    # --- Summary ---------------------------------------------------------
    summary = {
        "runtime_min": round((time.time() - t_start) / 60, 2),
        "feature_count": len(cohorts.features),
        "cohorts": {
            "train_rows": len(cohorts.train),
            "val_rows": len(cohorts.val),
            "test_rows": len(cohorts.test),
            "train_wallets": len(set(groups_tr.tolist())),
        },
        "test_roc": {
            name: V1.probability_metrics(y_te, preds["test"])["roc_auc"]
            for name, preds in all_preds.items()
        },
        "stacking_winner": winner,
        "test_roc_ci95": [ci_roc["ci95_lo"], ci_roc["ci95_hi"]],
        "residual_edge_partial_corr_test": r_edge["test"]["partial_corr_bc_vs_residual"],
        "calibrator": cal_name,
        "pnl_kelly_home_run": bt_kelly_homerun["total_pnl_usd"],
        "pnl_clipped_home_run": bt_clip_homerun["total_pnl_usd"],
        "permutation_top3": imp.head(3)[["feature", "mean_drop"]].to_dict(orient="records"),
        "magamyman_flagged_rows": mm["flagged_rows"],
    }
    V1.save_json(summary, OUT_DIR / "run_summary.json")
    print(f"\n[done] {summary['runtime_min']:.1f} min total; "
          f"stack_chosen test ROC = {summary['test_roc']['stack_chosen']:.4f}  "
          f"(95% CI {ci_roc['ci95_lo']:.4f}–{ci_roc['ci95_hi']:.4f})")


if __name__ == "__main__":
    main()
