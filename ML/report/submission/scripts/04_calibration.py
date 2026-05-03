"""
04_calibration.py — Isotonic calibration, reliability diagrams, bootstrap CI, permutation importance.

Source-of-truth merge of:
  alex/v5_final_ml_pipeline/scripts/07_rigor_additions.py  (was a skeleton — implemented here)
  alex/v5_final_ml_pipeline/scripts/09_shap_top_picks.py   (was a skeleton — substituted with
                                                             sklearn permutation importance,
                                                             which does not need pickled models)

Steps:
  1. Load each model's out-of-fold predictions saved by 03_train_models.py.
  2. Fit isotonic regression on (oof, y_train) -> the calibrator.
  3. Apply calibrator to raw test predictions -> calibrated test predictions.
  4. Compare Brier and ECE before vs after calibration.
  5. Save reliability diagrams (one figure per model + one combined) for the report.
  6. Bootstrap 95% CI on test AUC for every model (1,000 resamples).
  7. Permutation importance on the best model's top-15 features.

Run:
  python 04_calibration.py

Outputs:
  outputs/models/<name>/preds_test_cal.npz                  calibrated test predictions
  outputs/metrics/calibration_summary.csv                    raw vs calibrated Brier/ECE per model
  outputs/metrics/auc_bootstrap_ci.csv                       AUC + 95% CI per model
  outputs/metrics/permutation_importance_<best>.csv          top-15 features for the best model
  outputs/metrics/reliability_<name>.png                     per-model reliability diagram
  outputs/metrics/reliability_combined.png                   all models on one chart
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, OUTPUTS_DIR, RANDOM_SEED  # noqa: E402

TARGET = "bet_correct"
N_BOOTSTRAP = 1_000
PERM_N_REPEATS = 5


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Same definition as in 03_train_models.py — bucket gap × bucket weight."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def reliability_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_predicted_prob_per_bin, empirical_pos_rate_per_bin, count_per_bin)."""
    # what: build the data points behind a reliability diagram
    # how: digitise into n_bins, take per-bin mean of probs and per-bin empirical positive rate
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    mean_pred, emp_rate, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            mean_pred.append(float(y_prob[mask].mean()))
            emp_rate.append(float(y_true[mask].mean()))
            counts.append(int(mask.sum()))
        else:
            mean_pred.append(np.nan); emp_rate.append(np.nan); counts.append(0)
    return np.array(mean_pred), np.array(emp_rate), np.array(counts)


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_iter: int = N_BOOTSTRAP,
                     seed: int = RANDOM_SEED) -> tuple[float, float, float]:
    """Resampling 95% CI on test AUC."""
    # what: standard nonparametric bootstrap; resample row indices with replacement, recompute AUC
    # why: gives a confidence interval the report can cite alongside the point estimate
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        # how: skip degenerate resamples (all-positive or all-negative); rare but possible
        if len(np.unique(y_true[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_true[idx], y_prob[idx])
    aucs = aucs[~np.isnan(aucs)]
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ----------------------------------------------------------------------------
# Calibration loop
# ----------------------------------------------------------------------------

def calibrate_one(name: str, y_train: pd.Series, y_test: pd.Series,
                   models_dir: Path) -> dict | None:
    """Fit isotonic on (oof, y_train), apply to raw test, save calibrated preds."""
    # what: read OOF + raw test preds saved by 03_train_models
    model_dir = models_dir / name
    oof_path = model_dir / "preds_oof.npy"
    raw_path = model_dir / "preds_test.npz"
    if not oof_path.exists() or not raw_path.exists():
        print(f"  [{name}] missing predictions, skipping")
        return None
    oof = np.load(oof_path)
    raw = np.load(raw_path)["raw"]

    # what: fit isotonic on the train-side OOF; apply to test
    # why: monotone, distribution-free recalibration; well-suited to tree-based scores that are not probabilities
    cal_model = IsotonicRegression(out_of_bounds="clip").fit(oof, y_train.values)
    cal = cal_model.transform(raw)
    np.savez_compressed(model_dir / "preds_test_cal.npz", cal=cal.astype("float32"))
    joblib.dump(cal_model, model_dir / "isotonic.joblib")

    # what: report Brier + ECE before and after calibration
    # why: calibration should reduce Brier and ECE; if it does not, the model's ranking is good but its scores are uncalibrated
    return {
        "model": name,
        "test_auc_raw": float(roc_auc_score(y_test.values, raw)),
        "test_auc_cal": float(roc_auc_score(y_test.values, cal)),
        "brier_raw": float(brier_score_loss(y_test.values, raw)),
        "brier_cal": float(brier_score_loss(y_test.values, cal)),
        "ece_raw": expected_calibration_error(y_test.values, raw),
        "ece_cal": expected_calibration_error(y_test.values, cal),
    }


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def plot_reliability(y_true: np.ndarray, y_raw: np.ndarray, y_cal: np.ndarray,
                      name: str, out_path: Path) -> None:
    """One reliability diagram per model: raw vs calibrated vs perfect line."""
    mp_raw, er_raw, _ = reliability_points(y_true, y_raw)
    mp_cal, er_cal, _ = reliability_points(y_true, y_cal)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.plot(mp_raw, er_raw, "o-", color="#d62728", label="raw")
    ax.plot(mp_cal, er_cal, "s-", color="#2ca02c", label="isotonic-calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(f"Reliability diagram — {name}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_reliability_combined(y_true: np.ndarray, model_to_cal: dict[str, np.ndarray],
                                out_path: Path) -> None:
    """All models on one reliability diagram (calibrated only) for the report figure."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    for name, cal in model_to_cal.items():
        mp, er, _ = reliability_points(y_true, cal)
        ax.plot(mp, er, "o-", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Reliability diagram — all models (post calibration)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Permutation importance for the best model
# ----------------------------------------------------------------------------

def permutation_importance_top_k(best_model_name: str, k: int = 15) -> pd.DataFrame | None:
    """Refit best model on full train, run sklearn permutation_importance on test, return top-k."""
    # what: identify best model from comparison.csv; refit; permute each feature on test; record AUC drop
    # why: model-agnostic feature importance (works for trees, linear, MLP) — answers "which features matter?"
    feature_cols = json.loads((OUTPUTS_DIR / "data" / "feature_cols.json").read_text())
    df = pd.read_parquet(DATA_DIR / "consolidated_modeling_data.parquet")
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    # what: re-instantiate best model with default tuned-down settings (matches 03_train_models)
    # how: import the matching factory from 03_train_models lazily to avoid duplication
    from importlib import util as _u
    sweep_path = Path(__file__).parent / "03_train_models.py"
    spec = _u.spec_from_file_location("train_models_mod", sweep_path)
    mod = _u.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    pca_k = mod.pca_elbow_k(X_train)
    factories = {n: (f, s) for n, f, s in mod.make_factories(pca_k)}
    if best_model_name not in factories:
        print(f"  best model {best_model_name} not in factories; falling back to random_forest")
        best_model_name = "random_forest"
    factory, scale = factories[best_model_name]

    if scale:
        scaler = StandardScaler().fit(X_train)
        X_tr_arr = scaler.transform(X_train)
        X_te_arr = scaler.transform(X_test)
    else:
        scaler = None
        X_tr_arr = X_train.values
        X_te_arr = X_test.values
    clf = factory().fit(X_tr_arr, y_train)

    # what: permute each feature 5 times on test, measure AUC drop
    # why: average drop = the feature's marginal contribution; not just split-count importance
    print(f"  running permutation importance on {best_model_name} ({PERM_N_REPEATS} repeats)...")
    result = permutation_importance(clf, X_te_arr, y_test.values, n_repeats=PERM_N_REPEATS,
                                     random_state=RANDOM_SEED, n_jobs=-1, scoring="roc_auc")
    imp = pd.DataFrame({"feature": feature_cols,
                        "auc_drop_mean": result.importances_mean,
                        "auc_drop_std": result.importances_std})
    imp = imp.sort_values("auc_drop_mean", ascending=False).head(k)
    return imp


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Stage 4 — Calibration, reliability, bootstrap CI, permutation importance")
    print("=" * 60)
    metrics_dir = OUTPUTS_DIR / "metrics"
    models_dir = OUTPUTS_DIR / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # what: load labels (only y_train and y_test needed for calibration loop)
    df = pd.read_parquet(DATA_DIR / "consolidated_modeling_data.parquet")
    y_train = df[df["split"] == "train"][TARGET].astype(int).reset_index(drop=True)
    y_test = df[df["split"] == "test"][TARGET].astype(int).reset_index(drop=True)

    # what: discover which models 03_train_models actually produced predictions for
    model_names = sorted([p.name for p in models_dir.iterdir() if p.is_dir()])
    print(f"  found {len(model_names)} models with predictions: {model_names}")

    # what: calibrate every model and collect the summary rows
    cal_rows: list[dict] = []
    cal_preds_for_combined: dict[str, np.ndarray] = {}
    for name in model_names:
        row = calibrate_one(name, y_train, y_test, models_dir)
        if row is None:
            continue
        cal_rows.append(row)
        # what: per-model reliability plot
        raw = np.load(models_dir / name / "preds_test.npz")["raw"]
        cal = np.load(models_dir / name / "preds_test_cal.npz")["cal"]
        plot_reliability(y_test.values, raw, cal, name, metrics_dir / f"reliability_{name}.png")
        cal_preds_for_combined[name] = cal
        print(f"  [{name}]  AUC raw={row['test_auc_raw']:.4f} cal={row['test_auc_cal']:.4f}  "
              f"Brier raw={row['brier_raw']:.4f} -> cal={row['brier_cal']:.4f}  "
              f"ECE raw={row['ece_raw']:.4f} -> cal={row['ece_cal']:.4f}")

    # what: combined reliability figure for the report
    if cal_preds_for_combined:
        plot_reliability_combined(y_test.values, cal_preds_for_combined,
                                   metrics_dir / "reliability_combined.png")

    cal_df = pd.DataFrame(cal_rows).sort_values("test_auc_cal", ascending=False)
    cal_df.to_csv(metrics_dir / "calibration_summary.csv", index=False)

    # what: bootstrap 95% CI on test AUC for the calibrated predictions
    print("\nBootstrapping 95% CI on test AUC ...")
    ci_rows = []
    for name in model_names:
        cal_path = models_dir / name / "preds_test_cal.npz"
        if not cal_path.exists():
            continue
        cal = np.load(cal_path)["cal"]
        mean_auc, lo, hi = bootstrap_auc_ci(y_test.values, cal)
        ci_rows.append({"model": name, "test_auc_cal_mean": mean_auc,
                        "ci_lower_2_5": lo, "ci_upper_97_5": hi,
                        "ci_width": hi - lo})
        print(f"  {name:14s}  AUC = {mean_auc:.4f}  CI = [{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(ci_rows).to_csv(metrics_dir / "auc_bootstrap_ci.csv", index=False)

    # what: permutation importance on the best model (head of the cal_df)
    if not cal_df.empty:
        best = cal_df.iloc[0]["model"]
        print(f"\nBest model by calibrated AUC: {best}")
        imp = permutation_importance_top_k(best)
        if imp is not None:
            imp.to_csv(metrics_dir / f"permutation_importance_{best}.csv", index=False)
            print(imp.to_string(index=False))

    print(f"\nStage 4 complete. Outputs in {metrics_dir.relative_to(OUTPUTS_DIR.parent)}.")
    print("Proceed to 05_backtest.py.")
    return 0


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    sys.exit(main())
