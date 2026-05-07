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
N_BOOTSTRAP = 500
PERM_N_REPEATS = 3


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


def paired_bootstrap_auc_diff(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_iter: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict:
    """Paired bootstrap test on the AUC difference between two models.

    what: resample BOTH prediction arrays with the same indices (paired) and
          measure AUC(m_a) - AUC(m_b) on each resample.
    why:  the per-model bootstrap CIs don't tell us whether the gap between
          two models is statistically real — paired resampling accounts for
          correlation in errors across models.
    how:  p_value = 2 * min(P(diff <= 0), P(diff >= 0)), two-tailed.
          Degenerate resamples (single class) are skipped to keep the estimator
          unbiased — identical to the guard in bootstrap_auc_ci.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs: list[float] = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_s = y_true[idx]
        if len(np.unique(y_s)) < 2:
            continue
        auc_a = roc_auc_score(y_s, p_a[idx])
        auc_b = roc_auc_score(y_s, p_b[idx])
        diffs.append(auc_a - auc_b)
    diffs_arr = np.array(diffs)
    mean_diff = float(np.mean(diffs_arr))
    ci_lower = float(np.percentile(diffs_arr, 2.5))
    ci_upper = float(np.percentile(diffs_arr, 97.5))
    p_val = float(2.0 * min((diffs_arr <= 0).mean(), (diffs_arr >= 0).mean()))
    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
    }


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


def shap_on_top_picks(best_model_name: str, top_k_pct: float = 0.01) -> None:
    """SHAP analysis restricted to the top-1% highest-confidence test predictions.

    what: compute SHAP values only for the subset of test rows where the best
          calibrated model had the highest predicted probabilities.
    why:  global permutation importance averages over all rows; SHAP on the
          top-1% picks answers "what drove the model's confidence on the bets
          it actually wanted to make?" — closer to Lecture 14 (Explainable AI)
          requirements for a report-relevant explanation.
    how:  use shap.TreeExplainer (fast, exact for tree ensembles); fall back to
          the best non-MLP model if the overall best is MLP.
          For binary classification TreeExplainer returns a list [neg, pos];
          take element [1] (positive class).
    """
    try:
        import shap  # noqa: PLC0415
    except ImportError:
        print("  [shap_on_top_picks] shap not installed — skipping (pip install shap)")
        return

    TREE_MODELS = {"decision_tree", "random_forest", "hist_gbm", "lightgbm"}

    metrics_dir = OUTPUTS_DIR / "metrics"
    models_dir = OUTPUTS_DIR / "models"

    # what: pick the right model — prefer best, but fall back if MLP
    if best_model_name not in TREE_MODELS:
        # what: scan cal_df ordering to find next-best tree model
        cal_path = metrics_dir / "calibration_summary.csv"
        if not cal_path.exists():
            print("  [shap_on_top_picks] calibration_summary.csv not found — skipping")
            return
        ordered = pd.read_csv(cal_path).sort_values("test_auc_cal", ascending=False)["model"].tolist()
        candidates = [m for m in ordered if m in TREE_MODELS]
        if not candidates:
            print("  [shap_on_top_picks] no tree-based model found — skipping")
            return
        model_name = candidates[0]
        print(f"  [shap_on_top_picks] best model is {best_model_name} (not tree); "
              f"falling back to {model_name}")
    else:
        model_name = best_model_name

    # what: reload features and data (same pipeline as permutation_importance_top_k)
    feature_cols = json.loads((OUTPUTS_DIR / "data" / "feature_cols.json").read_text())
    df = pd.read_parquet(DATA_DIR / "consolidated_modeling_data.parquet")
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train[TARGET].astype(int)

    from importlib import util as _u
    sweep_path = Path(__file__).parent / "03_train_models.py"
    spec = _u.spec_from_file_location("train_models_mod", sweep_path)
    mod = _u.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    pca_k = mod.pca_elbow_k(X_train)
    factories = {n: (f, s) for n, f, s in mod.make_factories(pca_k)}

    if model_name not in factories:
        print(f"  [shap_on_top_picks] {model_name} not in factories — skipping")
        return
    factory, scale = factories[model_name]

    if scale:
        scaler = StandardScaler().fit(X_train)
        X_tr_arr = scaler.transform(X_train)
        X_te_arr = scaler.transform(X_test)
    else:
        X_tr_arr = X_train.values
        X_te_arr = X_test.values

    clf = factory().fit(X_tr_arr, y_train)

    # what: select top-1% test rows by calibrated probability
    cal_npz = models_dir / model_name / "preds_test_cal.npz"
    if not cal_npz.exists():
        print(f"  [shap_on_top_picks] calibrated preds for {model_name} not found — skipping")
        return
    cal_probs = np.load(cal_npz)["cal"]
    k = max(1, int(np.ceil(top_k_pct * len(cal_probs))))
    top_idx = np.argsort(cal_probs)[-k:]
    X_top = X_te_arr[top_idx]

    print(f"  [shap_on_top_picks] running SHAP on {k} top-{top_k_pct:.0%} picks for {model_name}...")
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_top)
    # how: binary classification returns list [neg_class, pos_class]; take pos_class
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # what: save summary bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_vals, X_top, feature_names=feature_cols, plot_type="bar", show=False)
    plt.title(f"SHAP (top {top_k_pct:.0%} picks) — {model_name}")
    plt.tight_layout()
    plot_path = metrics_dir / f"shap_summary_top1pct_{model_name}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [shap_on_top_picks] saved {plot_path.name}")

    # what: save CSV ranking by mean absolute SHAP
    mean_abs = np.abs(shap_vals).mean(axis=0)
    std_abs = np.abs(shap_vals).std(axis=0)
    ranking = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs, "std_shap": std_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = metrics_dir / f"shap_ranking_top1pct_{model_name}.csv"
    ranking.to_csv(csv_path, index=False)
    print(f"  [shap_on_top_picks] saved {csv_path.name}")


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

    # what: paired bootstrap AUC differences for all model pairs
    # why:  per-model CIs don't tell us if pairwise gaps are significant; paired
    #       resampling gives a proper test and Bonferroni-corrected p-values
    print("\nPaired bootstrap AUC differences ...")
    # how:  collect calibrated preds for all models that produced a cal file
    cal_map: dict[str, np.ndarray] = {}
    for name in model_names:
        cp = models_dir / name / "preds_test_cal.npz"
        if cp.exists():
            cal_map[name] = np.load(cp)["cal"]
    pairs = [(a, b) for i, a in enumerate(sorted(cal_map)) for b in sorted(cal_map)[i + 1:]]
    n_pairs = len(pairs)
    pair_rows = []
    for m_a, m_b in pairs:
        res = paired_bootstrap_auc_diff(y_test.values, cal_map[m_a], cal_map[m_b], n_iter=200)
        bonf = min(res["p_value"] * n_pairs, 1.0)
        pair_rows.append({
            "model_a": m_a,
            "model_b": m_b,
            "mean_auc_diff": res["mean_diff"],
            "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"],
            "p_value": res["p_value"],
            "p_value_bonferroni": bonf,
        })
        print(f"  {m_a} vs {m_b}: diff={res['mean_diff']:+.4f} "
              f"CI=[{res['ci_lower']:+.4f},{res['ci_upper']:+.4f}] "
              f"p={res['p_value']:.3f} p_bonf={bonf:.3f}")
    if pair_rows:
        pair_df = pd.DataFrame(pair_rows)
        pair_df.to_csv(metrics_dir / "auc_pairwise.csv", index=False)
        top5 = pair_df.assign(abs_diff=pair_df["mean_auc_diff"].abs()).sort_values(
            "abs_diff", ascending=False
        ).head(5).drop(columns="abs_diff")
        print("\nTop-5 pairs by |mean AUC diff|:")
        print(top5.to_string(index=False))

    # what: permutation importance on the best model (head of the cal_df)
    if not cal_df.empty:
        best = cal_df.iloc[0]["model"]
        print(f"\nBest model by calibrated AUC: {best}")
        imp = permutation_importance_top_k(best)
        if imp is not None:
            imp.to_csv(metrics_dir / f"permutation_importance_{best}.csv", index=False)
            print(imp.to_string(index=False))

    # what: SHAP on top-1% picks for the best (tree-based) model
    # why:  complements global permutation importance with a focused explainability
    #       view for the model's most confident predictions (Lecture 14 requirement)
    if not cal_df.empty:
        best = cal_df.iloc[0]["model"]
        print(f"\nSHAP on top-1% picks (best model: {best}) ...")
        shap_on_top_picks(best)

    print(f"\nStage 4 complete. Outputs in {metrics_dir.relative_to(OUTPUTS_DIR.parent)}.")
    print("Proceed to 05_backtest.py.")
    return 0


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    sys.exit(main())
