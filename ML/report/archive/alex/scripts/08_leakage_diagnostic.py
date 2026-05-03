"""
08_leakage_diagnostic.py

Three checks to confirm the market-identity leakage hypothesis from the v3 sweep
(trees scoring AUC 0.85+ with bimodal fold AUCs and per-market AUC of [0.0, 1.0]).

Hypothesis: tree models can identify which market each trade belongs to via
features that are constant or near-constant within a market. The strongest
suspect is `kyle_lambda_market_static` (literally one value per market),
followed by 24h-window rolling features and per-market-state features that
have low intra-market variance.

CHECK 1 — Single-feature attack:
  Train RF on `kyle_lambda_market_static` alone. If test AUC > 0.6, this
  feature alone encodes market identity. If AUC > 0.7, smoking gun.

CHECK 2 — Drop the prime suspect:
  Train RF on (full set − kyle_lambda_market_static). If AUC drops from
  0.88 toward 0.62 (LogReg L2), kyle is the dominant leak channel.

CHECK 3 — Drop the suspect family:
  Train RF on (full set − {kyle_lambda, all *_24h features, sister_price_dispersion,
  recent_price_high_1h, recent_price_low_1h}). Expected AUC: 0.65-0.75. If still
  > 0.85, more leaks remain.

Outputs:
  alex/outputs/leakage_diagnostic/
    check1_single_feature/metrics.json
    check2_drop_kyle/metrics.json
    check3_drop_suspect_family/metrics.json
    summary.md

Each model is the SAME RF config we used in the sweep (200 trees, depth 10)
so results are directly comparable.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "leakage_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
RANDOM_SEED = 42

# Suspect feature family for Check 3
SUSPECT_FEATURES = {
    "kyle_lambda_market_static",
    "log_recent_volume_24h",
    "log_trade_count_24h",
    "market_price_vol_last_24h",
    "order_flow_imbalance_24h",
    "recent_price_mean_24h",
    "pre_trade_price_change_24h",
    "recent_price_high_1h",
    "recent_price_low_1h",
    "recent_price_range_1h",
    "sister_price_dispersion_5min",
}


def make_rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=200,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def cv_score(X, y, groups, n_folds=N_FOLDS):
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(y), dtype=float)
    fold_aucs = []
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        clf = make_rf()
        clf.fit(X.iloc[tr_idx].values, y.iloc[tr_idx])
        preds = clf.predict_proba(X.iloc[va_idx].values)[:, 1]
        oof[va_idx] = preds
        auc = roc_auc_score(y.iloc[va_idx], preds)
        fold_aucs.append(auc)
        print(f"    fold {fold_idx + 1}: AUC = {auc:.4f}")
    return oof, fold_aucs


def fit_and_score_test(X_train, y_train, X_test, y_test, test_df):
    clf = make_rf()
    clf.fit(X_train.values, y_train)
    raw = clf.predict_proba(X_test.values)[:, 1]
    test_auc = float(roc_auc_score(y_test, raw))
    test_brier = float(brier_score_loss(y_test, raw))
    # Per-market range
    per_market_aucs = []
    for mid in test_df["market_id"].unique():
        mask = (test_df["market_id"] == mid).values
        if mask.sum() < 20 or len(np.unique(y_test.values[mask])) < 2:
            continue
        per_market_aucs.append(float(roc_auc_score(y_test.values[mask], raw[mask])))
    pm_min = float(min(per_market_aucs)) if per_market_aucs else None
    pm_max = float(max(per_market_aucs)) if per_market_aucs else None
    return {
        "test_auc": test_auc,
        "test_brier": test_brier,
        "per_market_auc_min": pm_min,
        "per_market_auc_max": pm_max,
        "per_market_n_resolved": len(per_market_aucs),
        "feature_importances": dict(
            sorted(
                zip(X_train.columns, clf.feature_importances_.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }


def run_check(
    name: str,
    feature_cols: list[str],
    X_train_full,
    y_train,
    g_train,
    X_test_full,
    y_test,
    test_df,
):
    print(f"\n{'=' * 60}\n[{name}] features: {len(feature_cols)}\n{'=' * 60}")
    if len(feature_cols) <= 5:
        print(f"  feature list: {feature_cols}")
    X_tr = X_train_full[feature_cols]
    X_te = X_test_full[feature_cols]

    print(f"  CV...")
    oof, fold_aucs = cv_score(X_tr, y_train, g_train)
    cv_oof_auc = float(roc_auc_score(y_train, oof))
    print(f"  OOF AUC: {cv_oof_auc:.4f} (folds: {[f'{a:.3f}' for a in fold_aucs]})")
    print(f"  fitting on full train + scoring test...")
    test_result = fit_and_score_test(X_tr, y_train, X_te, y_test, test_df)
    print(
        f"  test AUC: {test_result['test_auc']:.4f}, "
        f"per-market range: [{test_result['per_market_auc_min']:.3f}, "
        f"{test_result['per_market_auc_max']:.3f}]"
    )

    summary = {
        "check_name": name,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "cv_oof_auc": cv_oof_auc,
        "cv_fold_aucs": [float(a) for a in fold_aucs],
        "cv_fold_auc_mean": float(np.mean(fold_aucs)),
        "cv_fold_auc_std": float(np.std(fold_aucs)),
        **test_result,
    }
    out_dir = OUT / name
    out_dir.mkdir(exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    print("=" * 60)
    print("v3 sweep leakage diagnostic — RandomForest, 3 ablations")
    print("=" * 60)
    train = pd.read_parquet(DATA / "train_features.parquet")
    test = pd.read_parquet(DATA / "test_features.parquet")
    feature_cols = json.loads((DATA / "feature_cols.json").read_text())
    print(f"train: {train.shape}, test: {test.shape}, n_features: {len(feature_cols)}")

    X_train = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train["bet_correct"].astype(int)
    g_train = train["market_id"]
    X_test = test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test["bet_correct"].astype(int)

    summaries = []

    # CHECK 1: kyle_lambda_market_static alone
    summaries.append(
        run_check(
            "check1_single_feature",
            ["kyle_lambda_market_static"],
            X_train,
            y_train,
            g_train,
            X_test,
            y_test,
            test,
        )
    )

    # CHECK 2: drop kyle_lambda only
    cols_minus_kyle = [c for c in feature_cols if c != "kyle_lambda_market_static"]
    summaries.append(
        run_check(
            "check2_drop_kyle",
            cols_minus_kyle,
            X_train,
            y_train,
            g_train,
            X_test,
            y_test,
            test,
        )
    )

    # CHECK 3: drop the full suspect family
    cols_clean = [c for c in feature_cols if c not in SUSPECT_FEATURES]
    summaries.append(
        run_check(
            "check3_drop_suspect_family",
            cols_clean,
            X_train,
            y_train,
            g_train,
            X_test,
            y_test,
            test,
        )
    )

    # Summary report
    lines = ["# Leakage diagnostic summary\n", ""]
    lines.append(
        "| check | n_features | OOF AUC | fold AUCs | test AUC | per-market AUC range |"
    )
    lines.append("|---|---:|---:|---|---:|---|")
    for s in summaries:
        folds = ", ".join(f"{a:.3f}" for a in s["cv_fold_aucs"])
        lines.append(
            f"| {s['check_name']} | {s['n_features']} | {s['cv_oof_auc']:.3f} | "
            f"[{folds}] | {s['test_auc']:.3f} | [{s['per_market_auc_min']:.2f}, {s['per_market_auc_max']:.2f}] |"
        )
    lines.append("\n## Reference (v3 sweep, full feature set)")
    lines.append("- LogReg L2: OOF 0.623, test 0.615, per-market [0.42, 0.74] — clean")
    lines.append(
        "- Random Forest (full set): OOF 0.867, test 0.884, per-market [0.03, 1.00] — leaky"
    )
    lines.append("\n## Suspect feature family (Check 3 drops these)")
    for f in sorted(SUSPECT_FEATURES):
        lines.append(f"- `{f}`")

    summary_md = "\n".join(lines)
    (OUT / "summary.md").write_text(summary_md)
    print("\n" + summary_md)
    print(f"\noutputs: {OUT}")


if __name__ == "__main__":
    main()
