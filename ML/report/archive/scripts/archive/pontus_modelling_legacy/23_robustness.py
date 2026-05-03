"""
23_robustness.py

Three generalisation checks that accompany the primary market-cohort
results. Each addresses a concrete concern raised during the split-strategy
review in `pontus/notes/split-strategy.md`:

  A. **Market-identity audit**. Can a classifier identify which of the 74
     sub-markets a trade belongs to from the 36-feature vector alone?
     Random baseline is 1/74 = 1.35 %. If accuracy is low, the feature
     set is genuinely market-agnostic and the cross-family test ROC is
     a clean generalisation claim. If high, residual market-identity
     leakage caps the interpretation.

  B. **Random 20 % within-strike split**. Retrain LogReg + MLP on 80 %
     of the Feb 25-28 strike trades, test on the held-out 20 %. Every
     market appears in both folds so the model cannot use market
     identity as a shortcut. Contrasts with the cross-family ROC to
     quantify the difficulty of transferring to ceasefires.

  C. **Wallet-novel vs wallet-seen test subset**. Re-scores the saved
     stack-chosen test predictions split by whether the row's
     `proxyWallet` also appears in train. Cleanest generalisation
     number is the wallet-novel subset.

Outputs land in `pontus/outputs/v2/robustness/`:
  market_identity_audit.json, within_strike_split.json, wallet_novelty.json.

Framework: sklearn + tf.keras (as the rest of the pipeline, §0 of the
adventure).

Usage:
  python pontus/scripts/23_robustness.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Reuse v1 helpers (load, preprocess, MLP builder)
ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / "pontus" / "scripts" / "21_full_pipeline.py"
_spec = importlib.util.spec_from_file_location("pipeline_v1", V1_PATH)
V1 = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_v1"] = V1
_spec.loader.exec_module(V1)

OUT_DIR = ROOT / "pontus" / "outputs" / "v2" / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FULL = ROOT / "data" / "03_consolidated_dataset.csv"
TRAIN_PARQUET = ROOT / "data" / "experiments" / "train.parquet"
PRED_DIR = ROOT / "pontus" / "outputs" / "v2" / "modelling"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------
# A. Market-identity audit
# ---------------------------------------------------------------------------
def market_identity_audit(
    max_rows_per_market: int = 2000,
    n_estimators: int = 200,
) -> dict:
    print("\n[A] market-identity audit — 74-class RF on the feature vector")
    t0 = time.time()

    # Read only the columns we need to stay within memory budget
    usecols = (
        list(V1.NON_FEATURE_COLS) +
        [c for c in pd.read_csv(CSV_FULL, nrows=0).columns
         if c not in V1.NON_FEATURE_COLS]
    )
    df = pd.read_csv(CSV_FULL, usecols=usecols, low_memory=False)
    print(f"  loaded full CSV: {len(df):,} rows × {df.shape[1]} cols")

    # Keep only rows passing the §4 filter so we audit the modelling cohort,
    # not post-resolution close-outs.
    df = df[pd.to_numeric(df["settlement_minus_trade_sec"], errors="coerce") > 0].copy()
    print(f"  after §4 filter: {len(df):,} rows")

    # Stratified subsample: up to `max_rows_per_market` per condition_id
    parts = []
    rng = np.random.default_rng(RANDOM_SEED)
    for cid, g in df.groupby("condition_id", sort=False):
        if len(g) > max_rows_per_market:
            idx = rng.choice(len(g), size=max_rows_per_market, replace=False)
            parts.append(g.iloc[idx])
        else:
            parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    print(f"  stratified subsample: {len(df):,} rows across {df['condition_id'].nunique()} markets")

    feats = [c for c in df.columns if c not in V1.NON_FEATURE_COLS]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-999).to_numpy(dtype=np.float32)
    y = df["condition_id"].to_numpy()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    (tr_idx, te_idx) = next(sss.split(X, y))
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf.fit(X[tr_idx], y[tr_idx])
    print(f"  RF fit in {time.time() - t0:.0f}s")

    p = rf.predict_proba(X[te_idx])
    y_te = y[te_idx]
    top1 = accuracy_score(y_te, rf.classes_[p.argmax(axis=1)])
    top5 = top_k_accuracy_score(y_te, p, k=5, labels=rf.classes_)
    n_classes = len(rf.classes_)
    baseline_top1 = 1.0 / n_classes
    baseline_top5 = 5.0 / n_classes

    # Per-market accuracy — how identifiable is each market?
    preds = rf.classes_[p.argmax(axis=1)]
    per_market = []
    for cid in np.unique(y_te):
        mask = y_te == cid
        if not mask.any():
            continue
        acc = float((preds[mask] == cid).mean())
        per_market.append({"condition_id": str(cid), "n": int(mask.sum()), "accuracy": acc})
    per_market.sort(key=lambda r: -r["accuracy"])

    # Top feature importances
    importances = sorted(
        zip(feats, rf.feature_importances_.astype(float)),
        key=lambda x: -x[1],
    )[:10]

    return {
        "n_classes": int(n_classes),
        "n_test": int(len(te_idx)),
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "baseline_top1_random": float(baseline_top1),
        "baseline_top5_random": float(baseline_top5),
        "top1_uplift_x": float(top1 / baseline_top1),
        "top5_uplift_x": float(top5 / baseline_top5),
        "per_market_top_10_most_identifiable": per_market[:10],
        "per_market_top_10_least_identifiable": per_market[-10:],
        "feature_importance_top10": [
            {"feature": f, "importance": float(imp)} for f, imp in importances
        ],
        "runtime_sec": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# B. Random 20 % within-strike split — MLP + LogReg
# ---------------------------------------------------------------------------
def within_strike_split() -> dict:
    print("\n[B] random 20 % within-strike split — MLP + LogReg")
    t0 = time.time()
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"  loaded train.parquet: {len(df):,} rows")

    feats = [c for c in df.columns if c not in V1.NON_FEATURE_COLS]
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = df[V1.TARGET].to_numpy().astype(np.int64)

    # 80/20 stratified split on bet_correct (within-family random holdout)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # Winsorise on train percentiles
    # Identify feature indices that need winsorisation
    wcols = [c for c in V1.WINSORISE_COLS if c in feats]
    for c in wcols:
        j = feats.index(c)
        lo, hi = np.quantile(Xtr[:, j], [0.01, 0.99])
        Xtr[:, j] = np.clip(Xtr[:, j], lo, hi)
        Xte[:, j] = np.clip(Xte[:, j], lo, hi)

    # Median impute + standardise (fit on train)
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(Xtr)).astype(np.float32)
    Xte = sc.transform(imp.transform(Xte)).astype(np.float32)

    # LogReg
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000, solver="lbfgs",
        class_weight="balanced", random_state=RANDOM_SEED,
    )
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]

    # MLP (same spec as V1) — use a small internal val for early stopping
    from sklearn.model_selection import train_test_split as tts
    Xtr_in, Xva, ytr_in, yva = tts(Xtr, ytr, test_size=0.1, stratify=ytr, random_state=RANDOM_SEED)
    mlp, hist = V1.fit_mlp(Xtr_in, ytr_in, Xva, yva)
    p_mlp = mlp.predict(Xte, batch_size=V1.MLP_BATCH, verbose=0).ravel()

    def m(y, p):
        return {
            "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
            "pr_auc": float(average_precision_score(y, p)) if len(set(y)) > 1 else None,
            "brier": float(brier_score_loss(y, p)),
        }

    result = {
        "n_train": int(len(Xtr)),
        "n_test_random_holdout": int(len(Xte)),
        "logreg": m(yte, p_lr),
        "mlp": m(yte, p_mlp),
        "cross_family_reference_test_roc": {
            "logreg": 0.5735,
            "mlp": 0.5787,
            "stack_no_rf_cal": 0.5788,
            "note": "From pontus/outputs/v2/modelling — primary market-cohort test on Apr ceasefires.",
        },
        "runtime_sec": float(time.time() - t0),
    }
    print(f"  logreg within-strike ROC: {result['logreg']['roc_auc']:.4f}  "
          f"(cross-family: 0.5735)")
    print(f"  mlp    within-strike ROC: {result['mlp']['roc_auc']:.4f}  "
          f"(cross-family: 0.5787)")
    return result


# ---------------------------------------------------------------------------
# C. Wallet-novel vs wallet-seen test subset (no retraining)
# ---------------------------------------------------------------------------
def wallet_novelty() -> dict:
    print("\n[C] wallet-novel vs wallet-seen test subset")

    # Use stack_no_rf_cal predictions as the "winner on test" from the
    # v2 run (0.5788 test ROC, beats stack_all_cal by +0.04).
    pred_file = PRED_DIR / "stack_no_rf_cal" / "predictions_test.parquet"
    if not pred_file.exists():
        # Fallback to stack_chosen
        pred_file = PRED_DIR / "stack_chosen" / "predictions_test.parquet"
    preds = pd.read_parquet(pred_file)
    print(f"  loaded {pred_file.name}: {len(preds):,} rows")

    train_wallets = set(pd.read_parquet(TRAIN_PARQUET, columns=["proxyWallet"])["proxyWallet"].unique())
    preds["wallet_in_train"] = preds["proxyWallet"].isin(train_wallets).astype(int)

    def m(df):
        if len(df) == 0:
            return {"n": 0, "roc_auc": None, "pr_auc": None, "brier": None}
        y = df["bet_correct"].to_numpy().astype(int)
        p = df["p_hat"].to_numpy()
        if len(set(y)) < 2:
            return {"n": int(len(df)), "roc_auc": None, "brier": float(brier_score_loss(y, p))}
        return {
            "n": int(len(df)),
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
        }

    overall = m(preds)
    seen = m(preds[preds["wallet_in_train"] == 1])
    novel = m(preds[preds["wallet_in_train"] == 0])

    result = {
        "predictions_source": pred_file.relative_to(ROOT).as_posix(),
        "overall_test": overall,
        "wallet_seen_in_train": seen,
        "wallet_novel": novel,
        "seen_share": float(preds["wallet_in_train"].mean()),
        "delta_roc_seen_minus_novel": (
            None if seen["roc_auc"] is None or novel["roc_auc"] is None
            else float(seen["roc_auc"] - novel["roc_auc"])
        ),
    }
    print(f"  overall ROC: {overall['roc_auc']}  (n={overall['n']:,})")
    print(f"  wallet_seen ROC: {seen['roc_auc']}  (n={seen['n']:,}, share={result['seen_share']:.2f})")
    print(f"  wallet_novel ROC: {novel['roc_auc']}  (n={novel['n']:,})")
    if result["delta_roc_seen_minus_novel"] is not None:
        print(f"  Δ seen-minus-novel: {result['delta_roc_seen_minus_novel']:+.4f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    overall_t = time.time()
    r_audit = market_identity_audit()
    save_json(r_audit, OUT_DIR / "market_identity_audit.json")

    r_within = within_strike_split()
    save_json(r_within, OUT_DIR / "within_strike_split.json")

    r_novelty = wallet_novelty()
    save_json(r_novelty, OUT_DIR / "wallet_novelty.json")

    # Summary at the top level of robustness/
    summary = {
        "runtime_min_total": round((time.time() - overall_t) / 60, 2),
        "market_identity": {
            "top1_accuracy": r_audit["top1_accuracy"],
            "top1_uplift_x_vs_random": r_audit["top1_uplift_x"],
            "n_classes": r_audit["n_classes"],
            "baseline_top1_random": r_audit["baseline_top1_random"],
        },
        "within_strike_random20": {
            "logreg_roc": r_within["logreg"]["roc_auc"],
            "mlp_roc": r_within["mlp"]["roc_auc"],
            "cross_family_mlp_roc": r_within["cross_family_reference_test_roc"]["mlp"],
        },
        "wallet_novelty": {
            "overall": r_novelty["overall_test"]["roc_auc"],
            "seen": r_novelty["wallet_seen_in_train"]["roc_auc"],
            "novel": r_novelty["wallet_novel"]["roc_auc"],
            "delta": r_novelty["delta_roc_seen_minus_novel"],
            "seen_share": r_novelty["seen_share"],
        },
    }
    save_json(summary, OUT_DIR / "summary.json")
    print(f"\n[done] {summary['runtime_min_total']:.1f} min total")
    print(f"  market-identity top1 uplift: {summary['market_identity']['top1_uplift_x_vs_random']:.1f}× random")
    print(f"  within-strike MLP ROC: {summary['within_strike_random20']['mlp_roc']:.4f}  "
          f"(cross-family reference: {summary['within_strike_random20']['cross_family_mlp_roc']:.4f})")
    if summary["wallet_novelty"]["delta"] is not None:
        print(f"  wallet-seen minus wallet-novel ROC: "
              f"{summary['wallet_novelty']['delta']:+.4f}")


if __name__ == "__main__":
    main()
