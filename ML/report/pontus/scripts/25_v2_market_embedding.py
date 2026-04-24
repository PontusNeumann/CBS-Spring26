"""
25_v2_market_embedding.py — Option 2 for isolating market identity.

Architecture
------------
Two-input tf.keras Functional model. The market-id input goes through an
Embedding(n_train_markets, emb_dim=4), whose vector is gated by an
explicit 0/1 mask input. Training sees all 4 strike markets (mask=1) so
the embedding learns a per-market intercept. Validation (Mar 15
conflict-end) and test (April ceasefires) markets are novel — there is
no learned embedding for them — so the mask is set to 0 at inference,
producing the **pure feature-driven prediction**.

                    features (36-dim, standardised)
                       │
                       │          market_idx (int 0..3) ── Embedding(4, 4) ─┐
                       │                                                    │
                       │                                                    ▼
                       │                                          mask × emb (∈ R^4)
                       ▼                                                    │
                     concat ◄─────────────────────────────────────────────┘
                       │
                       ▼ Dense(256, SELU) + BN + Dropout(0.3)
                       ▼ Dense(128, SELU) + BN + Dropout(0.3)
                       ▼ Dense(  64, SELU) + BN + Dropout(0.3)
                       ▼ Dense(1, sigmoid) → p_hat

Why this is a test of the market-identity channel
-------------------------------------------------
At training time the embedding can absorb "this market's log-odds
intercept," freeing the rest of the network to learn cross-market
patterns in the features. At inference time the mask is 0, so the
prediction comes entirely from the trunk applied to the feature vector.

Comparison to the v2 MLP baseline (test ROC 0.5787 on stack_no_rf_cal ≈
MLP alone) tells us:

  * If embedding-zero test ROC **>= 0.58**: the V2 MLP was already
    carrying only feature-driven signal — the embedding merely freed
    capacity, and the honest (market-identity-stripped) signal is as
    strong as the raw headline.
  * If **< 0.58**: the V2 MLP was using per-market base-rate
    information in the 36 feature columns. The embedding absorbed it,
    so the trunk learnt a signal that lives only in the market-
    intercept space. At inference the mask drops the intercept and
    the remaining trunk prediction is weaker — by the difference.

Outputs → `pontus/outputs/v2_market_embedding/`:
  modelling/mlp_emb/ metrics.json, predictions_{train,val,test}.parquet,
                     loss_curve.png, market_embeddings.json
  residual_edge/metrics.json
  summary.json

Framework: tf.keras (CBS MLDP rubric).

Usage:
  caffeinate -i python pontus/scripts/25_v2_market_embedding.py \\
    2>&1 | tee pontus/outputs/v2_market_embedding/run.log
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Import V1 helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / "pontus" / "scripts" / "21_full_pipeline.py"
_spec = importlib.util.spec_from_file_location("pipeline_v1", V1_PATH)
V1 = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_v1"] = V1
_spec.loader.exec_module(V1)

OUT_DIR = ROOT / "pontus" / "outputs" / "v2_market_embedding"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
keras.utils.set_random_seed(RANDOM_SEED)

# MLP hyper-params — kept identical to V1 so the only difference is the
# embedding input.
MLP_HIDDEN = V1.MLP_HIDDEN         # [256, 128, 64]
MLP_DROPOUT = V1.MLP_DROPOUT       # 0.3
MLP_LR = V1.MLP_LR                 # 1e-3
MLP_BATCH = V1.MLP_BATCH           # 4096
MLP_EPOCHS = V1.MLP_EPOCHS         # 40
MLP_PATIENCE = V1.MLP_PATIENCE     # 8
MLP_LR_PATIENCE = V1.MLP_LR_PATIENCE

EMB_DIM = 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_mlp_with_embedding(n_features: int, n_markets: int) -> keras.Model:
    feat_in = keras.Input(shape=(n_features,), dtype="float32", name="features")
    mkt_in = keras.Input(shape=(1,), dtype="int32", name="market_idx")
    mask_in = keras.Input(shape=(1,), dtype="float32", name="market_mask")

    emb = keras.layers.Embedding(
        input_dim=n_markets,
        output_dim=EMB_DIM,
        name="market_embedding",
        embeddings_initializer=keras.initializers.RandomNormal(stddev=0.05, seed=RANDOM_SEED),
    )(mkt_in)
    emb = keras.layers.Flatten()(emb)          # (batch, EMB_DIM)
    emb = keras.layers.Multiply(name="masked_emb")([emb, mask_in])

    x = keras.layers.Concatenate()([feat_in, emb])
    for units in MLP_HIDDEN:
        x = keras.layers.Dense(
            units, activation="selu",
            kernel_initializer="lecun_normal",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(MLP_DROPOUT)(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="p_hat")(x)

    m = keras.Model(inputs=[feat_in, mkt_in, mask_in], outputs=out,
                    name="mlp_with_market_embedding")
    m.compile(
        optimizer=keras.optimizers.Adam(MLP_LR),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return m


# ---------------------------------------------------------------------------
# Data preparation — map train condition_id → int index, mask val/test to 0
# ---------------------------------------------------------------------------
def build_market_index(train_df: pd.DataFrame) -> dict[str, int]:
    unique = sorted(train_df["condition_id"].unique().tolist())
    return {c: i for i, c in enumerate(unique)}


def map_market_idx(df: pd.DataFrame, index: dict[str, int]) -> np.ndarray:
    return df["condition_id"].map(index).fillna(0).astype(np.int32).to_numpy()


# ---------------------------------------------------------------------------
# Training + eval
# ---------------------------------------------------------------------------
def train_mlp_emb(
    X_tr: np.ndarray, mkt_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, mkt_va: np.ndarray, y_va: np.ndarray,
    n_markets: int,
) -> tuple[keras.Model, dict]:
    print(f"[mlp-emb] building model: n_features={X_tr.shape[1]} n_markets={n_markets} "
          f"emb_dim={EMB_DIM}")
    model = build_mlp_with_embedding(X_tr.shape[1], n_markets)
    # Train with mask=1 (embedding active); val uses mask=0 (novel market —
    # no learned embedding for Mar 15 conflict-end).
    mask_tr = np.ones((len(X_tr), 1), dtype=np.float32)
    mask_va = np.zeros((len(X_va), 1), dtype=np.float32)

    from sklearn.utils.class_weight import compute_class_weight
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weight = {0: float(w[0]), 1: float(w[1])}

    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=MLP_PATIENCE, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=MLP_LR_PATIENCE, min_lr=1e-5,
        ),
    ]

    t0 = time.time()
    hist = model.fit(
        x={"features": X_tr, "market_idx": mkt_tr.reshape(-1, 1), "market_mask": mask_tr},
        y=y_tr,
        validation_data=(
            {"features": X_va, "market_idx": mkt_va.reshape(-1, 1), "market_mask": mask_va},
            y_va,
        ),
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH,
        class_weight=class_weight,
        callbacks=cb,
        verbose=2,
    )
    print(f"[mlp-emb] trained in {time.time() - t0:.0f}s ({len(hist.history['loss'])} epochs)")
    return model, hist.history


def predict_with_mask(model, X, market_idx, mask_val):
    mask = np.full((len(X), 1), mask_val, dtype=np.float32)
    return model.predict(
        {"features": X, "market_idx": market_idx.reshape(-1, 1), "market_mask": mask},
        batch_size=MLP_BATCH, verbose=0,
    ).ravel()


def probability_metrics(y, p):
    if len(set(y)) < 2:
        return {
            "n": int(len(y)), "positive_rate": float(np.mean(y)),
            "roc_auc": None, "pr_auc": None,
            "brier": float(brier_score_loss(y, p)),
            "ece_15bin": V1.expected_calibration_error(y, p),
        }
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_15bin": V1.expected_calibration_error(y, p),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()

    # --- Load + preprocess (reuse V1 helpers) ------------------------------
    cohorts = V1.load_cohorts()
    V1.winsorise_train_fit(cohorts, V1.WINSORISE_COLS)
    X_tr, X_va, X_te = V1.impute_and_scale(cohorts)
    y_tr = cohorts.train[V1.TARGET].to_numpy().astype(np.int64)
    y_va = cohorts.val[V1.TARGET].to_numpy().astype(np.int64)
    y_te = cohorts.test[V1.TARGET].to_numpy().astype(np.int64)
    mip_tr = cohorts.train[V1.BENCHMARK].to_numpy().astype(np.float64)
    mip_va = cohorts.val[V1.BENCHMARK].to_numpy().astype(np.float64)
    mip_te = cohorts.test[V1.BENCHMARK].to_numpy().astype(np.float64)

    # --- Build the train-market index ------------------------------------
    market_index = build_market_index(cohorts.train)
    n_markets = len(market_index)
    print(f"[index] n_train_markets={n_markets}")
    for cid, idx in market_index.items():
        q = cohorts.train.loc[cohorts.train["condition_id"] == cid, "question"].iloc[0]
        print(f"  [{idx}] {q[:70]}  ({cid[:12]})")

    mkt_tr = map_market_idx(cohorts.train, market_index)
    mkt_va = map_market_idx(cohorts.val, market_index)
    mkt_te = map_market_idx(cohorts.test, market_index)
    # Diagnostic: no val/test market should be in the train index
    va_overlap = sum(1 for c in cohorts.val["condition_id"].unique() if c in market_index)
    te_overlap = sum(1 for c in cohorts.test["condition_id"].unique() if c in market_index)
    print(f"  val-market overlap with train index: {va_overlap}")
    print(f"  test-market overlap with train index: {te_overlap}")

    # --- Train ------------------------------------------------------------
    model, hist = train_mlp_emb(
        X_tr, mkt_tr, y_tr, X_va, mkt_va, y_va, n_markets,
    )

    # --- Save learned embeddings ----------------------------------------
    emb_layer = model.get_layer("market_embedding")
    emb_weights = emb_layer.get_weights()[0]  # (n_markets, EMB_DIM)
    emb_record = {}
    for cid, idx in market_index.items():
        q = cohorts.train.loc[cohorts.train["condition_id"] == cid, "question"].iloc[0]
        emb_record[cid] = {
            "index": int(idx),
            "question": q,
            "embedding": emb_weights[idx].astype(float).tolist(),
            "l2_norm": float(np.linalg.norm(emb_weights[idx])),
        }

    mdir = OUT_DIR / "modelling" / "mlp_emb"
    mdir.mkdir(parents=True, exist_ok=True)
    V1.save_json(emb_record, mdir / "market_embeddings.json")
    V1.plot_loss_curve(hist, "MLP + market embedding — BCE loss", mdir / "loss_curve.png")

    # --- Predictions: two inference modes ---------------------------------
    # Mode A: mask=1 with the "mean of train embeddings" for novel markets.
    #         Applied to val + test via a fake market index = 0 combined
    #         with the learnt per-market embeddings averaged manually.
    # Mode B: mask=0 — pure feature-driven prediction, no per-market intercept.
    # Mode B is the primary report metric; Mode A is a sensitivity check.
    mean_emb = emb_weights.mean(axis=0, keepdims=False)  # (EMB_DIM,)

    # --- Mode B: mask=0 (the clean test) ---
    p_tr_maskB = predict_with_mask(model, X_tr, mkt_tr, 0.0)  # for reference
    p_va_maskB = predict_with_mask(model, X_va, mkt_va, 0.0)
    p_te_maskB = predict_with_mask(model, X_te, mkt_te, 0.0)
    # Train-as-inference with mask=1 (how well the model fits train with the
    # learnt per-market embedding — upper-bound sanity).
    p_tr_maskA = predict_with_mask(model, X_tr, mkt_tr, 1.0)

    # --- Mode A on val/test using mean embedding: run model with mask=0
    # and a small Python-side shift. Simpler: manually add the trunk output
    # shift (skip — the trunk is non-linear downstream of the concat). So
    # we approximate Mode A by temporarily monkey-patching the embedding
    # to return the mean vector and predicting with mask=1.
    # Implementation: build a sibling model that swaps the mean emb in.
    # Skip this for simplicity — Mode A on novel markets is ambiguous by
    # design (no learned intercept for those markets).

    # --- Persist predictions -----------------------------------------------
    def _save_preds(fold_name, df_fold, p):
        id_cols = [c for c in ("proxyWallet", "condition_id", "transactionHash",
                               "timestamp", "bet_correct", "market_implied_prob")
                   if c in df_fold.columns]
        out = df_fold[id_cols].copy()
        out["p_hat"] = p
        out.to_parquet(mdir / f"predictions_{fold_name}.parquet", index=False)

    _save_preds("train_maskB", cohorts.train, p_tr_maskB)
    _save_preds("train_maskA", cohorts.train, p_tr_maskA)
    _save_preds("val_maskB",   cohorts.val,   p_va_maskB)
    _save_preds("test_maskB",  cohorts.test,  p_te_maskB)

    # --- Metrics ----------------------------------------------------------
    mB_metrics = {
        "train_maskB": probability_metrics(y_tr, p_tr_maskB),
        "train_maskA_fit_upper_bound": probability_metrics(y_tr, p_tr_maskA),
        "val_maskB":   probability_metrics(y_va, p_va_maskB),
        "test_maskB":  probability_metrics(y_te, p_te_maskB),
    }
    V1.save_json(mB_metrics, mdir / "metrics.json")
    t = mB_metrics["test_maskB"]
    print(f"\n[metrics mlp_emb maskB] test roc={t['roc_auc']}  brier={t['brier']:.4f}  "
          f"ece={t['ece_15bin']:.4f}")
    print(f"[metrics mlp_emb maskB] val  roc={mB_metrics['val_maskB']['roc_auc']}  "
          f"brier={mB_metrics['val_maskB']['brier']:.4f}")
    print(f"[metrics mlp_emb maskA] train roc={mB_metrics['train_maskA_fit_upper_bound']['roc_auc']}  "
          f"(with learnt embeddings — fit upper bound, not a generalisation claim)")

    # --- Residual-edge on mask=B test predictions -----------------------
    r_edge = V1.residual_edge_analysis(
        p_tr_maskB, mip_tr, y_tr,
        p_va_maskB, mip_va, y_va,
        p_te_maskB, mip_te, y_te,
    )
    (OUT_DIR / "residual_edge").mkdir(parents=True, exist_ok=True)
    V1.save_json(r_edge, OUT_DIR / "residual_edge" / "metrics.json")
    print(f"[residual-edge maskB] test partial-corr = "
          f"{r_edge['test']['partial_corr_bc_vs_residual']:+.4f}  "
          f"(v2 baseline: +0.1787, v2-residualised: +0.3099)")

    # --- Summary vs V2 baseline ------------------------------------------
    summary = {
        "runtime_min": round((time.time() - t_start) / 60, 2),
        "feature_count": X_tr.shape[1],
        "n_train_markets": n_markets,
        "emb_dim": EMB_DIM,
        "epochs_trained": len(hist["loss"]),
        "embedding_l2_norms": {
            market_index_inv: float(np.linalg.norm(emb_weights[i]))
            for market_index_inv, i in market_index.items()
        },
        "test_roc_mask0_pure_features": mB_metrics["test_maskB"]["roc_auc"],
        "test_brier_mask0": mB_metrics["test_maskB"]["brier"],
        "test_ece_mask0": mB_metrics["test_maskB"]["ece_15bin"],
        "residual_edge_partial_corr_test": r_edge["test"]["partial_corr_bc_vs_residual"],
        "comparison_v2_baseline": {
            "v2_mlp_test_roc": 0.5787,
            "v2_stack_no_rf_cal_test_roc": 0.5788,
            "v2_residualised_mlp_test_roc": 0.5432,
            "v2_residual_edge_partial_corr": 0.1787,
            "v2_residualised_residual_edge_partial_corr": 0.3099,
        },
        "interpretation_hint": (
            "If test_roc_mask0_pure_features >= 0.58, the v2 MLP was "
            "feature-driven — the embedding absorbed market identity "
            "cleanly and the honest signal equals the raw headline. If "
            "materially lower, the v2 baseline relied on per-market base-"
            "rate information and the honest cross-family signal is the "
            "mask=0 number, not the 0.58 headline."
        ),
    }
    V1.save_json(summary, OUT_DIR / "summary.json")
    print(f"\n[done] {summary['runtime_min']:.1f} min total")
    print(f"  test ROC mask=0 (pure features): {summary['test_roc_mask0_pure_features']:.4f}")
    print(f"  v2 baseline                    : 0.5788")
    print(f"  residualised v2                : 0.5432")


if __name__ == "__main__":
    main()
