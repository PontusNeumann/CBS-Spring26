"""MLP training on the Iran-strike labeled dataset (v1, without Layer 6).

Architecture per design-decisions.md §8:
  - 2-4 hidden layers, SELU + Glorot init, BatchNorm, Dropout 0.2-0.4
  - BCE loss, Adam + LR schedule, class_weight='balanced' via sample weights
  - Early stopping on val loss
  - Isotonic calibration on val

Same feature set as baselines.py (keeps `price`, drops `market_implied_prob`).

Outputs:
  data/mlp_outputs/
    model.pt                 trained weights
    scaler.pkl               StandardScaler state
    calibrator.pkl           isotonic regression state
    feature_list.json        ordered feature columns
    metrics.json             val/test ROC-AUC, PR-AUC, Brier, accuracy, gap_auc
    predictions.parquet      per-trade p_hat + p_hat_calibrated for val + test
    loss_curve.png           train/val loss per epoch

Usage:
  python scripts/train_mlp.py [--labeled data/iran_strike_labeled.parquet]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

# Feature column exclusions — mirrors baselines.py
DROP_COLS = {
    "timestamp",
    "block_number",
    "transaction_hash",
    "condition_id",
    "maker",
    "taker",
    "nonusdc_side",
    "resolved",
    "winner_token",
    "settlement_ts",
    "bucket",
    "wallet",
    "side",
    "question",
    "bet_correct",
    "market_implied_prob",  # kept out — it's the benchmark, not an input
    "usd_amount",
    "token_amount",
    "wallet_t1_position_before",
    "wallet_t2_position_before",
    "wallet_t1_cumvol_in_market",
    "wallet_t2_cumvol_in_market",
    "wallet_position_same_token_before",
    "wallet_total_cumvol_in_market",
    "market_cumvol",
    "market_vol_1h",
    "market_vol_24h",
    "market_trades_1h",
    "wallet_prior_trades",  # keep _log variant
}

# Clipping from EDA
CLIP_COLS = {"size_vs_market_cumvol_pct": (0.0, None)}  # clip to 99th pct


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden: list[int] = [128, 64, 32], dropout: float = 0.3
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            lin = nn.Linear(prev, h)
            nn.init.xavier_uniform_(lin.weight)  # Glorot init
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.SELU())  # SELU activation
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_and_prep(labeled_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(labeled_path)
    print(f"loaded {len(df):,} rows × {len(df.columns)} cols")
    for col, (lo, hi) in CLIP_COLS.items():
        if col in df.columns:
            hi = hi if hi is not None else df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lo, upper=hi)

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric if c not in DROP_COLS]
    print(f"using {len(feats)} features")
    return df, feats


def to_tensors(
    df: pd.DataFrame, feats: list[str], scaler: StandardScaler | None
) -> tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    X = (
        df[feats]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values.astype(np.float32)
    )
    y = df["bet_correct"].astype(np.float32).values
    if scaler is None:
        scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X).astype(np.float32)
    return torch.from_numpy(X_scaled), torch.from_numpy(y), scaler


def train_mlp(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 8,
) -> tuple[MLP, list[float], list[float]]:
    model = MLP(in_dim=Xtr.shape[1])
    # class_weight='balanced' via sample weights → pos_weight in BCEWithLogits
    pos = float(ytr.sum().item())
    neg = float((1 - ytr).sum().item())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_state = None
    bad = 0
    ntr = Xtr.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(ntr)
        total = 0.0
        count = 0
        for i in range(0, ntr, batch_size):
            idx = perm[i : i + batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        train_loss = total / count

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = loss_fn(val_logits, yva).item()
        sched.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={opt.param_groups[0]['lr']:.1e}"
            )

        if bad >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


def predict_proba(model: MLP, X: torch.Tensor, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            logits = model(X[i : i + batch])
            probs = torch.sigmoid(logits)
            out.append(probs.numpy())
    return np.concatenate(out)


def metrics_block(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "split": name,
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "log_loss": round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 4),
        "accuracy": round(float(((p >= 0.5).astype(int) == y).mean()), 4),
    }


def gap_metrics(
    name: str, y: np.ndarray, p_hat: np.ndarray, p_market: np.ndarray
) -> dict:
    gap = p_hat - p_market
    return {
        "split": name,
        "mean_gap": round(float(gap.mean()), 4),
        "mean_abs_gap": round(float(np.abs(gap).mean()), 4),
        "gap_roc_auc": round(float(roc_auc_score(y, gap)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labeled", default=str(ROOT / "data" / "iran_strike_labeled.parquet")
    )
    ap.add_argument("--out", default=str(ROOT / "data" / "mlp_outputs"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    df, feats = load_and_prep(Path(args.labeled))
    tr = df[df["bucket"] == "train"].reset_index(drop=True)
    va = df[df["bucket"] == "val"].reset_index(drop=True)
    te = df[df["bucket"] == "test"].reset_index(drop=True)

    Xtr, ytr, scaler = to_tensors(tr, feats, None)
    Xva, yva, _ = to_tensors(va, feats, scaler)
    Xte, yte, _ = to_tensors(te, feats, scaler)
    print(f"train={Xtr.shape}  val={Xva.shape}  test={Xte.shape}")

    print("\ntraining MLP...")
    model, tlosses, vlosses = train_mlp(
        Xtr, ytr, Xva, yva, epochs=args.epochs, batch_size=args.batch, lr=args.lr
    )

    # Uncalibrated predictions
    p_val = predict_proba(model, Xva)
    p_te = predict_proba(model, Xte)

    # Isotonic calibration: fit on val, apply on test
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, va["bet_correct"].values)
    p_val_cal = iso.transform(p_val)
    p_te_cal = iso.transform(p_te)

    # Metrics
    all_metrics: list[dict] = []
    print("\nraw predictions:")
    for name, y, p in [
        ("val", va["bet_correct"].values, p_val),
        ("test", te["bet_correct"].values, p_te),
    ]:
        m = metrics_block(name, y, p)
        m["stage"] = "raw"
        all_metrics.append(m)
        print(f"  {name}: {m}")

    print("\ncalibrated predictions (isotonic on val):")
    for name, y, p in [
        ("val", va["bet_correct"].values, p_val_cal),
        ("test", te["bet_correct"].values, p_te_cal),
    ]:
        m = metrics_block(name, y, p)
        m["stage"] = "calibrated"
        all_metrics.append(m)
        print(f"  {name}: {m}")

    print("\ngap evaluation (p_hat_calibrated − market_implied_prob):")
    gap_all = []
    for name, subdf, p in [("val", va, p_val_cal), ("test", te, p_te_cal)]:
        g = gap_metrics(
            name, subdf["bet_correct"].values, p, subdf["market_implied_prob"].values
        )
        gap_all.append(g)
        print(f"  {name}: {g}")

    # Save artifacts
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    (out_dir / "feature_list.json").write_text(json.dumps(feats, indent=2))
    (out_dir / "metrics.json").write_text(
        json.dumps({"eval": all_metrics, "gap": gap_all}, indent=2)
    )

    preds = pd.concat(
        [
            va.assign(p_hat=p_val, p_hat_cal=p_val_cal, split="val")[
                [
                    "wallet",
                    "condition_id",
                    "timestamp",
                    "bet_correct",
                    "market_implied_prob",
                    "p_hat",
                    "p_hat_cal",
                    "split",
                ]
            ],
            te.assign(p_hat=p_te, p_hat_cal=p_te_cal, split="test")[
                [
                    "wallet",
                    "condition_id",
                    "timestamp",
                    "bet_correct",
                    "market_implied_prob",
                    "p_hat",
                    "p_hat_cal",
                    "split",
                ]
            ],
        ]
    )
    preds.to_parquet(out_dir / "predictions.parquet", index=False)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tlosses, label="train", color="#4C72B0")
    ax.plot(vlosses, label="val", color="#DD8452")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss (pos-weighted)")
    ax.set_title("MLP training loss")
    ax.legend()
    fig.savefig(out_dir / "loss_curve.png", bbox_inches="tight", dpi=140)
    plt.close(fig)

    print(f"\nall outputs in {out_dir}/")


if __name__ == "__main__":
    main()
