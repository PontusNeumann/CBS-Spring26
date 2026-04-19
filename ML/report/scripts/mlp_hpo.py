"""Small grid search over MLP architecture (L13 — hyperparameter optimization).

Searches over: hidden dim sizes × dropout × learning rate. Reports val ROC-AUC
and test ROC-AUC for each config. Small grid to keep runtime <15 min.

Outputs:
  data/mlp_outputs/hpo_results.csv
"""

from __future__ import annotations

import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

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
    "market_implied_prob",
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
    "wallet_prior_trades",
}


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            lin = nn.Linear(prev, h)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.extend([lin, nn.BatchNorm1d(h), nn.SELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_data():
    df = pd.read_parquet(ROOT / "data" / "iran_strike_labeled.parquet")
    if "size_vs_market_cumvol_pct" in df.columns:
        hi = df["size_vs_market_cumvol_pct"].quantile(0.99)
        df["size_vs_market_cumvol_pct"] = df["size_vs_market_cumvol_pct"].clip(
            lower=0, upper=hi
        )
    feats = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in DROP_COLS
    ]
    tr = df[df["bucket"] == "train"].reset_index(drop=True)
    va = df[df["bucket"] == "val"].reset_index(drop=True)
    te = df[df["bucket"] == "test"].reset_index(drop=True)
    return tr, va, te, feats


def to_tensor(df, feats, scaler):
    X = (
        df[feats]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values.astype(np.float32)
    )
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return (
        torch.from_numpy(scaler.transform(X).astype(np.float32)),
        torch.from_numpy(df["bet_correct"].astype(np.float32).values),
        scaler,
    )


def train_once(
    Xtr, ytr, Xva, yva, hidden, dropout, lr, epochs=30, batch=512, patience=5
):
    model = MLP(Xtr.shape[1], hidden, dropout)
    pos_weight = torch.tensor([float((1 - ytr).sum() / ytr.sum())], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2
    )

    best_val = float("inf")
    best_state = None
    bad = 0
    ntr = Xtr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(ntr)
        for i in range(0, ntr, batch):
            idx = perm[i : i + batch]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva), yva).item()
        sched.step(vl)
        if vl < best_val - 1e-4:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict(model, X, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            out.append(torch.sigmoid(model(X[i : i + batch])).numpy())
    return np.concatenate(out)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    tr, va, te, feats = load_data()
    Xtr, ytr, scaler = to_tensor(tr, feats, None)
    Xva, yva, _ = to_tensor(va, feats, scaler)
    Xte, yte, _ = to_tensor(te, feats, scaler)
    print(f"train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

    # Compact grid — 6 configs
    grid = list(
        itertools.product(
            [[128, 64], [128, 64, 32], [256, 128, 64]],  # depth/width
            [0.2, 0.4],  # dropout
        )
    )
    print(f"\ntraining {len(grid)} configs...")

    results = []
    t0 = time.time()
    for i, (hidden, dropout) in enumerate(grid):
        cfg_name = f"{'-'.join(map(str, hidden))}_d{dropout}"
        t = time.time()
        model = train_once(Xtr, ytr, Xva, yva, hidden, dropout, lr=1e-3)
        pv = predict(model, Xva)
        pt = predict(model, Xte)
        val_roc = roc_auc_score(yva.numpy(), pv)
        test_roc = roc_auc_score(yte.numpy(), pt)
        val_pr = average_precision_score(yva.numpy(), pv)
        test_pr = average_precision_score(yte.numpy(), pt)
        dt = time.time() - t
        results.append(
            {
                "config": cfg_name,
                "hidden": hidden,
                "dropout": dropout,
                "val_roc_auc": round(val_roc, 4),
                "val_pr_auc": round(val_pr, 4),
                "test_roc_auc": round(test_roc, 4),
                "test_pr_auc": round(test_pr, 4),
                "train_seconds": round(dt, 1),
            }
        )
        print(
            f"  [{i + 1}/{len(grid)}] {cfg_name}: val_roc={val_roc:.4f} test_roc={test_roc:.4f} ({dt:.0f}s)"
        )

    out = ROOT / "data" / "mlp_outputs" / "hpo_results.csv"
    pd.DataFrame(results).sort_values("val_roc_auc", ascending=False).to_csv(
        out, index=False
    )
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")
    print(f"saved {out}")
    print("\nsorted by val_roc:")
    print(
        pd.DataFrame(results)
        .sort_values("val_roc_auc", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
