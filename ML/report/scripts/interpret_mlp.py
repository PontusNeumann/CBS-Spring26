"""Permutation importance on the trained MLP (L14 — XAI).

Measures Δ ROC-AUC on the validation split when each feature is shuffled.
High Δ = feature is important to the model's predictions. Complements the
L1 / RF feature importances we already have.

Outputs:
  data/mlp_outputs/permutation_importance.csv
  data/mlp_outputs/permutation_importance.png
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MLP_DIR = ROOT / "data" / "mlp_outputs"


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden: list[int] = [128, 64, 32], dropout: float = 0.3
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> None:
    feats = json.loads((MLP_DIR / "feature_list.json").read_text())
    with open(MLP_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = MLP(in_dim=len(feats))
    model.load_state_dict(torch.load(MLP_DIR / "model.pt", map_location="cpu"))
    model.eval()

    # Load val only — permutation importance uses one split
    df = pd.read_parquet(ROOT / "data" / "iran_strike_labeled.parquet")
    va = df[df["bucket"] == "val"].reset_index(drop=True)
    print(f"val shape: {va.shape}")

    # Build X on a subsample for speed (50k rows is enough for stable permutation)
    va = va.sample(n=min(50_000, len(va)), random_state=42).reset_index(drop=True)
    X = (
        va[feats]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values.astype(np.float32)
    )
    y = va["bet_correct"].astype(int).values
    Xs = scaler.transform(X).astype(np.float32)

    def auc_of(Xarr: np.ndarray) -> float:
        with torch.no_grad():
            logits = model(torch.from_numpy(Xarr))
            p = torch.sigmoid(logits).numpy()
        return float(roc_auc_score(y, p))

    base_auc = auc_of(Xs)
    print(f"baseline val ROC-AUC: {base_auc:.4f}")

    # Permutation: shuffle one column at a time, measure AUC drop
    rng = np.random.default_rng(42)
    results = []
    for i, feat in enumerate(feats):
        Xperm = Xs.copy()
        Xperm[:, i] = rng.permutation(Xperm[:, i])
        auc = auc_of(Xperm)
        drop = base_auc - auc
        results.append(
            {
                "feature": feat,
                "auc_after_permute": round(auc, 4),
                "delta_auc": round(drop, 4),
            }
        )
        if (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(feats)}] {feat}: Δ={drop:.4f}")

    imp = pd.DataFrame(results).sort_values("delta_auc", ascending=False)
    imp.to_csv(MLP_DIR / "permutation_importance.csv", index=False)
    print(f"\ntop 15 features by Δ ROC-AUC:")
    print(imp.head(15).to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 10))
    top = imp.head(25).iloc[::-1]
    colors = ["#C44E52" if d < 0 else "#4C72B0" for d in top["delta_auc"]]
    ax.barh(top["feature"], top["delta_auc"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Δ ROC-AUC (baseline − permuted)")
    ax.set_title(
        f"MLP permutation importance (val, n=50k)\nbaseline AUC={base_auc:.3f}"
    )
    fig.tight_layout()
    fig.savefig(MLP_DIR / "permutation_importance.png", dpi=140)
    plt.close(fig)
    print(f"saved {MLP_DIR / 'permutation_importance.png'}")


if __name__ == "__main__":
    main()
