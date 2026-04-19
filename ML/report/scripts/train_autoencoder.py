"""Unsupervised autoencoder arm per design-decisions §8 (L11 coverage).

An undercomplete stacked autoencoder trained on all trade feature vectors
(unsupervised — does NOT use `bet_correct`). Per-trade reconstruction error
becomes an anomaly score. We cross-check whether trades with the largest
|p_hat − market_implied_prob| gap from the MLP also carry high reconstruction
error — that overlap is our unsupervised sanity check on the supervised signal.

Architecture: 35 → 20 → 10 → 5 (bottleneck) → 10 → 20 → 35, SELU + Glorot,
MSE loss, Adam + LR schedule, early stop on val recon loss.

Outputs:
  data/ae_outputs/
    ae_model.pt
    recon_errors.parquet        per-trade reconstruction error for all splits
    overlap_analysis.json       top-decile overlap vs MLP gap signal, vs null
    loss_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


class StackedAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int] = [20, 10, 5]):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = in_dim
        for h in hidden_dims:
            lin = nn.Linear(prev, h)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            enc_layers.extend([lin, nn.SELU()])
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers = []
        for h in reversed(hidden_dims[:-1]):
            lin = nn.Linear(prev, h)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            dec_layers.extend([lin, nn.SELU()])
            prev = h
        out = nn.Linear(prev, in_dim)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        dec_layers.append(out)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def load_and_prep(labeled_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(labeled_path)
    print(f"loaded {len(df):,} rows × {len(df.columns)} cols")
    if "size_vs_market_cumvol_pct" in df.columns:
        hi = df["size_vs_market_cumvol_pct"].quantile(0.99)
        df["size_vs_market_cumvol_pct"] = df["size_vs_market_cumvol_pct"].clip(
            lower=0, upper=hi
        )
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric if c not in DROP_COLS]
    print(f"using {len(feats)} features")
    return df, feats


def to_tensors(df: pd.DataFrame, feats: list[str], scaler: StandardScaler | None):
    X = (
        df[feats]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values.astype(np.float32)
    )
    if scaler is None:
        scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    return torch.from_numpy(Xs), scaler


def train_ae(
    Xtr: torch.Tensor,
    Xva: torch.Tensor,
    in_dim: int,
    epochs: int = 40,
    batch_size: int = 1024,
    lr: float = 1e-3,
    patience: int = 5,
) -> tuple[StackedAE, list[float], list[float]]:
    model = StackedAE(in_dim=in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    bad = 0
    ntr = Xtr.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(ntr)
        total, count = 0.0, 0
        for i in range(0, ntr, batch_size):
            idx = perm[i : i + batch_size]
            xb = Xtr[idx]
            recon = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        train_loss = total / count

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), Xva).item()
        sched.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={opt.param_groups[0]['lr']:.1e}"
            )

        if bad >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


def recon_error(model: StackedAE, X: torch.Tensor, batch: int = 4096) -> np.ndarray:
    """Per-row MSE between input and reconstruction."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = X[i : i + batch]
            recon = model(xb)
            err = ((recon - xb) ** 2).mean(dim=-1)
            out.append(err.numpy())
    return np.concatenate(out)


def compute_overlap(df: pd.DataFrame) -> dict:
    """Overlap analysis: top-decile reconstruction error ∩ top-decile |gap| from MLP.

    Null baseline: 10% overlap if independent. Ratio > 1.0 = signals correlated.
    """
    preds_path = ROOT / "data" / "mlp_outputs" / "predictions.parquet"
    if not preds_path.exists():
        return {
            "note": "predictions.parquet not found — run train_mlp.py first",
            "status": "skipped",
        }
    preds = pd.read_parquet(preds_path)
    preds["abs_gap"] = (preds["p_hat_cal"] - preds["market_implied_prob"]).abs()
    merged = df.merge(
        preds[["wallet", "condition_id", "timestamp", "abs_gap", "split"]],
        on=["wallet", "condition_id", "timestamp"],
        how="inner",
    )

    out = {}
    for split in ["val", "test"]:
        sub = merged[merged["split"] == split]
        if len(sub) == 0:
            continue
        top_re = sub["recon_error"] >= sub["recon_error"].quantile(0.90)
        top_gap = sub["abs_gap"] >= sub["abs_gap"].quantile(0.90)
        overlap = (top_re & top_gap).sum()
        expected = len(sub) * 0.10 * 0.10
        out[split] = {
            "n_trades": int(len(sub)),
            "top10_recon_error_count": int(top_re.sum()),
            "top10_abs_gap_count": int(top_gap.sum()),
            "intersection": int(overlap),
            "expected_if_independent": round(float(expected), 1),
            "ratio_vs_null": round(float(overlap / max(expected, 1)), 3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labeled", default=str(ROOT / "data" / "iran_strike_labeled.parquet")
    )
    ap.add_argument("--out", default=str(ROOT / "data" / "ae_outputs"))
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    df, feats = load_and_prep(Path(args.labeled))
    tr = df[df["bucket"] == "train"].reset_index(drop=True)
    va = df[df["bucket"] == "val"].reset_index(drop=True)
    te = df[df["bucket"] == "test"].reset_index(drop=True)

    Xtr, scaler = to_tensors(tr, feats, None)
    Xva, _ = to_tensors(va, feats, scaler)
    Xte, _ = to_tensors(te, feats, scaler)

    print(f"train={Xtr.shape}  val={Xva.shape}  test={Xte.shape}")
    print("\ntraining autoencoder...")
    model, tl, vl = train_ae(Xtr, Xva, in_dim=Xtr.shape[1], epochs=args.epochs)

    # Per-trade recon error
    print("\ncomputing reconstruction errors...")
    re_tr = recon_error(model, Xtr)
    re_va = recon_error(model, Xva)
    re_te = recon_error(model, Xte)

    recon_df = pd.concat(
        [
            tr.assign(recon_error=re_tr)[
                [
                    "wallet",
                    "condition_id",
                    "timestamp",
                    "bucket",
                    "bet_correct",
                    "recon_error",
                ]
            ],
            va.assign(recon_error=re_va)[
                [
                    "wallet",
                    "condition_id",
                    "timestamp",
                    "bucket",
                    "bet_correct",
                    "recon_error",
                ]
            ],
            te.assign(recon_error=re_te)[
                [
                    "wallet",
                    "condition_id",
                    "timestamp",
                    "bucket",
                    "bet_correct",
                    "recon_error",
                ]
            ],
        ]
    )
    recon_df.to_parquet(out_dir / "recon_errors.parquet", index=False)
    print(
        f"  saved recon_errors.parquet — "
        f"train mean={re_tr.mean():.3f} / val mean={re_va.mean():.3f} / test mean={re_te.mean():.3f}"
    )

    # Overlap analysis
    print("\noverlap analysis: top-decile recon error ∩ top-decile |gap|...")
    overlap = compute_overlap(recon_df)
    (out_dir / "overlap_analysis.json").write_text(json.dumps(overlap, indent=2))
    for k, v in overlap.items():
        print(f"  {k}: {v}")

    torch.save(model.state_dict(), out_dir / "ae_model.pt")

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tl, label="train", color="#4C72B0")
    ax.plot(vl, label="val", color="#DD8452")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE recon loss")
    ax.set_title("Autoencoder training loss")
    ax.legend()
    fig.savefig(out_dir / "loss_curve.png", bbox_inches="tight", dpi=140)
    plt.close(fig)

    print(f"\nall outputs in {out_dir}/")


if __name__ == "__main__":
    main()
