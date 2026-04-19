"""Reframed MLP: predict P(market resolves YES) instead of P(bet_correct).

Why: the previous formulation predicted `bet_correct` directly, entangling
the trader's direction with the market's outcome. The training bucket
(70% NO-resolved markets) vs test bucket (99% YES-market trades) produced
a distribution shift on the direction × outcome interaction, inverting all
supervised baselines on test (ROC 0.15 RF, 0.23 MLP).

Reframing: predict the market outcome directly from market-state + wallet
features, exclude direction features (is_buy, is_token1, price). Derive
per-trade correctness at inference from the known direction.

  - If the trade is a BUY on token1 or a SELL on token2:
        P(correct) = P(model predicts YES)
  - Else (BUY token2 or SELL token1):
        P(correct) = 1 − P(model predicts YES)

The gap-based signal becomes:
  gap_yes = p_model − market_implied_yes_prob
  where market_implied_yes_prob = price if token1 else 1 − price

Outputs:
  data/mlp_reframed_outputs/
    model.pt, scaler.pkl, calibrator.pkl, feature_list.json
    metrics.json                   both target-direct and derived bet_correct metrics
    predictions.parquet            per-trade p_yes + p_correct (derived)
    loss_curve.png
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

# Direction features + market's instantaneous belief — excluded
# so the model predicts outcome from independent signals.
DIRECTION_AND_PRICE = {
    "is_buy",
    "is_token1",
    "price",
    "market_implied_prob",
    "side",
    "nonusdc_side",
}

# Meta / identifiers / raw versions (we keep the log versions)
META_AND_RAW = {
    "timestamp",
    "block_number",
    "transaction_hash",
    "condition_id",
    "maker",
    "taker",
    "resolved",
    "winner_token",
    "settlement_ts",
    "bucket",
    "wallet",
    "question",
    "bet_correct",
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

DROP_COLS = DIRECTION_AND_PRICE | META_AND_RAW


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden: list[int] = [256, 128, 64], dropout: float = 0.3
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


def load_and_prep(labeled_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(labeled_path)
    print(f"loaded {len(df):,} rows × {len(df.columns)} cols")

    # New target: P(market resolves YES) — per-trade but set by market
    df["market_resolves_yes"] = (df["resolved"] == "YES").astype(np.int8)

    # Market's implied P(YES) at trade time — used for gap comparison, not model input
    df["market_implied_yes_prob"] = np.where(
        df["is_token1"] == 1, df["price"], 1.0 - df["price"]
    )

    # Clipping from EDA
    if "size_vs_market_cumvol_pct" in df.columns:
        hi = df["size_vs_market_cumvol_pct"].quantile(0.99)
        df["size_vs_market_cumvol_pct"] = df["size_vs_market_cumvol_pct"].clip(
            lower=0, upper=hi
        )

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [
        c
        for c in numeric
        if c not in DROP_COLS
        and c not in ("market_resolves_yes", "market_implied_yes_prob")
    ]
    print(f"using {len(feats)} features (direction + price excluded)")
    return df, feats


def to_tensors(df, feats, scaler):
    X = (
        df[feats]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values.astype(np.float32)
    )
    y = df["market_resolves_yes"].astype(np.float32).values
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return (
        torch.from_numpy(scaler.transform(X).astype(np.float32)),
        torch.from_numpy(y),
        scaler,
    )


def train(Xtr, ytr, Xva, yva, epochs=60, batch=512, lr=1e-3, patience=8):
    model = MLP(Xtr.shape[1])
    pos = float(ytr.sum())
    neg = float((1 - ytr).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )

    tl, vl = [], []
    best_val = float("inf")
    best_state = None
    bad = 0
    ntr = Xtr.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(ntr)
        total, count = 0.0, 0
        for i in range(0, ntr, batch):
            idx = perm[i : i + batch]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        train_loss = total / count

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        sched.step(val_loss)
        tl.append(train_loss)
        vl.append(val_loss)
        if val_loss < best_val - 1e-4:
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
    return model, tl, vl


def predict(model, X, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            out.append(torch.sigmoid(model(X[i : i + batch])).numpy())
    return np.concatenate(out)


def metrics_block(name, y, p):
    return {
        "split": name,
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "log_loss": round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 4),
        "accuracy": round(float(((p >= 0.5).astype(int) == y).mean()), 4),
    }


def derive_correctness(
    p_yes: np.ndarray, is_buy: np.ndarray, is_token1: np.ndarray
) -> np.ndarray:
    """P(trade correct | direction, p_yes).

    Trade correct iff:
      (BUY on winning token) OR (SELL on losing token)

    Under the event "YES wins":
      BUY token1 → correct      — P(correct) = p_yes
      BUY token2 → incorrect    — P(correct) = 1 − p_yes
      SELL token1 → incorrect   — P(correct) = 1 − p_yes
      SELL token2 → correct     — P(correct) = p_yes

    So P(correct) = p_yes iff (BUY & token1) OR (SELL & token2).
    Equivalent formulation: P(correct) = p_yes if (is_buy XOR not is_token1) is False
    Simpler: P(correct) = p_yes if is_buy == is_token1 else 1 − p_yes
    """
    align = is_buy == is_token1
    return np.where(align, p_yes, 1.0 - p_yes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labeled", default=str(ROOT / "data" / "iran_strike_labeled.parquet")
    )
    ap.add_argument("--out", default=str(ROOT / "data" / "mlp_reframed_outputs"))
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    df, feats = load_and_prep(Path(args.labeled))
    tr = df[df["bucket"] == "train"].reset_index(drop=True)
    va = df[df["bucket"] == "val"].reset_index(drop=True)
    te = df[df["bucket"] == "test"].reset_index(drop=True)

    print(f"\ntarget `market_resolves_yes` distribution:")
    for bucket, subdf in [("train", tr), ("val", va), ("test", te)]:
        yes_rate = subdf["market_resolves_yes"].mean()
        print(f"  {bucket}: {len(subdf):>7,} rows,  YES rate = {yes_rate:.3f}")

    Xtr, ytr, scaler = to_tensors(tr, feats, None)
    Xva, yva, _ = to_tensors(va, feats, scaler)
    Xte, yte, _ = to_tensors(te, feats, scaler)
    print(f"\ntrain={Xtr.shape}  val={Xva.shape}  test={Xte.shape}")

    print("\ntraining MLP (reframed target)...")
    model, tl, vl = train(Xtr, ytr, Xva, yva, epochs=args.epochs)

    # Raw predictions for YES
    p_val_yes = predict(model, Xva)
    p_te_yes = predict(model, Xte)

    # Isotonic on val
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_yes, va["market_resolves_yes"].values)
    p_val_yes_cal = iso.transform(p_val_yes)
    p_te_yes_cal = iso.transform(p_te_yes)

    # Target-direct metrics
    print("\ntarget-direct (P(YES resolves)):")
    target_metrics = []
    for name, subdf, p in [("val", va, p_val_yes_cal), ("test", te, p_te_yes_cal)]:
        m = metrics_block(
            f"{name}_p_yes_calibrated", subdf["market_resolves_yes"].values, p
        )
        target_metrics.append(m)
        print(f"  {m}")

    # Derived per-trade correctness metrics
    print("\nderived per-trade correctness (p_correct from direction + p_yes):")
    derived_metrics = []
    pred_rows = []
    for name, subdf, p_yes in [("val", va, p_val_yes_cal), ("test", te, p_te_yes_cal)]:
        p_correct = derive_correctness(
            p_yes,
            subdf["is_buy"].values.astype(int),
            subdf["is_token1"].values.astype(int),
        )
        m = metrics_block(f"{name}_p_correct", subdf["bet_correct"].values, p_correct)
        derived_metrics.append(m)
        print(f"  {m}")

        # Gap-based signal: p_yes minus market's implied P(YES)
        gap_yes = p_yes - subdf["market_implied_yes_prob"].values
        # ...and derived gap on correctness space
        gap_correct = p_correct - subdf["market_implied_prob"].values
        gap_roc = roc_auc_score(subdf["bet_correct"].values, gap_correct)
        print(
            f"  {name}_gap_correct: mean_gap={gap_correct.mean():+.4f}  |gap|={np.abs(gap_correct).mean():.4f}  gap_roc={gap_roc:.4f}"
        )

        pred_rows.append(
            subdf.assign(
                p_yes=p_yes,
                p_correct=p_correct,
                gap_yes=gap_yes,
                gap_correct=gap_correct,
                split=name,
            )[
                [
                    "wallet",
                    "condition_id",
                    "timestamp",
                    "bet_correct",
                    "market_resolves_yes",
                    "market_implied_prob",
                    "market_implied_yes_prob",
                    "p_yes",
                    "p_correct",
                    "gap_yes",
                    "gap_correct",
                    "split",
                ]
            ]
        )

    pd.concat(pred_rows).to_parquet(out_dir / "predictions.parquet", index=False)

    # Save artifacts
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    (out_dir / "feature_list.json").write_text(json.dumps(feats, indent=2))
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {"target_yes": target_metrics, "derived_bet_correct": derived_metrics},
            indent=2,
        )
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tl, label="train", color="#4C72B0")
    ax.plot(vl, label="val", color="#DD8452")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Reframed MLP training loss — target=P(YES)")
    ax.legend()
    fig.savefig(out_dir / "loss_curve.png", bbox_inches="tight", dpi=140)
    plt.close(fig)

    print(f"\nall outputs in {out_dir}/")


if __name__ == "__main__":
    main()
