"""
12_train_mlp.py
Train and evaluate the primary MLP and the sklearn baselines (logistic regression,
random forest) for the mispricing project. Scaffold only — safe to run, but the
intent at this point is a syntax-clean skeleton ready for the full 75-column
frame once Layer 6 and cross-market entropy land.

Maps to project_plan.md:
  §5.1  Primary model: MLP (SELU, Glorot, dropout 0.2-0.4, batchnorm, Adam+LR sched, BCE)
  §5.2  Calibration: isotonic on validation slice, Brier and ECE
  §5.4  Baselines: logistic regression, random forest (naive market baseline handled in backtest)
  §5.5  Class imbalance / winsorisation of trade_value_usd and wallet_prior_volume_usd
  §4    Pre-modelling filter: settlement_minus_trade_sec > 0
        market_implied_prob excluded from features (it is the benchmark)
        split column drives train / val / test
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPORT_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = REPORT_DIR / "data" / "03_consolidated_dataset.csv"
OUT_DIR = REPORT_DIR / "outputs" / "modelling"

# Columns that are NOT features. Everything else in the frame is treated as a
# feature, which lets the script auto-adapt when Layer 6 and the cross-market
# entropy column are appended.
NON_FEATURE_COLS = {
    # Identifiers and raw metadata
    "proxyWallet",
    "asset",
    "transactionHash",
    "condition_id",
    "conditionId",
    "source",
    "title",
    "slug_x",
    "slug_y",
    "icon",
    "eventSlug",
    "outcome",
    "name",
    "pseudonym",
    "bio",
    "profileImage",
    "profileImageOptimized",
    "question",
    "end_date",
    "winning_outcome_index",
    "resolved",
    "resolution_ts",
    "outcomes",
    "is_yes",
    # Raw columns superseded by derived features
    "size",  # replaced by log_size
    "price",  # duplicates market_implied_prob
    "timestamp",  # only used to construct the split column
    # Filtering column
    "settlement_minus_trade_sec",
    # Label, benchmark, split
    "bet_correct",
    "market_implied_prob",  # §4: deliberately excluded — benchmark, not input
    "split",
    # Dropped per data-pipeline-issues.md P0-1 and P0-2:
    # wallet_is_whale_in_market leaks via end-of-market p95 threshold;
    # is_position_exit misfires on first-ever SELL (denominator uses current trade size).
    "wallet_is_whale_in_market",
    "is_position_exit",
}

CATEGORICAL_COLS = {"side"}  # string BUY/SELL, encoded to 0/1

# Columns to winsorise at 1st / 99th percentile (§5.5)
WINSORISE_COLS = ["trade_value_usd", "wallet_prior_volume_usd"]

RANDOM_SEED = 42
BATCH_SIZE = 4096
EPOCHS = 40
EARLY_STOP_PATIENCE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width ECE. y_prob in [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, low_memory=False)
    # §4 / §5.5 pre-modelling filter
    df = df[df["settlement_minus_trade_sec"] > 0].copy()
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return feats


def encode_categoricals(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = df[feats].copy()
    if "side" in out.columns:
        # BUY -> 1, anything else (SELL) -> 0; unknowns become 0.
        out["side"] = (out["side"].astype(str).str.upper() == "BUY").astype(int)
    # Any remaining object columns get dropped as a safety net — modelling
    # should run on numerics only at this stage.
    obj_cols = out.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        out = out.drop(columns=obj_cols)
    return out


def winsorise(
    df: pd.DataFrame, cols: list[str], bounds: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Clip `cols` at their 1st/99th percentile. If `bounds` is given, apply it
    (used to reuse training-set bounds on val/test)."""
    out = df.copy()
    if bounds is None:
        bounds = {}
        for c in cols:
            if c in out.columns:
                lo, hi = out[c].quantile([0.01, 0.99]).tolist()
                bounds[c] = (lo, hi)
    for c, (lo, hi) in bounds.items():
        if c in out.columns:
            out[c] = out[c].clip(lo, hi)
    return out, bounds


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"].copy()
    te = df[df["split"] == "test"].copy()
    return tr, va, te


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """2-4 hidden layers, SELU, Glorot, dropout, BN — per §5.1."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        assert 2 <= len(hidden_dims) <= 4, (
            "project_plan.md §5.1 specifies 2-4 hidden layers"
        )
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            lin = nn.Linear(prev, h)
            nn.init.xavier_uniform_(lin.weight)  # Glorot
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.BatchNorm1d(h), nn.SELU(), nn.Dropout(dropout)]
            prev = h
        out = nn.Linear(prev, 1)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
    dropout: float = 0.3,
    lr: float = 1e-3,
) -> tuple[MLP, list[float], list[float]]:
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")
    model = MLP(X_tr.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2
    )
    loss_fn = nn.BCEWithLogitsLoss()

    tr_ds = TensorDataset(
        torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()
    )
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    X_va_t = torch.from_numpy(X_va).float().to(device)
    y_va_t = torch.from_numpy(y_va).float().to(device)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_state = None
    since_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(X_va_t), y_va_t).item()
        val_losses.append(vl)
        sched.step(vl)

        if vl < best_val - 1e-4:
            best_val = vl
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


def mlp_predict_proba(model: MLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
        return torch.sigmoid(logits).numpy()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("logreg", "rf", "mlp"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    print(f"[load] {DATA_CSV}")
    df = load_frame()
    print(
        f"[load] post-filter rows: {len(df):,} ({df['split'].value_counts().to_dict()})"
    )

    features = select_features(df)
    print(
        f"[features] {len(features)} features in use (total cols in frame: {df.shape[1]})"
    )

    tr, va, te = split_frame(df)
    y_tr = tr["bet_correct"].to_numpy().astype(int)
    y_va = va["bet_correct"].to_numpy().astype(int)
    y_te = te["bet_correct"].to_numpy().astype(int)

    X_tr = encode_categoricals(tr, features)
    X_va = encode_categoricals(va, features)
    X_te = encode_categoricals(te, features)

    final_feats = X_tr.columns.tolist()
    (OUT_DIR / "mlp" / "feature_list.json").write_text(
        json.dumps(final_feats, indent=2)
    )

    # Winsorise using training bounds
    X_tr, wbounds = winsorise(X_tr, WINSORISE_COLS, bounds=None)
    X_va, _ = winsorise(X_va, WINSORISE_COLS, bounds=wbounds)
    X_te, _ = winsorise(X_te, WINSORISE_COLS, bounds=wbounds)

    # Impute + scale (fit on train only)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(imputer.fit_transform(X_tr))
    X_va_np = scaler.transform(imputer.transform(X_va))
    X_te_np = scaler.transform(imputer.transform(X_te))

    # ----- Baseline: Logistic Regression -----
    print("[logreg] fitting")
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    logreg.fit(X_tr_np, y_tr)
    p_va_lr = logreg.predict_proba(X_va_np)[:, 1]
    p_te_lr = logreg.predict_proba(X_te_np)[:, 1]
    (OUT_DIR / "logreg" / "metrics.json").write_text(
        json.dumps(
            {
                "val": probability_metrics(y_va, p_va_lr),
                "test": probability_metrics(y_te, p_te_lr),
            },
            indent=2,
        )
    )
    (OUT_DIR / "logreg" / "feature_list.json").write_text(
        json.dumps(final_feats, indent=2)
    )

    # ----- Baseline: Random Forest -----
    print("[rf] fitting")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    # RF tolerates NaN + unscaled inputs natively; use un-imputed frame for fairness.
    X_tr_rf = X_tr.to_numpy()
    X_va_rf = X_va.to_numpy()
    X_te_rf = X_te.to_numpy()
    rf.fit(np.nan_to_num(X_tr_rf, nan=0.0), y_tr)
    p_va_rf = rf.predict_proba(np.nan_to_num(X_va_rf, nan=0.0))[:, 1]
    p_te_rf = rf.predict_proba(np.nan_to_num(X_te_rf, nan=0.0))[:, 1]
    (OUT_DIR / "rf" / "metrics.json").write_text(
        json.dumps(
            {
                "val": probability_metrics(y_va, p_va_rf),
                "test": probability_metrics(y_te, p_te_rf),
            },
            indent=2,
        )
    )
    (OUT_DIR / "rf" / "feature_list.json").write_text(json.dumps(final_feats, indent=2))

    # ----- Primary: MLP -----
    print("[mlp] training")
    model, train_losses, val_losses = train_mlp(
        X_tr_np, y_tr.astype(float), X_va_np, y_va.astype(float)
    )
    p_va_mlp_raw = mlp_predict_proba(model, X_va_np)
    p_te_mlp_raw = mlp_predict_proba(model, X_te_np)

    # §5.2: isotonic calibration fit on the validation slice, applied to val + test.
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_va_mlp_raw, y_va)
    p_va_mlp = iso.transform(p_va_mlp_raw)
    p_te_mlp = iso.transform(p_te_mlp_raw)

    metrics_mlp = {
        "val_raw": probability_metrics(y_va, p_va_mlp_raw),
        "val_calibrated": probability_metrics(y_va, p_va_mlp),
        "test_raw": probability_metrics(y_te, p_te_mlp_raw),
        "test_calibrated": probability_metrics(y_te, p_te_mlp),
        "train_epochs": len(train_losses),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
    }
    (OUT_DIR / "mlp" / "metrics.json").write_text(json.dumps(metrics_mlp, indent=2))

    # Loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("MLP training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mlp" / "loss_curve.png", dpi=140)
    plt.close(fig)

    print(f"[done] feature count: {len(final_feats)} | outputs: {OUT_DIR}")


if __name__ == "__main__":
    run()
