"""
06_baseline_idea1.py

LogReg baseline for cohort design "idea 1" (multi-market train on US-strike-Iran
ladder; multi-market test on US x Iran ceasefire ladder).

Pipeline:
  1. Load data/archive/alex/{train,test}.parquet (raw HF trade rows)
  2. Join to data/archive/alex/markets_subset.parquet for metadata
  3. Derive `bet_correct` target from outcome_prices + nonusdc_side + taker_direction
  4. Engineer 16 market-agnostic, no-lookahead features
  5. 5-fold GroupKFold CV on train (groups = market_id), time-respecting within fold
  6. LogReg with class_weight='balanced'; small C grid via OOF AUC
  7. Isotonic calibration fit on out-of-fold predictions
  8. Score on test (AUC, Brier, ECE, per-market breakdown)
  9. Save outputs to alex/outputs/baselines/idea1_v1/

Constraints:
  - No wallet enrichment. No price-level features (price is withheld so p_hat is independent).
  - Market-agnostic: no absolute-scale features that identify markets.
  - TF/Keras-only constraint applies to NN models — irrelevant here (sklearn LogReg).
"""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]  # alex/
DATA = ROOT.parent / "data" / "archive" / "alex"
OUT = ROOT / "outputs" / "baselines" / "idea1_v2"
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
C_GRID = [0.01, 0.1, 1.0, 10.0]
RANDOM_SEED = 42

# Apr 7 2026: Trump announced US-Iran ceasefire (used as fallback target derivation).
# End-of-day UTC so a "by April 7" market with deadline 00:00 UTC correctly resolves NO
# (announcement came during the day, after the deadline).
CEASEFIRE_ANNOUNCEMENT_UTC = pd.Timestamp("2026-04-07T23:59:59", tz="UTC")
# Feb 28 2026 06:35 UTC: US strike on Iran (canonical resolution event for strike ladder)
STRIKE_EVENT_UTC = pd.Timestamp("2026-02-28T06:35:00", tz="UTC")

# Parse deadline from question title — dataset's end_date is unreliable for some markets
import re

DEADLINE_RE = re.compile(
    r"by\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d+)(?:,\s*(\d{4}))?",
    re.IGNORECASE,
)


def parse_deadline_from_question(
    question: str, default_year: int = 2026
) -> pd.Timestamp | None:
    """Parse 'by [Month] [Day]?' or 'by [Month] [Day], [Year]?' to UTC timestamp at 00:00."""
    m = DEADLINE_RE.search(question)
    if not m:
        return None
    month_str, day_str, year_str = m.groups()
    year = int(year_str) if year_str else default_year
    try:
        return pd.Timestamp(f"{month_str} {day_str} {year}", tz="UTC")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Load + target derivation
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA / "train.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    markets = pd.read_parquet(DATA / "markets_subset.parquet")
    print(f"[load] train rows: {len(train):,}")
    print(f"[load] test rows:  {len(test):,}")
    print(f"[load] markets: {len(markets)}")
    print(f"[load] train cols: {list(train.columns)}")
    return train, test, markets


def derive_winning_token(markets: pd.DataFrame) -> pd.DataFrame:
    """Resolve each market's winning token from `outcome_prices` + deadline fallback."""
    m = markets.copy()
    winning = []
    for _, row in m.iterrows():
        # Try outcome_prices first
        try:
            prices = ast.literal_eval(row["outcome_prices"])
            p1, p2 = float(prices[0]), float(prices[1])
            if abs(p1 - 1.0) < 0.01 and abs(p2 - 0.0) < 0.01:
                winning.append("token1")  # answer1 won (typically YES)
                continue
            if abs(p1 - 0.0) < 0.01 and abs(p2 - 1.0) < 0.01:
                winning.append("token2")  # answer2 won (typically NO)
                continue
        except Exception:
            pass

        # Fallback: parse deadline from question (dataset end_date is unreliable
        # for some markets — sometimes shows snapshot date, not stated deadline).
        question = row["question"]
        parsed_deadline = parse_deadline_from_question(question)
        if parsed_deadline is None:
            # Last-resort: use dataset end_date
            parsed_deadline = (
                pd.Timestamp(row["end_date"]).tz_convert("UTC")
                if hasattr(row["end_date"], "tzinfo") and row["end_date"].tzinfo
                else pd.Timestamp(row["end_date"], tz="UTC")
            )
        cohort = row["cohort"]

        if cohort == "train":
            # "US strikes Iran by [date]". Strike happened Feb 28 06:35 UTC.
            # YES (token1) wins iff deadline >= STRIKE_EVENT_UTC.
            if parsed_deadline >= STRIKE_EVENT_UTC:
                winning.append("token1")
            else:
                winning.append("token2")
        elif cohort == "test":
            # "US x Iran ceasefire by [date]". Trump announced Apr 7 (end-of-day UTC).
            # YES (token1) wins iff deadline > CEASEFIRE_ANNOUNCEMENT_UTC (i.e.,
            # the announcement was BEFORE the deadline).
            if parsed_deadline >= CEASEFIRE_ANNOUNCEMENT_UTC:
                winning.append("token1")
            else:
                winning.append("token2")
        else:
            winning.append(None)
    m["winning_token"] = winning
    return m


def derive_bet_correct(trades: pd.DataFrame, markets: pd.DataFrame) -> pd.Series:
    """For each trade: 1 if the trade ended up on the winning side, else 0.

    Logic:
      BUY of token X = bet that X wins  → correct iff X == winning_token
      SELL of token X = bet that X loses → correct iff X != winning_token
    """
    win_map = dict(zip(markets["id"].astype(str), markets["winning_token"]))
    market_id = trades["market_id"].astype(str)
    winning = market_id.map(win_map)

    is_buy = trades["taker_direction"].astype(str).str.upper().eq("BUY")
    side = trades["nonusdc_side"].astype(str)

    # Map nonusdc_side ("token1"/"token2") to bet correctness
    correct = np.where(
        is_buy,
        (side == winning).astype(int),
        (side != winning).astype(int),
    )
    return pd.Series(correct, index=trades.index, name="bet_correct")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(trades: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Build the 16-feature matrix. No-lookahead, market-agnostic."""
    df = trades.copy()
    # market metadata join
    md = markets[["id", "created_at", "end_date", "winning_token"]].copy()
    md["id"] = md["id"].astype(str)
    df["market_id"] = df["market_id"].astype(str)
    df = df.merge(md, left_on="market_id", right_on="id", how="left")

    # Sort by market + timestamp once
    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    # Convert timestamp to datetime
    df["ts_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # --- Trade-local features
    # log_size: trade tokens. usd_amount / price ≈ token amount.
    # Some HF schemas have `size` (in tokens) directly. Use that if present.
    if "size" in df.columns:
        df["log_size"] = np.log1p(df["size"].clip(lower=0))
    else:
        # Derive from usd_amount / price
        if "price" in df.columns:
            df["_tokens"] = df["usd_amount"] / df["price"].clip(lower=0.001)
            df["log_size"] = np.log1p(df["_tokens"].clip(lower=0))
        else:
            df["log_size"] = np.log1p(df["usd_amount"].clip(lower=0))  # best effort

    # log_trade_value_usd (winsorise at 1st/99th percentile of TRAIN; here we just clip per-cohort)
    df["_uval"] = df["usd_amount"].clip(lower=0)
    p1, p99 = df["_uval"].quantile([0.01, 0.99])
    df["_uval_w"] = df["_uval"].clip(lower=p1, upper=p99)
    df["log_trade_value_usd"] = np.log1p(df["_uval_w"])

    # side_buy: 1 if taker_direction=BUY, 0 if SELL
    df["side_buy"] = df["taker_direction"].astype(str).str.upper().eq("BUY").astype(int)

    # outcome_yes: 1 if nonusdc_side=token1 (YES side, since answer1='Yes' for our cohorts)
    df["outcome_yes"] = (df["nonusdc_side"].astype(str) == "token1").astype(int)

    # --- Time features
    open_ts = pd.to_datetime(df["created_at"], utc=True)
    end_ts = pd.to_datetime(df["end_date"], utc=True)
    df["_secs_to_deadline"] = (end_ts - df["ts_dt"]).dt.total_seconds().clip(lower=1)
    df["_secs_since_open"] = (df["ts_dt"] - open_ts).dt.total_seconds().clip(lower=1)
    total_lifetime = (end_ts - open_ts).dt.total_seconds().clip(lower=1)

    df["log_time_to_deadline_hours"] = np.log1p(df["_secs_to_deadline"] / 3600)
    df["pct_time_elapsed"] = (df["_secs_since_open"] / total_lifetime).clip(0, 1)

    # log_time_since_last_trade: per market gap between successive trades
    df["_gap_sec"] = df.groupby("market_id")["timestamp"].diff().fillna(0).clip(lower=0)
    df["log_time_since_last_trade"] = np.log1p(df["_gap_sec"])

    # is_first_trade_after_quiet: gap > 1h (3600s)
    df["is_first_trade_after_quiet"] = (df["_gap_sec"] > 3600).astype(int)

    # hour_of_day cyclical
    hour = df["ts_dt"].dt.hour + df["ts_dt"].dt.minute / 60.0
    df["hour_of_day_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_of_day_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # --- Market-state features (per-market, time-respecting rolling)
    # log_n_trades_to_date: cumulative trade count up to but not including this trade
    df["_n_to_date"] = df.groupby("market_id").cumcount()
    df["log_n_trades_to_date"] = np.log1p(df["_n_to_date"])

    # market_buy_share_running: running buy/sell share to date
    df["_buy_cumvol"] = (
        df.groupby("market_id")
        .apply(
            lambda g: (g["side_buy"] * g["_uval"]).cumsum().shift(1).fillna(0),
            include_groups=False,
        )
        .reset_index(level=0, drop=True)
    )
    df["_total_cumvol"] = (
        df.groupby("market_id")["_uval"]
        .apply(lambda s: s.cumsum().shift(1).fillna(0))
        .reset_index(level=0, drop=True)
    )
    df["market_buy_share_running"] = (
        df["_buy_cumvol"] / df["_total_cumvol"].clip(lower=1.0)
    ).clip(0, 1)

    # Rolling time-windowed features per market.
    # For each trade compute features over the prior 1h and 5min windows
    # using time-indexed groupby rolling.
    df = df.set_index("ts_dt")

    if "price" in df.columns:
        # 1h price volatility (std of price changes, not levels)
        df["_price_diff"] = df.groupby("market_id")["price"].diff().fillna(0)
        df["market_price_vol_last_1h"] = (
            df.groupby("market_id")["_price_diff"]
            .rolling("1h", closed="left")
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        df["market_price_vol_last_5min"] = (
            df.groupby("market_id")["_price_diff"]
            .rolling("5min", closed="left")
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
    else:
        df["market_price_vol_last_1h"] = 0.0
        df["market_price_vol_last_5min"] = 0.0

    # log_recent_volume_1h: USD volume in last 1h, exclusive of current trade
    df["log_recent_volume_1h"] = np.log1p(
        df.groupby("market_id")["_uval"]
        .rolling("1h", closed="left")
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Order flow imbalance 5min: (buy_vol - sell_vol) / total_vol over last 5min
    df["_signed_uval"] = df["_uval"] * (
        2 * df["side_buy"] - 1
    )  # +uval if buy, -uval if sell
    buy_minus_sell_5min = (
        df.groupby("market_id")["_signed_uval"]
        .rolling("5min", closed="left")
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    total_5min = (
        df.groupby("market_id")["_uval"]
        .rolling("5min", closed="left")
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    df["order_flow_imbalance_5min"] = (
        buy_minus_sell_5min / total_5min.clip(lower=1.0)
    ).clip(-1, 1)

    # trade_size_to_recent_volume_ratio: this trade size / last-1h volume
    last_1h_vol = (
        df.groupby("market_id")["_uval"]
        .rolling("1h", closed="left")
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    df["trade_size_to_recent_volume_ratio"] = (
        df["_uval"] / last_1h_vol.clip(lower=1.0)
    ).clip(0, 100)

    df = df.reset_index()
    return df


FEATURE_COLS = [
    "log_size",
    # log_trade_value_usd dropped in v2 — combined with log_size it leaks
    # market_implied_prob via log(value) − log(size) = log(price), which we
    # explicitly withhold so p_hat stays independent for the trading-rule edge.
    "side_buy",
    "outcome_yes",
    "log_time_to_deadline_hours",
    "pct_time_elapsed",
    "log_time_since_last_trade",
    "is_first_trade_after_quiet",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "log_n_trades_to_date",
    "market_buy_share_running",
    "market_price_vol_last_1h",
    "market_price_vol_last_5min",
    "log_recent_volume_1h",
    "order_flow_imbalance_5min",
    "trade_size_to_recent_volume_ratio",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def metrics_block(y_true, y_prob, label):
    return {
        f"{label}_auc": float(roc_auc_score(y_true, y_prob)),
        f"{label}_brier": float(brier_score_loss(y_true, y_prob)),
        f"{label}_ece": expected_calibration_error(y_true, y_prob),
        f"{label}_n": int(len(y_true)),
        f"{label}_pos_rate": float(y_true.mean()),
    }


# ---------------------------------------------------------------------------
# CV + train/test
# ---------------------------------------------------------------------------


def cv_train(X, y, groups, C_grid=C_GRID, n_splits=N_FOLDS):
    """5-fold GroupKFold. For each C, compute mean OOF AUC. Pick best C."""
    gkf = GroupKFold(n_splits=n_splits)
    results = {}
    oof_preds_per_C = {}
    for C in C_grid:
        oof = np.zeros(len(y), dtype=float)
        fold_aucs = []
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
            # Time-respecting: sort within fold's train set by row order (already sorted by ts)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X.iloc[tr_idx])
            X_va = scaler.transform(X.iloc[va_idx])
            clf = LogisticRegression(
                C=C,
                penalty="l2",
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_SEED,
            )
            clf.fit(X_tr, y.iloc[tr_idx])
            preds = clf.predict_proba(X_va)[:, 1]
            oof[va_idx] = preds
            fold_aucs.append(roc_auc_score(y.iloc[va_idx], preds))
        results[C] = {
            "mean_oof_auc": float(np.mean(fold_aucs)),
            "std_oof_auc": float(np.std(fold_aucs)),
            "oof_brier": float(brier_score_loss(y, oof)),
        }
        oof_preds_per_C[C] = oof
        print(
            f"  C={C}: OOF AUC = {results[C]['mean_oof_auc']:.4f} ± {results[C]['std_oof_auc']:.4f}"
        )
    best_C = max(results.keys(), key=lambda c: results[c]["mean_oof_auc"])
    print(
        f"[cv] best C: {best_C} (mean OOF AUC: {results[best_C]['mean_oof_auc']:.4f})"
    )
    return best_C, results, oof_preds_per_C[best_C]


def fit_final(X, y, C):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(
        C=C,
        penalty="l2",
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_s, y)
    return scaler, clf


def predict_calibrated(scaler, clf, calibrator, X):
    X_s = scaler.transform(X)
    raw = clf.predict_proba(X_s)[:, 1]
    cal = calibrator.transform(raw)
    return raw, cal


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_roc(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC — test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(y_true, y_prob, path, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    bin_means = []
    bin_confs = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            bin_means.append(y_true[mask].mean())
            bin_confs.append(y_prob[mask].mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(bin_confs, bin_means, marker="o")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Calibration — test")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    train_raw, test_raw, markets = load_data()
    markets = derive_winning_token(markets)
    print(
        f"[target] winning_token derived for {markets['winning_token'].notna().sum()} / {len(markets)} markets"
    )

    print("[features] engineering for train")
    train_df = engineer_features(train_raw, markets)
    train_df["bet_correct"] = derive_bet_correct(train_df, markets)
    print(f"[features] engineering for test")
    test_df = engineer_features(test_raw, markets)
    test_df["bet_correct"] = derive_bet_correct(test_df, markets)

    # Drop rows with missing target
    train_df = train_df.dropna(subset=["bet_correct"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["bet_correct"]).reset_index(drop=True)
    print(
        f"[target] train n={len(train_df):,}, pos_rate={train_df['bet_correct'].mean():.3f}"
    )
    print(
        f"[target] test  n={len(test_df):,}, pos_rate={test_df['bet_correct'].mean():.3f}"
    )

    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df["bet_correct"].astype(int)
    g_train = train_df["market_id"]

    X_test = test_df[FEATURE_COLS].fillna(0)
    y_test = test_df["bet_correct"].astype(int)

    # CV
    print(f"[cv] {N_FOLDS}-fold GroupKFold, C grid: {C_GRID}")
    best_C, cv_results, oof = cv_train(X_train, y_train, g_train)

    # Calibrator on OOF
    print("[calibration] fitting isotonic on OOF predictions")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof, y_train)

    # Final model on full train
    scaler, clf = fit_final(X_train, y_train, best_C)

    # Score on test
    raw_test, cal_test = predict_calibrated(scaler, clf, calibrator, X_test)
    test_metrics_uncal = metrics_block(y_test, raw_test, "test_raw")
    test_metrics_cal = metrics_block(y_test, cal_test, "test_calibrated")

    # Per-market test metrics
    per_market = {}
    for mid in test_df["market_id"].unique():
        mask = test_df["market_id"] == mid
        if mask.sum() < 20:
            continue
        if y_test[mask].nunique() < 2:
            continue
        per_market[str(mid)] = {
            "n": int(mask.sum()),
            "pos_rate": float(y_test[mask].mean()),
            "auc": float(roc_auc_score(y_test[mask], raw_test[mask])),
            "brier": float(brier_score_loss(y_test[mask], cal_test[mask])),
        }

    # Feature importance (LogReg coefficients on standardised features)
    coefs = dict(zip(FEATURE_COLS, clf.coef_[0].tolist()))
    coefs_sorted = dict(sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True))

    # Save outputs
    metrics = {
        "best_C": best_C,
        "cv_results_per_C": {str(k): v for k, v in cv_results.items()},
        "oof_brier": float(brier_score_loss(y_train, oof)),
        "oof_auc": float(roc_auc_score(y_train, oof)),
        **test_metrics_uncal,
        **test_metrics_cal,
        "n_features": len(FEATURE_COLS),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "feature_list.json").write_text(json.dumps(FEATURE_COLS, indent=2))
    (OUT / "feature_importance.json").write_text(json.dumps(coefs_sorted, indent=2))
    (OUT / "per_market_test.json").write_text(json.dumps(per_market, indent=2))
    (OUT / "config.json").write_text(
        json.dumps(
            {
                "n_folds": N_FOLDS,
                "C_grid": C_GRID,
                "random_seed": RANDOM_SEED,
                "strike_event_utc": STRIKE_EVENT_UTC.isoformat(),
                "ceasefire_announcement_utc": CEASEFIRE_ANNOUNCEMENT_UTC.isoformat(),
            },
            indent=2,
        )
    )
    plot_roc(y_test.values, raw_test, OUT / "roc_test.png")
    plot_calibration(y_test.values, cal_test, OUT / "calibration_test.png")

    print("=" * 60)
    print(f"DONE — outputs in {OUT}")
    print(f"  best C: {best_C}")
    print(f"  OOF AUC: {metrics['oof_auc']:.4f}")
    print(f"  test AUC (raw):  {test_metrics_uncal['test_raw_auc']:.4f}")
    print(f"  test AUC (cal):  {test_metrics_cal['test_calibrated_auc']:.4f}")
    print(f"  test Brier (cal): {test_metrics_cal['test_calibrated_brier']:.4f}")
    print(f"  test ECE (cal):  {test_metrics_cal['test_calibrated_ece']:.4f}")
    print(
        f"  per-market AUC range: "
        f"{min(v['auc'] for v in per_market.values()):.3f} → "
        f"{max(v['auc'] for v in per_market.values()):.3f}"
        if per_market
        else "  per-market: <too few resolved markets>"
    )
    print(f"  top 3 features by |coef|: {list(coefs_sorted.keys())[:3]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
