"""
fix_cost_and_rerun.py — fix the pre_yes_price bug and re-run backtest math.

Problem: pre_trade_price in features is the previous trade's price regardless of
which token was traded. About half the time it's the YES-token price, half the
time the NO-token price. The cost/edge calc in 10_backtest.py and 11_realistic_backtest.py
assumed pre_trade_price = YES probability — wrong half the time.

Fix: compute proper pre_yes_price by tracking last-observed token1 and token2
prices separately. pre_yes_price = last_token1_price (or 1 - last_token2_price
if no token1 trade yet, or 0.5 if neither).

Output: re-runs all backtest comparisons with corrected costs:
  - top-1% by p_hat (unchanged: doesn't use cost)
  - top-1% by edge (CHANGED: uses corrected edge)
  - general_ev rule (CHANGED)
  - home_run rule (CHANGED)
  - PnL / ROI numbers
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PRED = ROOT / ".scratch" / "backtest"
OUT = ROOT / ".scratch" / "pressure_tests"
OUT.mkdir(parents=True, exist_ok=True)


def compute_pre_yes_price(raw: pd.DataFrame) -> np.ndarray:
    """For each trade, compute the YES-token price as observed BEFORE the trade.

    Logic:
      - Forward-fill last-observed token1 price within each market
      - Forward-fill last-observed token2 price within each market
      - Shift(1) to exclude the current trade
      - pre_yes_price = last_t1 if exists, else 1 - last_t2, else 0.5
    """
    df = raw.copy()
    df["market_id"] = df["market_id"].astype(str)
    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    df["_t1_price"] = np.where(
        df["nonusdc_side"].astype(str) == "token1", df["price"], np.nan
    )
    df["_t2_price"] = np.where(
        df["nonusdc_side"].astype(str) == "token2", df["price"], np.nan
    )

    df["_last_t1"] = df.groupby("market_id")["_t1_price"].ffill().shift(1)
    df["_last_t2"] = df.groupby("market_id")["_t2_price"].ffill().shift(1)

    # Reset first-of-market shift carryover
    first_idx = df.groupby("market_id").head(1).index
    df.loc[first_idx, ["_last_t1", "_last_t2"]] = np.nan

    pre_yes = df["_last_t1"].fillna(1 - df["_last_t2"]).fillna(0.5).clip(0, 1)
    df["pre_yes_price_corrected"] = pre_yes
    return df


def main():
    print("=" * 60)
    print("Fix pre_yes_price + re-run backtest math")
    print("=" * 60)

    test_raw = pd.read_parquet(DATA / "test.parquet")
    test_feat = pd.read_parquet(DATA / "test_features.parquet")
    print(f"test rows: {len(test_raw):,}")

    # Compute corrected pre_yes_price (sorted by market_id, timestamp)
    test_with = compute_pre_yes_price(test_raw)
    test_with["market_id"] = test_with["market_id"].astype(str)
    test_feat["market_id"] = test_feat["market_id"].astype(str)
    test_feat = test_feat.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    test_with = test_with.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    # Sanity: alignment
    assert (test_feat["timestamp"].values == test_with["timestamp"].values).all(), (
        "ts mismatch"
    )
    assert (test_feat["market_id"].values == test_with["market_id"].values).all(), (
        "mid mismatch"
    )

    pre_yes_old = test_feat["pre_trade_price"].values
    pre_yes_new = test_with["pre_yes_price_corrected"].values
    diff = pre_yes_new - pre_yes_old

    n_diff = int((np.abs(diff) > 0.01).sum())
    print(
        f"\npre_yes_price comparison: {n_diff:,} of {len(diff):,} trades have |diff| > 0.01 ({n_diff / len(diff) * 100:.1f}%)"
    )
    print(f"  mean abs diff: {np.abs(diff).mean():.4f}")
    print(f"  max abs diff:  {np.abs(diff).max():.4f}")

    # Show sample where the bug bites hardest
    print(f"\n  Old vs New pre_yes_price (sample of 10 with biggest divergence):")
    big_diff_idx = np.argsort(-np.abs(diff))[:10]
    for i in big_diff_idx:
        print(
            f"    mid {test_feat['market_id'].iloc[i]:>9}, "
            f"old={pre_yes_old[i]:.3f} new={pre_yes_new[i]:.3f} delta={diff[i]:+.3f} "
            f"outcome_yes={int(test_feat['outcome_yes'].iloc[i])}"
        )

    # ---- Recompute cost / edge / PnL with corrected pre_yes_price -----------
    side_buy = test_feat["side_buy"].values.astype(int)
    outcome_yes = test_feat["outcome_yes"].values.astype(int)
    bet_correct = test_feat["bet_correct"].astype(int).values

    trader_side_wins_yes = side_buy * outcome_yes + (1 - side_buy) * (1 - outcome_yes)

    def compute_metrics(pre_yes, label, cost_floor=0.05):
        cost = np.where(trader_side_wins_yes == 1, pre_yes, 1 - pre_yes)
        cost = np.clip(cost, cost_floor, 1.0 - cost_floor)
        rows = []
        for model in ["logreg_l2", "random_forest", "hist_gbm"]:
            d = np.load(PRED / f"preds_{model}.npz")
            # Reorder to match sorted test
            test_orig = pd.read_parquet(DATA / "test_features.parquet")
            test_orig["market_id"] = test_orig["market_id"].astype(str)
            test_orig["_orig_idx"] = np.arange(len(test_orig))
            sort_key = test_orig.sort_values(["market_id", "timestamp"])[
                "_orig_idx"
            ].values

            cal = d["cal"][sort_key]
            edge = cal - cost
            n = len(cal)

            for k_pct in [0.001, 0.01, 0.05, 0.10, 0.25]:
                k = max(1, int(n * k_pct))
                top_idx_phat = np.argsort(cal)[-k:]
                top_idx_edge = np.argsort(edge)[-k:]
                rows.append(
                    {
                        "model": model,
                        "label": label,
                        "k_pct": k_pct,
                        "k": k,
                        "top_phat_hit": float(bet_correct[top_idx_phat].mean()),
                        "top_edge_hit": float(bet_correct[top_idx_edge].mean()),
                        "top_edge_mean_edge": float(edge[top_idx_edge].mean()),
                    }
                )
        return pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("Comparison: top-k precision OLD (buggy) vs NEW (corrected) cost calc")
    print("=" * 60)
    old_df = compute_metrics(pre_yes_old, "OLD")
    new_df = compute_metrics(pre_yes_new, "NEW")

    # Merge for side-by-side
    merged = old_df.merge(
        new_df,
        on=["model", "k_pct", "k"],
        suffixes=("_old", "_new"),
    )
    print(
        f"\n{'model':<14} {'k_pct':>7} {'top_phat':>10} (unchanged) | "
        f"{'top_edge_old':>14} {'top_edge_new':>14}  Δ"
    )
    for _, r in merged.iterrows():
        delta = r["top_edge_hit_new"] - r["top_edge_hit_old"]
        print(
            f"{r['model']:<14} {r['k_pct']:>7.3f} "
            f"{r['top_phat_hit_new']:>10.3f}            | "
            f"{r['top_edge_hit_old']:>14.3f} {r['top_edge_hit_new']:>14.3f}  {delta:+.3f}"
        )

    # Save
    out_csv = OUT / "cost_fix_comparison.csv"
    merged.to_csv(out_csv, index=False)
    print(f"\nResults: {out_csv}")

    # ---- Re-run realistic backtest ROI with corrected cost ------------------
    print("\n" + "=" * 60)
    print("Realistic backtest ROI — OLD vs NEW")
    print("=" * 60)

    # Use simple unconstrained PnL: $100 stake per signal, cost floor 0.05.
    # No copycat scaler here (simpler comparison; the ratio matters more than absolute).
    def simple_pnl(pre_yes, mask, cost_floor=0.05, stake=100.0):
        cost = np.where(trader_side_wins_yes == 1, pre_yes, 1 - pre_yes)
        cost = np.clip(cost, cost_floor, 1.0 - cost_floor)
        n_sel = int(mask.sum())
        if n_sel == 0:
            return {"n": 0, "pnl": 0.0, "hit": None}
        pnl = np.where(
            bet_correct[mask] == 1, stake * (1 - cost[mask]) / cost[mask], -stake
        )
        return {
            "n": n_sel,
            "pnl": float(pnl.sum()),
            "hit": float(bet_correct[mask].mean()),
        }

    print(f"\n{'strategy':<22} {'OLD pnl':>14} {'NEW pnl':>14}  ratio")
    for model in ["random_forest", "hist_gbm"]:
        d = np.load(PRED / f"preds_{model}.npz")
        test_orig = pd.read_parquet(DATA / "test_features.parquet")
        test_orig["market_id"] = test_orig["market_id"].astype(str)
        test_orig["_orig_idx"] = np.arange(len(test_orig))
        sort_key = test_orig.sort_values(["market_id", "timestamp"])["_orig_idx"].values
        cal = d["cal"][sort_key]
        n = len(cal)

        # General +EV with old vs new
        for label, pre_yes in [("OLD", pre_yes_old), ("NEW", pre_yes_new)]:
            cost = np.where(trader_side_wins_yes == 1, pre_yes, 1 - pre_yes)
            cost = np.clip(cost, 0.05, 0.95)
            edge = cal - cost
            mask_ev = edge > 0.02
            r = simple_pnl(pre_yes, mask_ev)

        # Side-by-side pnls
        for strat_name, strat_fn in [
            ("general_ev>0.02", lambda pe, c: ((cal - c) > 0.02)),
            (
                "top_1pct_phat",
                lambda pe, c: np.isin(np.arange(n), np.argsort(cal)[-int(n * 0.01) :]),
            ),
            (
                "top_1pct_edge",
                lambda pe, c: np.isin(
                    np.arange(n), np.argsort(cal - c)[-int(n * 0.01) :]
                ),
            ),
        ]:
            cost_old = np.clip(
                np.where(trader_side_wins_yes == 1, pre_yes_old, 1 - pre_yes_old),
                0.05,
                0.95,
            )
            cost_new = np.clip(
                np.where(trader_side_wins_yes == 1, pre_yes_new, 1 - pre_yes_new),
                0.05,
                0.95,
            )
            mask_old = strat_fn(pre_yes_old, cost_old)
            mask_new = strat_fn(pre_yes_new, cost_new)
            r_old = simple_pnl(pre_yes_old, mask_old)
            r_new = simple_pnl(pre_yes_new, mask_new)
            ratio = r_new["pnl"] / r_old["pnl"] if r_old["pnl"] != 0 else float("nan")
            print(
                f"  {model[:5]} {strat_name:<16} ${r_old['pnl']:>12.0f} ${r_new['pnl']:>12.0f}  {ratio:>5.2f}x"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
