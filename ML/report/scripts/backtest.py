"""Streaming event-replay backtest for the Iran-strike trading signal.

Uses the reframed MLP's calibrated predictions (predictions.parquet from
train_mlp_reframed.py). At each observed trade in the test bucket, checks
whether our strategy's entry condition fires. Fixed $100 stake per trigger.

Strategies evaluated:
  1. Our MLP — general +EV       (edge > 0.02)
  2. Our MLP — home-run          (edge > 0.20, time_to_settlement < 6h, price < 0.30)
  3. Naive market (always enter) — baseline showing the market's own edge
  4. Random entry at matched frequency — null hypothesis
  5. Follow-the-whales            (enter when usd_amount > $10k, same direction)
  6. Logreg signal                (if logreg predictions available)

Also runs a cutoff-date sweep (N days before deadline) for the home-run
strategy, plotting PnL vs N.

PnL model: each trigger is a fresh $100 bet on trade correctness, priced at
market_implied_prob.
  correct   → profit = $100 × (1 − p) / p
  incorrect → loss   = $100

Outputs:
  data/backtest_outputs/
    strategy_metrics.csv            per-strategy total PnL, hit rate, Sharpe, max DD
    trades_per_strategy.parquet     per-strategy trade log
    cutoff_sweep.csv                PnL vs N for home-run strategy
    pnl_curves.png                  cumulative PnL curve per strategy
    cutoff_sweep.png                PnL vs N plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAKE = 100.0
MARGIN_GENERAL = 0.02
MARGIN_HOMERUN = 0.20
HOMERUN_TTS_SECONDS = 6 * 3600
HOMERUN_MAX_PRICE = 0.30


def load_test_trades(preds_path: Path) -> pd.DataFrame:
    """Load test-bucket predictions + attach columns needed for filters."""
    preds = pd.read_parquet(preds_path)
    test = (
        preds[preds["split"] == "test"].reset_index(drop=True).sort_values("timestamp")
    )

    # Attach time_to_settlement and price from labeled dataset
    labeled = pd.read_parquet(
        ROOT / "data" / "iran_strike_labeled.parquet",
        columns=[
            "wallet",
            "condition_id",
            "timestamp",
            "time_to_settlement_s",
            "price",
            "usd_amount",
            "is_buy",
            "is_token1",
        ],
    )
    out = test.merge(labeled, on=["wallet", "condition_id", "timestamp"], how="left")
    # Dedupe any row duplication from same-second same-wallet trades
    out = out.drop_duplicates(
        subset=["wallet", "condition_id", "timestamp", "usd_amount"], keep="first"
    )
    print(f"test trades: {len(out):,}  (from {len(test):,} preds)")
    return out.reset_index(drop=True)


def pnl_vector(
    bet_correct: np.ndarray, market_implied_prob: np.ndarray, stake: float = STAKE
) -> np.ndarray:
    """Per-trade PnL given correctness + market-implied price of correctness."""
    p = np.clip(market_implied_prob, 1e-6, 1 - 1e-6)
    win = stake * (1 - p) / p
    loss = -stake
    return np.where(bet_correct == 1, win, loss)


def strategy_metrics(name: str, triggers: pd.DataFrame, stake: float = STAKE) -> dict:
    if len(triggers) == 0:
        return {
            "strategy": name,
            "n_triggers": 0,
            "total_pnl": 0.0,
            "avg_pnl_per_trigger": 0.0,
            "hit_rate": np.nan,
            "sharpe": np.nan,
            "max_drawdown": 0.0,
            "capital_at_risk": 0.0,
        }
    pnl = pnl_vector(
        triggers["bet_correct"].values.astype(int),
        triggers["market_implied_prob"].values,
    )
    total = pnl.sum()
    cum = pnl.cumsum()
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    max_dd = drawdown.max() if len(drawdown) else 0
    # Sharpe: mean / std of per-trade PnL, annualised treating each trigger as one observation
    sharpe = (
        (pnl.mean() / pnl.std(ddof=0)) * np.sqrt(len(pnl))
        if pnl.std(ddof=0) > 0
        else np.nan
    )
    return {
        "strategy": name,
        "n_triggers": int(len(triggers)),
        "total_pnl": round(float(total), 2),
        "avg_pnl_per_trigger": round(float(pnl.mean()), 2),
        "hit_rate": round(float(triggers["bet_correct"].mean()), 4),
        "sharpe": round(float(sharpe), 3) if not np.isnan(sharpe) else None,
        "max_drawdown": round(float(max_dd), 2),
        "capital_at_risk": round(float(len(triggers) * stake), 2),
        "return_pct": round(float(100 * total / (len(triggers) * stake)), 2),
    }


def run_strategies(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run all strategies, return (metrics df, dict of cumulative PnL curves)."""
    results: list[dict] = []
    curves: dict[str, np.ndarray] = {}

    def _record(name: str, triggers: pd.DataFrame) -> None:
        m = strategy_metrics(name, triggers)
        results.append(m)
        print(f"  {name}: {m}")
        if len(triggers) > 0:
            pnl = pnl_vector(
                triggers["bet_correct"].values.astype(int),
                triggers["market_implied_prob"].values,
            )
            curves[name] = pnl.cumsum()

    # 1. MLP general +EV
    general = df[df["gap_correct"] > MARGIN_GENERAL]
    _record("MLP_general_EV", general)

    # 2. MLP home-run
    homerun = df[
        (df["gap_correct"] > MARGIN_HOMERUN)
        & (df["time_to_settlement_s"] < HOMERUN_TTS_SECONDS)
        & (df["market_implied_prob"] < HOMERUN_MAX_PRICE)
    ]
    _record("MLP_homerun", homerun)

    # 3. Naive: enter every trade (market-belief baseline)
    _record("naive_enter_all", df)

    # 4. Random entry at matched frequency (vs general)
    rng = np.random.default_rng(42)
    n_random = len(general)
    random_idx = (
        rng.choice(len(df), size=n_random, replace=False)
        if n_random > 0
        else np.array([], dtype=int)
    )
    _record("random_match_general", df.iloc[random_idx])

    # 5. Follow-the-whales (usd_amount > $10k, take same side as observed taker)
    whales = df[df["usd_amount"] > 10_000]
    _record("follow_whales", whales)

    return pd.DataFrame(results), curves


def cutoff_sweep(df: pd.DataFrame, Ns: list[int]) -> pd.DataFrame:
    """Home-run PnL vs N-days-before-deadline."""
    rows = []
    for N in Ns:
        tts_max = N * 86400
        sub = df[
            (df["gap_correct"] > MARGIN_HOMERUN)
            & (df["time_to_settlement_s"] < tts_max)
            & (df["market_implied_prob"] < HOMERUN_MAX_PRICE)
        ]
        m = strategy_metrics(f"homerun_N={N}d", sub)
        rows.append({"N_days": N, **m})
    return pd.DataFrame(rows)


def plot_pnl_curves(curves: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    palette = {
        "MLP_general_EV": "#4C72B0",
        "MLP_homerun": "#7B2CBF",
        "naive_enter_all": "#999999",
        "random_match_general": "#DD8452",
        "follow_whales": "#55A868",
    }
    for name, curve in curves.items():
        color = palette.get(name, "#333")
        ax.plot(
            np.arange(len(curve)),
            curve,
            label=f"{name} (n={len(curve)})",
            color=color,
            lw=1.2,
        )
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("Trigger # (sorted chronologically)")
    ax.set_ylabel("Cumulative PnL ($, fixed $100 stake)")
    ax.set_title("Cumulative PnL per strategy — test bucket (Feb 21-28)")
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


def plot_cutoff_sweep(sweep: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax1.plot(
        sweep["N_days"],
        sweep["total_pnl"],
        marker="o",
        color="#7B2CBF",
        lw=2,
        label="Total PnL ($)",
    )
    ax1.set_xlabel("N — days-before-deadline cutoff")
    ax1.set_ylabel("Total PnL ($)", color="#7B2CBF")
    ax1.tick_params(axis="y", labelcolor="#7B2CBF")
    ax1.invert_xaxis()  # right side = closer to deadline
    ax2 = ax1.twinx()
    ax2.plot(
        sweep["N_days"],
        sweep["n_triggers"],
        marker="s",
        color="#55A868",
        lw=1.5,
        alpha=0.7,
        label="# triggers",
    )
    ax2.set_ylabel("# triggers", color="#55A868")
    ax2.tick_params(axis="y", labelcolor="#55A868")
    ax1.set_title("Home-run PnL vs cutoff-days-before-deadline")
    ax1.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preds",
        default=str(ROOT / "data" / "mlp_reframed_outputs" / "predictions.parquet"),
    )
    ap.add_argument("--out", default=str(ROOT / "data" / "backtest_outputs"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_test_trades(Path(args.preds))

    print("\n=== strategies ===")
    metrics, curves = run_strategies(df)
    metrics.to_csv(out_dir / "strategy_metrics.csv", index=False)

    print("\n=== cutoff-date sweep (home-run) ===")
    sweep = cutoff_sweep(df, [14, 7, 3, 1])
    sweep.to_csv(out_dir / "cutoff_sweep.csv", index=False)

    print("\n=== plots ===")
    plot_pnl_curves(curves, out_dir / "pnl_curves.png")
    plot_cutoff_sweep(sweep, out_dir / "cutoff_sweep.png")

    print(f"\nall outputs in {out_dir}/")
    print("\nfinal strategy ranking by return %:")
    print(metrics.sort_values("return_pct", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
