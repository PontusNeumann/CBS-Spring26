"""
05_backtest.py — Realistic capital-aware backtest + naive baseline + overview chart.

Source-of-truth merge of:
  alex/v5_final_ml_pipeline/scripts/_common.py            (cost/edge helpers)
  alex/v5_final_ml_pipeline/scripts/10_backtest.py        (vector backtest)
  alex/v5_final_ml_pipeline/scripts/11_realistic_backtest.py (capital-aware sim)
  alex/v5_final_ml_pipeline/scripts/12_sensitivity_sweep.py (param grid)
  alex/v5_final_ml_pipeline/scripts/13_naive_baseline_backtest.py (falsification)
  alex/v5_final_ml_pipeline/scripts/14_overview_chart.py  (headline figure)
  alex/v5_final_ml_pipeline/scripts/06_phase2_falsification.py (consensus check)

What it does:
  1. Load each model's calibrated test predictions (saved by 04_calibration.py).
  2. For ten betting strategies, simulate trading with a $10K bankroll, 5% max bet,
     20% concentration cap, gas + slippage, no copycats.
  3. Run the same simulation for a naive baseline (bet aligned with the market favorite)
     as a falsification control — does the model add value over a free heuristic?
  4. Output the headline figure (overview.png) used in the report.
  5. Sweep capital, max-bet, and copycat scenarios for a sensitivity table.

Run:
  python 05_backtest.py

Outputs:
  outputs/backtest/sensitivity.csv         per (model, strategy, scenario): n, ROI, drawdown
  outputs/backtest/overview.png            headline heatmap (used in report main body)
  outputs/backtest/falsification.json      naive-vs-model verdict per strategy
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, OUTPUTS_DIR, RANDOM_SEED  # noqa: E402

TARGET = "bet_correct"

# what: realism constants (cost floor caps payoff at 19x; gas + slippage applied per trade)
COST_FLOOR = 0.05
GAS_COST_USD = 0.50
SLIPPAGE_THRESHOLD = 0.25
SLIPPAGE_FACTOR = 0.05

# what: real-world events that bracket the test cohort; bets resolve at the event
STRIKE_EVENT_UTC = pd.Timestamp("2026-02-28T06:35:00", tz="UTC").timestamp()
CEASEFIRE_EVENT_UTC = pd.Timestamp("2026-04-07T23:59:59", tz="UTC").timestamp()


# ----------------------------------------------------------------------------
# Cost + edge math
# ----------------------------------------------------------------------------

def compute_cost_and_edge(test: pd.DataFrame, p_hat: np.ndarray, cost_floor: float = COST_FLOOR):
    """Per-trade cost paid for the winning side + model edge (p_hat - cost)."""
    # what: every trade has a buy/sell side and a yes/no outcome; cost is the price for the winning side
    # how: BUY YES at p_yes -> cost = p_yes; BUY NO at (1-p_yes) -> cost = 1 - p_yes; etc
    # why: edge = model's view of P(win) minus market's implied price = expected profit per $1 staked
    p_yes = test["pre_yes_price_corrected"].values if "pre_yes_price_corrected" in test.columns \
        else test["pre_trade_price"].values
    side_buy = test["side_buy"].values
    outcome_yes = test["outcome_yes"].values
    trader_side_wins_yes = side_buy * outcome_yes + (1 - side_buy) * (1 - outcome_yes)
    cost = np.where(trader_side_wins_yes == 1, p_yes, 1 - p_yes)
    cost = np.clip(cost, cost_floor, 1.0 - cost_floor)
    edge = p_hat - cost
    return cost, edge


# ----------------------------------------------------------------------------
# Strategy masks
# ----------------------------------------------------------------------------

def strategy_masks(p_hat: np.ndarray, edge: np.ndarray, cost: np.ndarray,
                    time_to_deadline: np.ndarray) -> dict[str, np.ndarray]:
    """Return a dict {strategy_name: boolean mask of which trades to take}."""
    n = len(p_hat)

    def top_k(scores: np.ndarray, k_pct: float) -> np.ndarray:
        # what: boolean mask selecting the top k_pct by score
        k = max(1, int(n * k_pct))
        m = np.zeros(n, dtype=bool)
        m[np.argsort(scores)[-k:]] = True
        return m

    return {
        # what: high-confidence picks (model says >= 99% sure to win)
        "phat_gt_0.99":   p_hat >= 0.99,
        "phat_gt_0.95":   p_hat >= 0.95,
        "phat_gt_0.9":    p_hat >= 0.90,
        # what: top-1% by phat (compares across models even when phat distributions differ)
        "top1pct_phat":   top_k(p_hat, 0.01),
        # what: top-K% by edge (model's predicted alpha)
        "top1pct_edge":   top_k(edge, 0.01),
        "top5pct_edge":   top_k(edge, 0.05),
        # what: general EV rule — bet whenever edge > 2 cents
        "general_ev":     edge > 0.02,
        # what: late + edge filter — focuses on insider-style opportunities near deadline
        "general_ev_late": (edge > 0.02) & (time_to_deadline < 6 * 3600),
        # what: cheap + edge — bets where the market underprices an underdog
        "general_ev_cheap": (edge > 0.02) & (cost < 0.30),
        # what: high-edge late-cheap "home run" combo (asymmetric-info hypothesis)
        "home_run":       (edge > 0.20) & (cost < 0.30) & (time_to_deadline < 6 * 3600),
    }


# ----------------------------------------------------------------------------
# Capital-aware execution loop
# ----------------------------------------------------------------------------

def realistic_backtest(signal_mask: np.ndarray, cost: np.ndarray, bet_correct: np.ndarray,
                        timestamps: np.ndarray, market_ids: np.ndarray, usd_amount: np.ndarray,
                        market_res_times: dict, *,
                        initial_capital: float = 10_000, max_bet_usd: float = 100.0,
                        max_bet_pct_capital: float = 0.05, max_bet_pct_volume: float = 0.10,
                        max_concentration_pct: float = 0.20, gas_cost: float = GAS_COST_USD,
                        slippage_threshold: float = SLIPPAGE_THRESHOLD,
                        slippage_factor: float = SLIPPAGE_FACTOR,
                        liquidity_scaler: float = 1.0) -> dict:
    """Chronologically execute every signal subject to capital, concentration, gas, slippage."""
    # what: select only the signal trades and order them by timestamp
    sig_idx = np.where(signal_mask)[0]
    if len(sig_idx) == 0:
        return {"n_signals": 0, "n_executed": 0, "final_capital": initial_capital,
                "roi": 0.0, "max_drawdown": 0.0}
    order = sig_idx[np.argsort(timestamps[sig_idx])]

    # what: runtime state
    capital = initial_capital
    open_positions: dict[str, float] = {}     # market_id -> $ tied up
    pending: dict[str, list] = {}             # market_id -> [(resolve_ts, return_amt, entry_bet)]
    max_capital, min_capital = capital, capital
    n_executed, skipped = 0, 0

    def release_resolved(now_ts: int) -> None:
        # what: any market that has now resolved returns its winnings/zeroes to capital
        nonlocal capital, max_capital, min_capital
        for mid in list(pending.keys()):
            still = []
            for res_ts, ret_amt, entry_bet in pending[mid]:
                if res_ts <= now_ts:
                    capital += ret_amt
                    open_positions[mid] = open_positions.get(mid, 0) - entry_bet
                    if open_positions[mid] <= 0.01:
                        open_positions.pop(mid, None)
                else:
                    still.append((res_ts, ret_amt, entry_bet))
            if still:
                pending[mid] = still
            else:
                pending.pop(mid, None)
        max_capital = max(max_capital, capital)
        min_capital = min(min_capital, capital)

    for i in order:
        ts = int(timestamps[i])
        release_resolved(ts)

        # what: refuse to bet if broke
        if capital <= 1.0:
            skipped += 1
            continue

        # what: bet sizing combines bankroll-pct, absolute max, and a fraction of the trade's volume
        # why: bankroll-pct keeps drawdown bounded; vol-cap keeps us from being too large to fill
        effective_trade_usd = float(usd_amount[i]) * liquidity_scaler
        bet = min(max_bet_usd, max_bet_pct_capital * capital,
                  max_bet_pct_volume * effective_trade_usd)

        # what: concentration cap — never have more than X% of bankroll riding on one market
        mid = str(market_ids[i])
        in_market = open_positions.get(mid, 0)
        room = max_concentration_pct * capital - in_market
        if room < 1.0:
            skipped += 1
            continue
        bet = min(bet, room)
        if bet < 1.0:
            skipped += 1
            continue

        # what: slippage — large bets relative to trade size pay an effective cost penalty
        eff_cost = cost[i]
        if bet > slippage_threshold * effective_trade_usd:
            eff_cost = min(0.99, eff_cost * (1 + slippage_factor))

        # what: commit capital, schedule resolution
        capital -= bet
        open_positions[mid] = open_positions.get(mid, 0) + bet
        # what: gas debited up-front so it shows in min_capital tracking
        capital -= gas_cost

        if bet_correct[i]:
            # what: payoff = stake / cost (gross). On win we add it back at resolution time
            payoff = bet / eff_cost
            res_ts = market_res_times.get(mid, ts + 86400)
            pending.setdefault(mid, []).append((res_ts, payoff, bet))
        else:
            # what: loss — capital is gone, stake stays "tied up" until resolution to keep concentration accurate
            res_ts = market_res_times.get(mid, ts + 86400)
            pending.setdefault(mid, []).append((res_ts, 0.0, bet))

        n_executed += 1
        max_capital = max(max_capital, capital)
        min_capital = min(min_capital, capital)

    # what: settle anything still pending at the end of the test cohort
    release_resolved(int(max(timestamps[order])) + 86400 * 365)

    drawdown = (max_capital - min_capital) if max_capital > 0 else 0.0
    return {"n_signals": int(signal_mask.sum()), "n_executed": int(n_executed),
            "final_capital": float(capital), "roi": float(capital / initial_capital - 1),
            "max_drawdown": float(drawdown)}


# ----------------------------------------------------------------------------
# Naive baseline (falsification)
# ----------------------------------------------------------------------------

def naive_consensus_phat(test: pd.DataFrame) -> np.ndarray:
    """Naive baseline: 'predict the market favourite wins' with confidence = pre-trade YES price."""
    # what: a free heuristic that requires no model — just use the market's own price
    # why: every model must beat THIS to claim it adds value (Stage B1b falsification)
    p_yes = test["pre_yes_price_corrected"].values if "pre_yes_price_corrected" in test.columns \
        else test["pre_trade_price"].values
    side_buy = test["side_buy"].values
    outcome_yes = test["outcome_yes"].values
    trader_side_wins_yes = side_buy * outcome_yes + (1 - side_buy) * (1 - outcome_yes)
    # what: baseline prob of winning = market's price for the trader's chosen side
    return np.where(trader_side_wins_yes == 1, p_yes, 1 - p_yes)


# ----------------------------------------------------------------------------
# Overview heatmap (the report's headline figure)
# ----------------------------------------------------------------------------

def render_overview(df: pd.DataFrame, out_path: Path,
                     scenario: dict = {"initial_capital": 10000, "max_bet_pct": 0.05, "liquidity_scaler": 1.0}) -> None:
    """Build the heatmap of ROI per (strategy, model) for the headline scenario."""
    # what: filter to one scenario row per (model, strategy)
    sub = df.copy()
    for k, v in scenario.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        print("  no rows for headline scenario; skipping overview chart")
        return
    pivot = sub.pivot_table(index="strategy", columns="model", values="roi", aggfunc="first")
    counts = sub.pivot_table(index="strategy", columns="model", values="n_executed",
                              aggfunc="first").reindex_like(pivot).fillna(0).astype(int)
    # what: order rows/cols so report figure is consistent run-to-run
    strategy_order = ["phat_gt_0.99", "phat_gt_0.95", "phat_gt_0.9", "top1pct_phat",
                      "top1pct_edge", "top5pct_edge", "general_ev", "general_ev_late",
                      "general_ev_cheap", "home_run"]
    pivot = pivot.reindex(index=[s for s in strategy_order if s in pivot.index])
    counts = counts.reindex_like(pivot).fillna(0).astype(int)

    cmap = LinearSegmentedColormap.from_list("rg",
        [(0.0, "#7a1717"), (0.4, "#d96a6a"), (0.5, "#f0f0f0"),
         (0.6, "#7fcf86"), (1.0, "#15703a")])
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=0.30)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    img = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
    for i, _ in enumerate(pivot.index):
        for j, _ in enumerate(pivot.columns):
            val = pivot.values[i, j]
            n = counts.values[i, j]
            if pd.isna(val) or n == 0:
                ax.text(j, i, "—", ha="center", va="center", color="#999", fontsize=9)
                continue
            color = "white" if abs(val) > 0.30 else "#222"
            ax.text(j, i, f"{val * 100:+.0f}%", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
            ax.text(j, i + 0.32, f"n={n:,}", ha="center", va="center",
                    color=color, fontsize=8, alpha=0.8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, top=True, bottom=False)
    cbar = fig.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("ROI (return on $10K)", fontsize=9)
    fig.suptitle("Realistic backtest — does the model beat a free heuristic?",
                  fontsize=14, fontweight="bold", y=1.005)
    ax.set_title(f"$10K bankroll · 5% max bet · 20% concentration cap · gas + slippage · no copycats",
                 fontsize=10, color="#555", pad=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Stage 5 — Realistic backtest, naive baseline, overview chart")
    print("=" * 60)
    out_dir = OUTPUTS_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    # what: load test rows + auxiliary columns
    print("  loading test data and predictions ...")
    df = pd.read_parquet(DATA_DIR / "consolidated_modeling_data.parquet")
    test = df[df["split"] == "test"].reset_index(drop=True).copy()
    test["market_id"] = test["market_id"].astype(str)

    # what: helpers — time-to-deadline per cohort, market resolution times
    test_res_ts = float(CEASEFIRE_EVENT_UTC)
    test["time_to_deadline"] = test_res_ts - test["timestamp"].astype(float)
    market_res_times = {mid: int(test_res_ts) for mid in test["market_id"].unique()}

    # what: gather calibrated predictions written by 04_calibration.py
    models_dir = OUTPUTS_DIR / "models"
    model_preds = {}
    for d in sorted(models_dir.iterdir()):
        cal_path = d / "preds_test_cal.npz"
        if cal_path.exists():
            model_preds[d.name] = np.load(cal_path)["cal"]
    # what: also include the naive consensus baseline as an extra "model"
    model_preds["naive_consensus"] = naive_consensus_phat(test)
    print(f"  models in this run: {list(model_preds)}")

    # what: parameter grid for the sensitivity sweep
    capitals = [1_000, 10_000, 100_000]
    bet_pcts = [0.01, 0.05, 0.10]
    liquidity_scalers = [1.0, 0.10]   # 1.0 = no copycats; 0.10 = 10x copycats sharing fill

    rows: list[dict] = []
    bet_correct = test[TARGET].astype(int).values
    timestamps = test["timestamp"].astype(float).values
    market_ids = test["market_id"].values
    usd_amount = test["usd_amount"].values if "usd_amount" in test.columns \
        else np.full(len(test), 100.0)
    time_to_deadline = test["time_to_deadline"].values

    # what: outer loop = model, inner = strategy x scenario grid
    for model_name, p_hat in model_preds.items():
        cost, edge = compute_cost_and_edge(test, p_hat)
        masks = strategy_masks(p_hat, edge, cost, time_to_deadline)
        for strat_name, mask in masks.items():
            for capital in capitals:
                for bet_pct in bet_pcts:
                    for ls in liquidity_scalers:
                        out = realistic_backtest(mask, cost, bet_correct, timestamps,
                                                 market_ids, usd_amount, market_res_times,
                                                 initial_capital=capital,
                                                 max_bet_pct_capital=bet_pct,
                                                 liquidity_scaler=ls)
                        rows.append({"model": model_name, "strategy": strat_name,
                                     "initial_capital": capital, "max_bet_pct": bet_pct,
                                     "liquidity_scaler": ls, **out})
            print(f"  {model_name} / {strat_name}: signals={int(mask.sum()):,}")

    sens = pd.DataFrame(rows)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    print(f"  wrote {len(sens)} sensitivity rows -> sensitivity.csv")

    # what: headline overview chart for the report main body
    render_overview(sens, out_dir / "overview.png")
    print(f"  wrote overview.png")

    # what: falsification check — for each strategy in the headline scenario, does the best model beat naive?
    headline = sens[(sens["initial_capital"] == 10_000) & (sens["max_bet_pct"] == 0.05) &
                    (sens["liquidity_scaler"] == 1.0)]
    falsification: dict = {}
    for strat in headline["strategy"].unique():
        sub = headline[headline["strategy"] == strat]
        naive_roi = float(sub[sub["model"] == "naive_consensus"]["roi"].iloc[0]) \
            if (sub["model"] == "naive_consensus").any() else None
        if naive_roi is None:
            continue
        ml = sub[sub["model"] != "naive_consensus"].sort_values("roi", ascending=False)
        if ml.empty:
            continue
        best = ml.iloc[0]
        falsification[strat] = {"naive_roi": naive_roi, "best_model": best["model"],
                                 "best_model_roi": float(best["roi"]),
                                 "ml_beats_naive": bool(best["roi"] > naive_roi)}
    (out_dir / "falsification.json").write_text(json.dumps(falsification, indent=2))
    print("\nFalsification (does the best ML model beat the naive market-favourite baseline?):")
    for s, r in falsification.items():
        verdict = "yes" if r["ml_beats_naive"] else "no"
        print(f"  {s:20s}  naive={r['naive_roi']*100:+.1f}%  "
              f"best_ml={r['best_model']}={r['best_model_roi']*100:+.1f}%  -> {verdict}")

    print(f"\nStage 5 complete. Outputs in {out_dir.relative_to(OUTPUTS_DIR.parent)}.")
    print("Proceed to 06_tuning_optuna.py.")
    return 0


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    sys.exit(main())
