"""
11_realistic_backtest.py

Capital-constrained backtest with position limits, gas costs, slippage approx.
Answers "if you actually traded this with $X starting bankroll, what would you have?"

Loads cached predictions from alex/.scratch/backtest/preds_*.npz (written by
_backtest_worker.py during 10_backtest.py).

Constraints modeled:
  1. Bankroll: start with initial_capital. If broke, stop.
  2. Per-trade bet size: min(max_bet_usd, max_bet_pct_capital × current_capital,
                             max_bet_pct_volume × original_trade_usd)
  3. Concentration: max max_concentration_pct × current_capital tied up in any
     one market at a time. Positions release when market resolves.
  4. Gas: gas_cost USD per executed trade (Polygon mainnet typical).
  5. Slippage approximation: if bet > slippage_threshold × original_trade_usd,
     increase effective cost by slippage_factor.

Outputs:
  alex/outputs/v5/backtest/realistic/
    summary.json       — all scenarios
    capital_curves.png — equity curves
    sensitivity.csv    — final capital across the full grid
    progress.html      — live self-refreshing viewer (open during run)

Sensitivity grid:
  initial_capital     ∈ {1_000, 10_000, 100_000}
  max_bet_pct_capital ∈ {0.01, 0.05, 0.10}
  liquidity_scaler    ∈ {1.0 (no copycats, default), 0.10 (10× copycat stress)}
  → 18 scenarios per model × strategy
  → 5 models × 10 strategies × 18 = 900 cells
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    COST_FLOOR_DEFAULT as COST_FLOOR,
    DATA,
    LIQUIDITY_SCALER_DEFAULT as LIQUIDITY_SCALER,
    ROOT,
    SCRATCH as SCRATCH_BASE,
    compute_cost_and_edge,
    compute_pre_yes_price_corrected,
    general_ev_rule,
    home_run_rule,
    market_resolution_time,
    top_k_mask,
)

warnings.filterwarnings("ignore")

SCRATCH = SCRATCH_BASE / "backtest"
OUT = ROOT / "outputs" / "v5" / "backtest" / "realistic"
OUT.mkdir(parents=True, exist_ok=True)

GAS_COST = 0.50  # USD per executed trade
SLIPPAGE_THRESHOLD = 0.25  # if bet > 25% of original trade USD, apply slippage
SLIPPAGE_FACTOR = 0.05  # add 5% to effective cost


# ---------------------------------------------------------------------------
# Live progress viewer — writes self-refreshing HTML on each cell completion.
# Open alex/outputs/backtest/realistic/progress.html in a browser to watch.
# ---------------------------------------------------------------------------


def _render_progress_html(state: dict) -> str:
    rows = state.get("recent_results", [])
    headline = state.get("headline_pivot", [])

    def fmt_row_html(r):
        roi = r.get("roi", 0)
        col = "#1a7f1a" if roi > 0 else "#b03030" if roi < 0 else "#666"
        return (
            f"<tr><td>{r['model']}</td><td>{r['strategy']}</td>"
            f"<td>${r['initial_capital']:,}</td><td>{int(r['max_bet_pct'] * 100)}%</td>"
            f"<td>{r['liquidity_scaler']}</td><td style='text-align:right'>{r['n_executed']:,}</td>"
            f"<td style='text-align:right'>${r['final_capital']:,.0f}</td>"
            f"<td style='text-align:right;color:{col};font-weight:600'>{roi * 100:+.1f}%</td></tr>"
        )

    headline_html = ""
    if headline:
        headline_html = "<h2>Headline — $10K, 5% bet, no copycats (live)</h2><table>"
        headline_html += "<tr><th>strategy</th>"
        models = sorted({r["model"] for r in headline})
        for m in models:
            headline_html += f"<th>{m}</th>"
        headline_html += "</tr>"
        strategies = sorted({r["strategy"] for r in headline})
        by = {(r["model"], r["strategy"]): r for r in headline}
        for s in strategies:
            headline_html += f"<tr><td>{s}</td>"
            for m in models:
                r = by.get((m, s))
                if r is None:
                    headline_html += "<td>—</td>"
                else:
                    roi = r["roi"]
                    col = "#1a7f1a" if roi > 0 else "#b03030" if roi < 0 else "#666"
                    headline_html += (
                        f"<td style='text-align:right;color:{col};font-weight:600'>"
                        f"{roi * 100:+.1f}%</td>"
                    )
            headline_html += "</tr>"
        headline_html += "</table>"

    pct = state.get("pct", 0)
    eta_min = state.get("eta_seconds", 0) / 60
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>Realistic backtest — live</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #222; }}
  h1, h2 {{ font-weight: 600; }}
  .bar {{ background: #eee; border-radius: 8px; height: 22px; overflow: hidden; margin: 12px 0; }}
  .fill {{ background: linear-gradient(90deg, #2196f3, #00bcd4); height: 100%; transition: width 0.5s; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #fafafa; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #666; }}
  .meta {{ color: #666; font-size: 13px; }}
  .stat {{ display: inline-block; margin-right: 24px; }}
  .stat strong {{ font-size: 20px; display: block; color: #222; }}
  .stat span {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h1>Realistic backtest — live progress</h1>
<div class="bar"><div class="fill" style="width: {pct:.1f}%"></div></div>
<div>
  <div class="stat"><strong>{state.get("completed", 0)} / {state.get("total_cells", 0)}</strong><span>cells done ({pct:.1f}%)</span></div>
  <div class="stat"><strong>{eta_min:.1f} min</strong><span>ETA remaining</span></div>
  <div class="stat"><strong>{state.get("elapsed_min", 0):.1f} min</strong><span>elapsed</span></div>
  <div class="stat"><strong>{state.get("current_model", "")}</strong><span>current model</span></div>
</div>
<p class="meta">Last cell: {state.get("last_cell", "—")} &nbsp;·&nbsp; Auto-refreshes every 5s</p>
{headline_html}
<h2>Recent cells (last 30)</h2>
<table>
<tr><th>model</th><th>strategy</th><th>capital</th><th>bet%</th><th>ls</th><th>n exec</th><th>final $</th><th>ROI</th></tr>
{"".join(fmt_row_html(r) for r in rows[-30:][::-1])}
</table>
</body></html>"""


def _emit_progress(state_path: Path, html_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, default=str))
    html_path.write_text(_render_progress_html(state))


# v4 contract — fail fast if pointed at v3.5 parquets or pre-Stage-1 schema.
TEST_PARQUET = "test_features_v4.parquet"
EXPECTED_N_FEATURES = 64  # cleaned: 80 - 16 cohort-flip features dropped per D-042

# Realism parameters live in _common: COST_FLOOR=0.05, LIQUIDITY_SCALER=0.10.
# Override here only if you need a non-default realism scenario.


# ---------------------------------------------------------------------------
# Realistic backtest
# ---------------------------------------------------------------------------


def realistic_backtest(
    signal_mask: np.ndarray,
    cost: np.ndarray,
    bet_correct: np.ndarray,
    timestamps: np.ndarray,  # epoch seconds
    market_ids: np.ndarray,
    usd_amount: np.ndarray,
    market_res_times: dict,
    initial_capital: float = 10_000,
    max_bet_usd: float = 100.0,
    max_bet_pct_capital: float = 0.05,
    max_bet_pct_volume: float = 0.10,
    max_concentration_pct: float = 0.20,
    gas_cost: float = GAS_COST,
    slippage_threshold: float = SLIPPAGE_THRESHOLD,
    slippage_factor: float = SLIPPAGE_FACTOR,
    liquidity_scaler: float = LIQUIDITY_SCALER,
):
    """Chronological capital-aware execution."""
    # Order signals chronologically
    sig_idx = np.where(signal_mask)[0]
    if len(sig_idx) == 0:
        return {
            "n_signals": 0,
            "n_executed": 0,
            "skipped_capital": 0,
            "skipped_concentration": 0,
            "initial_capital": initial_capital,
            "final_capital": initial_capital,
            "max_capital": initial_capital,
            "min_capital": initial_capital,
            "max_drawdown": 0.0,
            "roi": 0.0,
            "total_pnl": 0.0,
            "capital_curve": [(0, initial_capital)],
            "trades_executed": [],
        }
    order = sig_idx[np.argsort(timestamps[sig_idx])]

    capital = initial_capital
    open_positions: dict[str, float] = {}  # market_id -> total $ committed (entry-side)
    # market_id -> list of (resolution_ts, return_amount, entry_bet)
    # entry_bet is the original commitment, decremented from open_positions on release
    # so concentration math reflects "$ tied up", not "$ returned" (the latter is 0 on loss).
    open_resolutions: dict[str, list] = {}
    capital_curve = [(int(timestamps[order[0]]) - 1, capital)]
    skipped_capital = 0
    skipped_concentration = 0
    skipped_size = 0
    n_executed = 0
    trades_log = []

    def release_resolved(now_ts):
        nonlocal capital
        for mid in list(open_resolutions.keys()):
            still_pending = []
            for res_ts, return_amt, entry_bet in open_resolutions[mid]:
                if res_ts <= now_ts:
                    capital += return_amt
                    open_positions[mid] = open_positions.get(mid, 0) - entry_bet
                    if open_positions[mid] <= 0.01:
                        open_positions.pop(mid, None)
                else:
                    still_pending.append((res_ts, return_amt, entry_bet))
            if still_pending:
                open_resolutions[mid] = still_pending
            else:
                open_resolutions.pop(mid, None)

    for i in order:
        ts = int(timestamps[i])
        release_resolved(ts)

        if capital <= 1.0:
            skipped_capital += 1
            continue

        # Multiply original trade USD by liquidity_scaler to account for copycats
        # competing for the same fill. liquidity_scaler=0.10 = 10x copycats sharing.
        effective_trade_usd = float(usd_amount[i]) * liquidity_scaler
        bet = min(
            max_bet_usd,
            max_bet_pct_capital * capital,
            max_bet_pct_volume * effective_trade_usd,
        )

        # Concentration
        mid = str(market_ids[i])
        current_in_market = open_positions.get(mid, 0)
        max_in_market = max_concentration_pct * capital
        if current_in_market + bet > max_in_market:
            available = max_in_market - current_in_market
            if available < 1:
                skipped_concentration += 1
                continue
            bet = available

        if bet < 1:
            skipped_size += 1
            continue
        if capital - bet < 0:
            skipped_capital += 1
            continue

        # Slippage — also gated by effective (copycat-shared) trade USD
        effective_cost = cost[i]
        if bet > slippage_threshold * effective_trade_usd:
            effective_cost = min(0.99, effective_cost * (1 + slippage_factor))

        # Execute: capital is committed
        capital -= bet
        # Resolution time
        res_ts = market_res_times.get(mid, ts + 86400 * 7)  # fallback: 1 week
        if bet_correct[i]:
            payoff = bet / effective_cost  # gross payout (incl. stake)
            return_amt = payoff - gas_cost
            pnl = bet * (1 - effective_cost) / effective_cost - gas_cost
        else:
            pnl = -bet - gas_cost
            # Apply gas debit immediately so we don't double-count later
            capital -= gas_cost
            # No future return — the bet is gone, capital won't be repaid on resolution
            return_amt = 0.0

        # Track open position for concentration math. entry_bet is the original
        # commitment; release_resolved subtracts it from open_positions[mid] so
        # losing bets free their concentration slot when the market resolves.
        open_positions[mid] = current_in_market + bet
        open_resolutions.setdefault(mid, []).append((res_ts, return_amt, bet))

        n_executed += 1
        trades_log.append(
            {
                "timestamp": ts,
                "market_id": mid,
                "bet": float(bet),
                "cost": float(effective_cost),
                "bet_correct": int(bet_correct[i]),
                "pnl": float(pnl),
            }
        )
        capital_curve.append((ts, capital + sum(open_positions.values())))

    # Release any remaining positions at end of test
    final_ts = int(timestamps.max())
    release_resolved(final_ts + 1)

    final_capital = capital + sum(open_positions.values())
    capital_curve.append((final_ts + 1, final_capital))

    cap_values = np.array([v for _, v in capital_curve])
    # Running max drawdown: at each point, drop from peak so far
    running_peak = np.maximum.accumulate(cap_values)
    drawdowns = running_peak - cap_values
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    max_dd_pct = (
        float((drawdowns / np.maximum(running_peak, 1)).max())
        if len(drawdowns) > 0
        else 0.0
    )
    return {
        "n_signals": int(len(sig_idx)),
        "n_executed": int(n_executed),
        "skipped_capital": int(skipped_capital),
        "skipped_concentration": int(skipped_concentration),
        "skipped_size": int(skipped_size),
        "initial_capital": float(initial_capital),
        "final_capital": float(final_capital),
        "max_capital": float(cap_values.max()),
        "min_capital": float(cap_values.min()),
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "roi": float((final_capital - initial_capital) / initial_capital),
        "total_pnl": float(final_capital - initial_capital),
        "capital_curve": capital_curve,
        "trades_executed": trades_log,
    }


# Strategy masks (general_ev_rule, home_run_rule, top_k_mask) live in _common.

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("realistic backtest — capital-aware execution")
    print("=" * 60)

    # --- v4 data guard ------------------------------------------------------
    test_path = DATA / TEST_PARQUET
    if not test_path.exists():
        raise SystemExit(
            f"v4 parquet missing: {test_path}. Pontus has not delivered, or "
            f"Stage 0 pre-flight was skipped. Run 01_validate_schema.py first."
        )
    fcols = json.loads((DATA / "feature_cols.json").read_text())
    if len(fcols) != EXPECTED_N_FEATURES:
        raise SystemExit(
            f"feature_cols.json has {len(fcols)} features, expected "
            f"{EXPECTED_N_FEATURES}. Run 01_validate_schema.py to update it."
        )

    test = pd.read_parquet(test_path)
    test_raw = pd.read_parquet(DATA / "test.parquet")
    markets = pd.read_parquet(DATA / "markets_subset.parquet")
    res_times = market_resolution_time(markets)

    # Re-attach original USD amount per trade. test_raw and test have same row count
    # but different sort orders. Sort both by (market_id, timestamp) to align.
    test_raw["market_id"] = test_raw["market_id"].astype(str)
    test["market_id"] = test["market_id"].astype(str)
    test_raw = test_raw.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    test = test.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    assert len(test) == len(test_raw), (
        f"row count mismatch {len(test)} vs {len(test_raw)}"
    )
    test["usd_amount"] = test_raw["usd_amount"].values

    # Compute corrected pre_yes_price (fixes the per-token-price bug). The values
    # returned are already aligned to test_raw sorted by (market_id, timestamp),
    # which matches `test`'s order after the sort above.
    test["pre_yes_price_corrected"] = compute_pre_yes_price_corrected(test_raw)
    print(
        f"[fix] corrected pre_yes_price computed (mean {test['pre_yes_price_corrected'].mean():.3f}, "
        f"vs old pre_trade_price mean {test['pre_trade_price'].mean():.3f})"
    )

    # Reload predictions in the same sort order — use the timestamp+market_id pair
    # to map cached preds (saved in original test_features order) back to sorted test.
    # Simpler: re-read test_features and apply same sort, then attach preds by index.
    test_orig = pd.read_parquet(test_path)
    test_orig["market_id"] = test_orig["market_id"].astype(str)
    test_orig["_orig_idx"] = np.arange(len(test_orig))
    sort_key = test_orig.sort_values(["market_id", "timestamp"])["_orig_idx"].values
    # sort_key[i] = original row index that ended up at position i after sorting

    bet_correct = test["bet_correct"].astype(int).values
    timestamps = test["timestamp"].values
    market_ids = test["market_id"].values
    usd_amount = test["usd_amount"].values
    n_test = len(test)
    print(f"test trades: {n_test:,}")
    print(
        f"markets: {len(markets)} ({sum(1 for v in res_times.values() if v < 1e15)} with resolution times)"
    )

    # Time-to-deadline (seconds)
    time_to_deadline_sec = (
        np.exp(test["log_time_to_deadline_hours"].values) - 1
    ) * 3600

    # Load cached predictions. Predictions are saved in the original test_features
    # row order (before sort). Reorder them via sort_key to align with the sorted
    # `test` DataFrame.
    models = [
        "logreg_l2",
        "random_forest",
        "hist_gbm",
        "lightgbm",
        "xgboost",
        "mlp_sklearn",
    ]
    model_data = {}
    for m in models:
        path = SCRATCH / f"preds_{m}.npz"
        if not path.exists():
            print(f"  ✗ {m}: no preds at {path} — skipping")
            continue
        d = np.load(path)
        cal_orig = d["cal"]
        if len(cal_orig) != n_test:
            raise SystemExit(
                f"[{m}] worker preds length {len(cal_orig)} != n_test {n_test}. "
                f"Sort-key reorder would silently misalign predictions. "
                f"Re-run with RETRAIN=1 (worker may have read a stale parquet)."
            )
        # Reorder: sort_key[i] = original index at sorted position i
        cal = cal_orig[sort_key]
        cost, edge, _ = compute_cost_and_edge(test, cal)
        model_data[m] = {"p_hat": cal, "cost": cost, "edge": edge}
        print(f"  ✓ {m}: loaded {len(cal)} preds (mean p_hat {cal.mean():.3f})")

    if not model_data:
        print("No predictions found — run 10_backtest.py first")
        return

    # ---- Strategy registry --------------------------------------------------
    def make_strategies(p_hat, edge, cost):
        return {
            "general_ev": general_ev_rule(edge),
            "home_run": home_run_rule(edge, cost, time_to_deadline_sec),
            "top1pct_phat": top_k_mask(p_hat, 0.01),
            "top1pct_edge": top_k_mask(edge, 0.01),
            "top5pct_edge": top_k_mask(edge, 0.05),
            "phat_gt_0.9": p_hat > 0.9,
            "phat_gt_0.95": p_hat > 0.95,
            "phat_gt_0.99": p_hat > 0.99,
            "general_ev_cheap": general_ev_rule(edge) & (cost < 0.30),
            "general_ev_late": general_ev_rule(edge) & (time_to_deadline_sec < 86400),
        }

    # ---- Sensitivity grid ---------------------------------------------------
    # Trimmed: $1M tier dropped (max_bet_usd=$100 caps deployment → ROI ≈ 0).
    capital_grid = [1_000, 10_000, 100_000]
    bet_pct_grid = [0.01, 0.05, 0.10]
    # 1.0 = default (no copycats); 0.10 = stress test with 10× copycats sharing.
    fill_share_grid = [1.0, 0.10]

    summary = {"models": {}}
    sensitivity_rows = []
    curves_for_plot = {}

    # Live progress viewer setup
    import time as _time

    state_path = OUT / "_progress.json"
    html_path = OUT / "progress.html"
    # Count strategies using the first model's real arrays (home_run_rule reads
    # time_to_deadline_sec which is full-length).
    _first_data = next(iter(model_data.values()))
    n_strategies_per_model = len(
        make_strategies(_first_data["p_hat"], _first_data["edge"], _first_data["cost"])
    )
    total_cells = (
        len(model_data)
        * n_strategies_per_model
        * len(capital_grid)
        * len(bet_pct_grid)
        * len(fill_share_grid)
    )
    _t0 = _time.time()
    _completed = 0

    print(f"[live] watch progress at file://{html_path}")

    for m, data in model_data.items():
        strategies = make_strategies(data["p_hat"], data["edge"], data["cost"])
        model_results = {}
        for s_name, mask in strategies.items():
            scenarios = {}
            for cap in capital_grid:
                for bet_pct in bet_pct_grid:
                    for ls in fill_share_grid:
                        r = realistic_backtest(
                            signal_mask=mask,
                            cost=data["cost"],
                            bet_correct=bet_correct,
                            timestamps=timestamps,
                            market_ids=market_ids,
                            usd_amount=usd_amount,
                            market_res_times=res_times,
                            initial_capital=cap,
                            max_bet_pct_capital=bet_pct,
                            liquidity_scaler=ls,
                        )
                        ls_label = "no_fill_share" if ls >= 1.0 else f"ls{ls:g}"
                        scenarios[f"cap{cap}_bet{int(bet_pct * 100)}pct_{ls_label}"] = {
                            k: v
                            for k, v in r.items()
                            if k not in ("capital_curve", "trades_executed")
                        }
                        row = {
                            "model": m,
                            "strategy": s_name,
                            "initial_capital": cap,
                            "max_bet_pct": bet_pct,
                            "liquidity_scaler": ls,
                            "n_signals": r["n_signals"],
                            "n_executed": r["n_executed"],
                            "final_capital": r["final_capital"],
                            "roi": r["roi"],
                            "max_drawdown": r["max_drawdown"],
                            "skipped_capital": r["skipped_capital"],
                            "skipped_concentration": r["skipped_concentration"],
                        }
                        sensitivity_rows.append(row)
                        if (
                            cap == 10_000
                            and bet_pct == 0.05
                            and ls >= 1.0
                            and s_name in ("general_ev", "top1pct_edge", "home_run")
                        ):
                            curves_for_plot[f"{m}_{s_name}"] = r["capital_curve"]

                        _completed += 1
                        elapsed = _time.time() - _t0
                        rate = _completed / max(elapsed, 1e-6)
                        eta = (total_cells - _completed) / rate if rate > 0 else 0
                        # Headline pivot = $10K, 5% bet, ls=1.0
                        headline = [
                            x
                            for x in sensitivity_rows
                            if x["initial_capital"] == 10_000
                            and x["max_bet_pct"] == 0.05
                            and x["liquidity_scaler"] == 1.0
                        ]
                        _emit_progress(
                            state_path,
                            html_path,
                            {
                                "completed": _completed,
                                "total_cells": total_cells,
                                "pct": 100 * _completed / total_cells,
                                "elapsed_min": elapsed / 60,
                                "eta_seconds": eta,
                                "current_model": m,
                                "last_cell": (
                                    f"{m} / {s_name} / cap=${cap:,} / bet={int(bet_pct * 100)}% / ls={ls}"
                                ),
                                "recent_results": sensitivity_rows,
                                "headline_pivot": headline,
                            },
                        )
            model_results[s_name] = scenarios
        summary["models"][m] = model_results
        print(
            f"  ✓ {m}: backtested {len(strategies)} strategies × "
            f"{len(capital_grid) * len(bet_pct_grid) * len(fill_share_grid)} scenarios"
        )

    # ---- Save ----------------------------------------------------------------
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df.to_csv(OUT / "sensitivity.csv", index=False)

    # Print headline tables: $10K capital, 5% bet, both fill-share scenarios
    for ls_val, label in [(1.0, "default no copycats"), (0.10, "10× copycats stress")]:
        print("\n" + "=" * 80)
        print(
            f"HEADLINE — $10K starting capital, 5% max bet per trade — {label} (ls={ls_val})"
        )
        print("=" * 80)
        headline = sens_df[
            (sens_df.initial_capital == 10_000)
            & (sens_df.max_bet_pct == 0.05)
            & (sens_df.liquidity_scaler == ls_val)
        ]
        headline_pivot = headline.pivot_table(
            index="strategy",
            columns="model",
            values=["final_capital", "roi", "n_executed"],
            aggfunc="first",
        )
        print(headline_pivot.round(2).to_string())

    # Plot capital curves
    if curves_for_plot:
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, curve in curves_for_plot.items():
            ts = [pd.to_datetime(t, unit="s") for t, _ in curve]
            v = [v for _, v in curve]
            ax.plot(ts, v, label=label, lw=1.0)
        ax.axhline(10_000, color="black", ls="--", lw=0.5, label="initial $10K")
        ax.set_xlabel("trade timestamp")
        ax.set_ylabel("capital ($)")
        ax.set_title("Capital curve — $10K start, 5% max bet, all constraints active")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUT / "capital_curves.png", dpi=140)
        plt.close(fig)
        print(f"\n  -> {OUT / 'capital_curves.png'}")

    print(f"\nDONE — outputs in {OUT}")


if __name__ == "__main__":
    main()
