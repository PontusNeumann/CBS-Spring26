"""Rebuild the report's design figures (fig1–fig4) from saved artefacts.

Produces:
  pontus/outputs/v2/_design/fig1_calibration.png
  pontus/outputs/v2/_design/fig2_permutation_importance.png
  pontus/outputs/v2/_design/fig3_per_market_roc.png
  pontus/outputs/v2/_design/fig4_pnl_equity.png   (new)

Improvements over the prior versions:
  * fig1: enlarged, Wilson 95% CIs, bin-count bar overlay (twin-y),
    annotated over/underconfidence regions, ECE in title.
  * fig2: top-12 (was 20), bars color-coded by feature group with legend.
    Easier to read; preserves the std-of-shuffle error bars.
  * fig3: untouched conceptually (already improved via 26_final_robustness.py),
    rebuilt here from `per_market_rocs.csv` so this script alone can
    regenerate the _design folder.
  * fig4 (new): cumulative PnL + drawdown + random-baseline overlay,
    addressing the "$617K reported with no visual proof" feedback.

Backups of the prior versions live in `_design/_pre_improvements/`.

Usage:
    python pontus/scripts/build_design_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "pontus" / "outputs" / "v2"
V2_FINAL = ROOT / "pontus" / "outputs" / "v2_final_robustness"
DESIGN = V2 / "_design"
DESIGN.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
})

PAL_10 = sns.color_palette("rocket_r", 10)
COL_DARK = "0.15"
COL_TRAIN = PAL_10[3]
COL_TEST = PAL_10[7]
COL_BAND = PAL_10[5]

# Feature-group palette for fig2. Mirrors the EDA taxonomy so the report
# can cross-reference it.
GROUP_COLORS = {
    "Trade-level": PAL_10[1],
    "Time-to-deadline": PAL_10[2],
    "Market-state rolling": PAL_10[4],
    "Price / vol microstructure": PAL_10[6],
    "Edge / payoff": PAL_10[7],
    "Wallet (HF taker / global)": PAL_10[8],
    "Wallet on-chain (Layer 6)": PAL_10[9],
    "Other": "0.55",
}

# Deterministic feature → group lookup for the strict-branch v2 model.
GROUP_RULES: list[tuple[str, list[str]]] = [
    ("Wallet on-chain (Layer 6)", [
        "wallet_polygon_age_at_t_days", "wallet_polygon_nonce_at_t",
        "wallet_log_polygon_nonce_at_t", "wallet_n_inbound_at_t",
        "wallet_log_n_inbound_at_t", "wallet_n_cex_deposits_at_t",
        "wallet_cex_usdc_cumulative_at_t", "wallet_log_cex_usdc_cum",
        "days_from_first_usdc_to_t", "wallet_funded_by_cex",
        "wallet_funded_by_cex_scoped",
    ]),
    ("Wallet (HF taker / global)", [
        "wallet_prior_win_rate_causal", "wallet_has_resolved_priors",
        "wallet_prior_trades", "wallet_prior_volume_usd",
        "wallet_prior_trades_in_market", "wallet_first_minus_trade_sec",
        "wallet_trades_in_market_last_1min", "wallet_trades_in_market_last_10min",
        "wallet_trades_in_market_last_60min", "wallet_is_burst",
        "wallet_is_whale_in_market", "wallet_spread_ratio",
        "wallet_median_gap_in_market", "wallet_market_category_entropy",
        "size_vs_wallet_avg",
    ]),
    ("Edge / payoff", [
        "consensus_strength", "contrarian_score", "is_long_shot_buy",
        "contrarian_strength", "log_payoff_if_correct", "risk_reward_ratio_pre",
        "trade_value_usd",
    ]),
    ("Time-to-deadline", [
        "log_time_to_deadline_hours", "pct_time_elapsed", "log_time_since_last_trade",
        "is_within_24h_of_deadline", "is_within_1h_of_deadline",
        "is_within_5min_of_deadline", "hour_of_day_sin", "day_of_week_sin",
        "day_of_week_cos",
    ]),
    ("Market-state rolling", [
        "log_n_trades_to_date", "market_buy_share_running",
        "log_recent_volume_5min", "log_recent_volume_1h", "log_recent_volume_24h",
        "log_trade_count_5min", "log_trade_count_1h", "log_trade_count_24h",
        "market_price_vol_last_5min", "market_price_vol_last_1h",
        "market_price_vol_last_24h", "order_flow_imbalance_5min",
        "order_flow_imbalance_1h", "order_flow_imbalance_24h",
        "yes_volume_share_recent_5min", "yes_volume_share_recent_1h",
        "yes_buy_pressure_5min", "token_side_skew_5min",
        "size_vs_market_cumvol_pct", "size_vs_market_avg",
    ]),
    ("Price / vol microstructure", [
        "pre_trade_price", "recent_price_mean_5min", "recent_price_mean_1h",
        "recent_price_mean_24h", "recent_price_high_1h", "recent_price_low_1h",
        "recent_price_range_1h", "pre_trade_price_change_5min",
        "pre_trade_price_change_1h", "pre_trade_price_change_24h",
        "implied_variance", "kyle_lambda_market_static", "realized_vol_1h",
        "jump_component_1h", "signed_oi_autocorr_1h", "distance_from_boundary",
    ]),
    ("Trade-level", [
        "log_size", "side_buy", "outcome_yes",
        "trade_size_to_recent_volume_ratio", "trade_size_vs_recent_avg",
        "avg_trade_size_recent_1h", "log_size_vs_taker_avg",
        "log_same_block_trade_count",
    ]),
]


def _group_for(feature: str) -> str:
    for group, members in GROUP_RULES:
        if feature in members:
            return group
    return "Other"


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson 95% CI for a binomial proportion. Stable at p=0/1."""
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=False)
    n = len(p)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        out += abs(p[m].mean() - y[m].mean()) * (m.sum() / n)
    return float(out)


# ---------------------------------------------------------------------------
# Fig 1 - Calibration / reliability with Wilson CIs and bin-count overlay
# ---------------------------------------------------------------------------
def fig1_calibration() -> None:
    pred_path = V2 / "modelling" / "stack_chosen" / "predictions_test.parquet"
    df = pd.read_parquet(pred_path, columns=["bet_correct", "p_hat"])
    p = df["p_hat"].to_numpy()
    y = df["bet_correct"].astype(int).to_numpy()

    n_bins = 15
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=False)

    centres, means, los, his, counts = [], [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rate = float(y[mask].mean())
        lo, hi = _wilson_ci(rate, n)
        centres.append((bins[b] + bins[b + 1]) / 2)
        means.append(rate)
        los.append(rate - lo)
        his.append(hi - rate)
        counts.append(n)

    ece = _ece(p, y, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Diagonal reference (perfect calibration)
    ax.plot([0, 1], [0, 1], color=COL_DARK, ls="--", lw=1.0, alpha=0.7,
            label="perfect calibration")

    # Reliability points with Wilson 95 percent CIs and a thin connecting
    # line. No region tints, no inline annotations - the over/underconfidence
    # narrative belongs in the report caption.
    ax.errorbar(
        centres, means, yerr=[los, his], fmt="o", color=PAL_10[7],
        ecolor=PAL_10[5], capsize=3, elinewidth=1.0, lw=0,
        markersize=6, label="model (Wilson 95 percent CI)",
    )
    ax.plot(centres, means, "-", color=PAL_10[7], lw=1.0, alpha=0.55)

    # Slight headroom both sides so a point at observed rate 0 or 1 stays
    # visible above the axis instead of getting clipped by the spine.
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title(
        f"Reliability curve: stacked ensemble, ceasefire test cohort\n"
        f"ECE = {ece:.3f}, n = {len(p):,} trades over {n_bins} bins",
        loc="center",
    )
    # Upper-left has empty whitespace; lower-right is occupied by the
    # rightmost-bin Wilson CI (very few trades at p_hat near 1, so the
    # interval is wide and crosses the corner).
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = DESIGN / "fig1_calibration.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}  (ECE={ece:.3f})")


# ---------------------------------------------------------------------------
# Fig 2 - Permutation importance, top-12, color-coded by feature group
# ---------------------------------------------------------------------------
def fig2_permutation_importance(top_k: int = 12) -> None:
    imp = pd.read_csv(V2 / "permutation_importance" / "val_roc_decay.csv")
    sub = imp.head(top_k).iloc[::-1]
    sub = sub.assign(group=[_group_for(f) for f in sub["feature"]])
    colors = [GROUP_COLORS[g] for g in sub["group"]]

    fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.42 * top_k)))
    ax.barh(
        sub["feature"], sub["mean_drop"], xerr=sub["std_drop"],
        color=colors, edgecolor="white", linewidth=0.6,
        error_kw=dict(ecolor=COL_DARK, capsize=2, lw=0.8),
    )
    ax.set_xlabel("mean ROC-AUC drop on validation when feature shuffled")
    ax.set_title(
        f"Permutation importance, top {top_k} features  ·  "
        f"baseline ROC = {sub['baseline_roc'].iloc[0]:.3f}"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Custom legend per group present in the top-k
    from matplotlib.patches import Patch
    groups_present = list(dict.fromkeys(sub["group"][::-1]))  # stable order
    legend_handles = [Patch(facecolor=GROUP_COLORS[g], edgecolor="white", label=g)
                      for g in groups_present]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout()
    out = DESIGN / "fig2_permutation_importance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Fig 3 - Per-market ROC histogram (rebuilt from artefact for portability)
# ---------------------------------------------------------------------------
def fig3_per_market_roc() -> None:
    pm = pd.read_csv(V2_FINAL / "per_market_temporal" / "per_market_rocs.csv")
    # Wider canvas so the legend sits to the right of the bars without
    # crowding the chance line or the two mean markers (which fall close
    # together near 0.71).
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.hist(
        pm["mlp_roc"], bins=20, alpha=0.55, color=PAL_10[7], label="MLP",
    )
    ax.hist(
        pm["logreg_roc"], bins=20, alpha=0.55, color=PAL_10[3], label="LogReg",
    )
    ax.axvline(0.5, color=COL_DARK, ls="--", lw=0.9, label="chance")
    # Bolder, slightly darker shades than the histogram fills so the means
    # read clearly without making the figure feel busy.
    ax.axvline(
        pm["mlp_roc"].mean(), color=PAL_10[8], ls=":", lw=2.0,
        label=f"MLP mean: {pm['mlp_roc'].mean():.3f}",
    )
    ax.axvline(
        pm["logreg_roc"].mean(), color=PAL_10[1], ls=":", lw=2.0,
        label=f"LogReg mean: {pm['logreg_roc'].mean():.3f}",
    )
    ax.set_xlabel("test ROC-AUC (last 15 percent of each market)")
    ax.set_ylabel("number of markets")
    ax.set_title(
        f"Per-market temporal ROC distribution (n={len(pm)} of 74 markets)"
    )
    # Legend placed outside the axes on the right so it does not overlap
    # the chance line or the mean labels.
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=False, fontsize=8.5,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = DESIGN / "fig3_per_market_roc.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Fig 4 - PnL equity curve + drawdown + random-baseline reference band.
# Uses the clipped-cost variants that match the paper's $617K headline.
# Kelly variants are excluded because their pnl_curve overflows under
# the bug-corrected cost model; they would dominate the y-axis without
# adding interpretive value.
# ---------------------------------------------------------------------------
def fig4_pnl_equity() -> None:
    clipped_home = json.loads((V2 / "backtest" / "clipped_home_run.json").read_text())
    clipped_general = json.loads((V2 / "backtest" / "clipped_general_ev.json").read_text())
    rand_path = V2 / "backtest" / "random_entry_vs_home_run.json"
    random_summary = json.loads(rand_path.read_text()) if rand_path.exists() else None

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]})
    ax_pnl, ax_dd = axes

    # mathtext interprets bare $ as math-mode delimiters; use \$ everywhere
    # the dollar sign appears inside a label or title.
    series = [
        (r"Home-run (primary, \$617k headline)",
         clipped_home["pnl_curve"], PAL_10[7], 1.8),
        ("General +EV (broader gate)",
         clipped_general["pnl_curve"], PAL_10[3], 1.2),
    ]
    max_x = max(len(curve) for _, curve, _, _ in series)
    for label, curve, colour, lw in series:
        x = np.arange(len(curve))
        ax_pnl.plot(x, np.asarray(curve) / 1000, color=colour, lw=lw,
                    label=fr"{label}, final \${curve[-1]/1000:,.0f}k")

    # Random-entry comparison: the 1,000-draw distribution describes
    # final PnL only; rendering it as a horizontal band that spans the
    # whole window makes random selection look constant in time, which
    # it is not. Render instead as a single 95-percent vertical interval
    # at the rightmost x position, where the comparison is valid.
    if random_summary and "random_entry" in random_summary:
        re = random_summary["random_entry"]
        lo_k = re["pnl_p025_usd"] / 1000
        hi_k = re["pnl_p975_usd"] / 1000
        mean_k = re["pnl_mean_usd"] / 1000
        x_anchor = max_x + max_x * 0.015
        ax_pnl.errorbar(
            x_anchor, mean_k,
            yerr=[[mean_k - lo_k], [hi_k - mean_k]],
            fmt="o", color="0.40", ecolor="0.40",
            elinewidth=1.6, capsize=4, capthick=1.4,
            markersize=5, zorder=3,
            label=(
                fr"Random-entry final ({re['n_draws']:,} draws): "
                fr"95 percent \${lo_k:,.0f}k-\${hi_k:,.0f}k, "
                fr"mean \${mean_k:,.0f}k"
            ),
        )

    # Curves should start flush with the y-axis (no left margin) and
    # leave just enough room on the right for the random-entry marker.
    ax_pnl.set_xlim(0, max_x * 1.05)
    ax_pnl.axhline(0, color=COL_DARK, ls="--", lw=0.7, alpha=0.5)
    ax_pnl.set_ylabel("Cumulative PnL ($k)")
    ax_pnl.set_title(
        "Cumulative PnL by strategy on the ceasefire test cohort\n"
        f"Triggers: home-run {clipped_home['triggers']:,}  ·  "
        f"general {clipped_general['triggers']:,}  ·  "
        f"home-run hit rate {clipped_home['hit_rate']:.1%}"
    )
    ax_pnl.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax_pnl.spines["top"].set_visible(False)
    ax_pnl.spines["right"].set_visible(False)

    # Drawdown of the headline strategy (clipped home-run)
    curve = np.asarray(clipped_home["pnl_curve"])
    rolling_max = np.maximum.accumulate(curve)
    drawdown = (curve - rolling_max) / 1000
    ax_dd.fill_between(np.arange(len(curve)), drawdown, 0, color=PAL_10[6], alpha=0.45)
    ax_dd.plot(np.arange(len(curve)), drawdown, color=PAL_10[7], lw=1.0)
    max_dd = drawdown.min()
    ax_dd.set_ylabel("Drawdown ($k)")
    ax_dd.set_xlabel("trigger index (chronological)")
    ax_dd.set_title(
        rf"Home-run drawdown, max \${-max_dd*1000:,.0f}", fontsize=10
    )
    ax_dd.spines["top"].set_visible(False)
    ax_dd.spines["right"].set_visible(False)

    fig.tight_layout()
    out = DESIGN / "fig4_pnl_equity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}  (final ${clipped_home['pnl_curve'][-1]/1000:,.0f}k, max DD ${-max_dd*1000:,.0f})")


def main() -> None:
    fig1_calibration()
    fig2_permutation_importance()
    fig3_per_market_roc()
    fig4_pnl_equity()


if __name__ == "__main__":
    main()
