"""
15_tuned_overview.py — generate the tuned-vs-baseline backtest comparison.

Runs the realistic backtest engine ONLY for the headline scenario
($10K bankroll, 5% bet, no copycats, ls=1.0) on:
  - 5 baseline models (default cached preds)
  - tuned RF (using preds_test_tuned.npz)
  - tuned MLP (using preds_test_tuned.npz)

Outputs:
  outputs/v5/backtest_tuned/sensitivity.csv
  outputs/v5/backtest_tuned/overview_tuned.png
  outputs/v5/backtest_tuned/tuning_summary.md

Tuned preds are loaded as both `raw` and `cal` (no isotonic refit). For top-K
strategies (top1pct_phat / top1pct_edge / top5pct_edge), this is fine because
they're rank-based. For absolute-threshold strategies (phat_gt_0.X) the
results are uncalibrated and flagged as such in the summary.
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA,
    ROOT,
    SCRATCH as SCRATCH_BASE,
    compute_cost_and_edge,
    compute_pre_yes_price_corrected,
    general_ev_rule,
    home_run_rule,
    market_resolution_time,
    top_k_mask,
)
from importlib import import_module as _imp  # noqa: E402

_rb11 = _imp("11_realistic_backtest")
realistic_backtest = _rb11.realistic_backtest

warnings.filterwarnings("ignore")

TUNED_RF = (
    ROOT
    / "outputs"
    / "v5"
    / "rigor"
    / "optuna"
    / "random_forest"
    / "preds_test_tuned.npz"
)
TUNED_MLP = (
    ROOT
    / "outputs"
    / "v5"
    / "rigor"
    / "optuna"
    / "mlp_sklearn"
    / "preds_test_tuned.npz"
)
DEFAULT_PREDS = SCRATCH_BASE / "backtest"
OUT = ROOT / "outputs" / "v5" / "backtest_tuned"
OUT.mkdir(parents=True, exist_ok=True)

TEST_PARQUET = "test_features_v4.parquet"
EXPECTED_N_FEATURES = 64


def _load_tuned(path: Path, sort_key: np.ndarray, n_test: int) -> dict:
    """Load tuned raw preds and use as both raw + cal (no isotonic refit)."""
    if not path.exists():
        return None
    d = np.load(path)
    raw = d["raw"]
    if len(raw) != n_test:
        raise SystemExit(f"tuned preds at {path}: length {len(raw)} != n_test {n_test}")
    raw_sorted = raw[sort_key]
    return {"raw": raw_sorted, "cal": raw_sorted}


def _load_default(model: str, sort_key: np.ndarray, n_test: int) -> dict:
    path = DEFAULT_PREDS / f"preds_{model}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {"raw": d["raw"][sort_key], "cal": d["cal"][sort_key]}


def main():
    print("=" * 60)
    print("15_tuned_overview: tuned-vs-baseline realistic backtest")
    print("=" * 60)

    test_path = DATA / TEST_PARQUET
    fcols = json.loads((DATA / "feature_cols.json").read_text())
    if len(fcols) != EXPECTED_N_FEATURES:
        raise SystemExit(f"feature count mismatch: {len(fcols)}")

    test = pd.read_parquet(test_path)
    test_raw = pd.read_parquet(DATA / "test.parquet")
    markets = pd.read_parquet(DATA / "markets_subset.parquet")
    res_times = market_resolution_time(markets)

    test_raw["market_id"] = test_raw["market_id"].astype(str)
    test["market_id"] = test["market_id"].astype(str)
    test_raw = test_raw.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    test_orig = test.copy()
    test_orig["_orig_idx"] = np.arange(len(test_orig))
    sort_key = test_orig.sort_values(["market_id", "timestamp"])["_orig_idx"].values
    test = test.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    assert len(test) == len(test_raw)
    test["usd_amount"] = test_raw["usd_amount"].values
    test["pre_yes_price_corrected"] = compute_pre_yes_price_corrected(test_raw)

    bet_correct = test["bet_correct"].astype(int).values
    timestamps = test["timestamp"].values
    market_ids = test["market_id"].values
    usd_amount = test["usd_amount"].values
    n_test = len(test)
    time_to_deadline_sec = (
        np.exp(test["log_time_to_deadline_hours"].values) - 1
    ) * 3600

    # Build labelled (model, preds) pairs: 5 baselines + tuned RF + tuned MLP
    candidates: list[tuple[str, dict]] = []
    for m in ["logreg_l2", "random_forest", "hist_gbm", "lightgbm", "mlp_sklearn"]:
        p = _load_default(m, sort_key, n_test)
        if p is not None:
            candidates.append((m, p))
    for label, path in [
        ("random_forest_tuned", TUNED_RF),
        ("mlp_sklearn_tuned", TUNED_MLP),
    ]:
        p = _load_tuned(path, sort_key, n_test)
        if p is not None:
            candidates.append((label, p))
    print(f"loaded models: {[n for n, _ in candidates]}")

    strategies = [
        "phat_gt_0.99",
        "phat_gt_0.95",
        "phat_gt_0.9",
        "top1pct_phat",
        "top1pct_edge",
        "top5pct_edge",
        "general_ev",
        "general_ev_late",
        "general_ev_cheap",
        "home_run",
    ]

    rows = []
    for label, preds in candidates:
        p_hat = preds["cal"]
        cost, edge, _ = compute_cost_and_edge(test, p_hat)
        masks = {
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
        for s_name in strategies:
            mask = masks[s_name]
            r = realistic_backtest(
                signal_mask=mask,
                cost=cost,
                bet_correct=bet_correct,
                timestamps=timestamps,
                market_ids=market_ids,
                usd_amount=usd_amount,
                market_res_times=res_times,
                initial_capital=10_000,
                max_bet_pct_capital=0.05,
                liquidity_scaler=1.0,
            )
            rows.append(
                {
                    "model": label,
                    "strategy": s_name,
                    "n_signals": r["n_signals"],
                    "n_executed": r["n_executed"],
                    "final_capital": r["final_capital"],
                    "roi": r["roi"],
                    "max_drawdown": r["max_drawdown"],
                }
            )
        print(f"  ✓ {label}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sensitivity.csv", index=False)
    print(f"  → {OUT / 'sensitivity.csv'}")

    # Overview chart with tuned columns
    model_order = [
        "logreg_l2",
        "random_forest",
        "random_forest_tuned",
        "hist_gbm",
        "lightgbm",
        "mlp_sklearn",
        "mlp_sklearn_tuned",
    ]
    model_order = [m for m in model_order if m in df.model.values]
    pivot_roi = df.pivot_table(
        index="strategy", columns="model", values="roi", aggfunc="first"
    ).reindex(strategies, columns=model_order)
    pivot_n = (
        df.pivot_table(
            index="strategy", columns="model", values="n_executed", aggfunc="first"
        )
        .reindex(strategies, columns=model_order)
        .fillna(0)
        .astype(int)
    )

    cmap = LinearSegmentedColormap.from_list(
        "rg",
        [
            (0.0, "#7a1717"),
            (0.4, "#d96a6a"),
            (0.5, "#f0f0f0"),
            (0.6, "#7fcf86"),
            (1.0, "#15703a"),
        ],
    )
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=0.30)

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    img = ax.imshow(pivot_roi.values, cmap=cmap, norm=norm, aspect="auto")
    for i, _ in enumerate(strategies):
        for j, m in enumerate(model_order):
            val = pivot_roi.values[i, j]
            n = pivot_n.values[i, j]
            if pd.isna(val) or n == 0:
                ax.text(j, i, "—", ha="center", va="center", color="#999", fontsize=10)
                continue
            color = "white" if abs(val) > 0.30 else "#222"
            ax.text(
                j,
                i,
                f"{val * 100:+.0f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )
            ax.text(
                j,
                i + 0.32,
                f"n={n:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                alpha=0.8,
            )

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, fontsize=10, rotation=15, ha="right")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=10)
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, top=True, bottom=False)
    for label in ax.get_xticklabels():
        if "_tuned" in label.get_text():
            label.set_fontweight("bold")
            label.set_color("#1565c0")

    cbar = fig.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("ROI on $10K", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Realistic backtest — tuned vs baseline (RF + MLP)",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    ax.set_title(
        "$10K · 5% bet · 20% concentration · gas + slippage · no copycats · 31-day cross-regime test\n"
        "Tuned columns use raw best-trial preds (no isotonic refit) — phat_gt_0.X cells are not calibration-aligned.",
        fontsize=9,
        color="#555",
        pad=14,
    )
    fig.text(
        0.5,
        -0.01,
        "Bold blue columns = tuned configs (Optuna). Other columns = baseline preds from main pipeline.",
        ha="center",
        fontsize=8,
        color="#666",
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "overview_tuned.png", dpi=160, bbox_inches="tight", facecolor="white"
    )
    print(f"  → {OUT / 'overview_tuned.png'}")

    # Tuning summary markdown
    summary_lines = ["# v5 tuning summary", "", "## Best configs"]
    for m in ["random_forest", "mlp_sklearn"]:
        comp_path = ROOT / "outputs" / "v5" / "rigor" / "optuna" / m / "comparison.json"
        if comp_path.exists():
            comp = json.loads(comp_path.read_text())
            summary_lines.append(f"### {m}")
            t = comp.get("tuned", {})
            d = comp.get("default", {})
            summary_lines.append(
                f"- Best OOF AUC: **{t.get('best_oof_auc', 'n/a'):.5f}**"
            )
            summary_lines.append(
                f"- Tuned test AUC: **{t.get('test_auc_raw', 'n/a'):.5f}**"
            )
            if "test_auc_raw" in d:
                summary_lines.append(f"- Default test AUC: {d['test_auc_raw']:.5f}")
                summary_lines.append(
                    f"- **Δ test AUC: {comp.get('delta_test_auc', 0):+.5f}**"
                )
            summary_lines.append(f"- Best params: `{t.get('best_params', {})}`")
            summary_lines.append("")

    summary_lines += [
        "## Realistic-backtest deltas ($10K, 5%, no copycats, headline scenario)",
        "",
        "| strategy | RF baseline | RF tuned | Δ | MLP baseline | MLP tuned | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in strategies:

        def fmt(model_label):
            cell = df[(df.model == model_label) & (df.strategy == s)]
            if cell.empty:
                return "—"
            r = cell.iloc[0]
            return f"{r['roi'] * 100:+.1f}%"

        rf_b, rf_t = fmt("random_forest"), fmt("random_forest_tuned")
        mlp_b, mlp_t = fmt("mlp_sklearn"), fmt("mlp_sklearn_tuned")

        def delta(b, t):
            try:
                return f"{(float(t.replace('%', '').replace('+', '')) - float(b.replace('%', '').replace('+', ''))):+.1f}pp"
            except Exception:
                return "—"

        summary_lines.append(
            f"| `{s}` | {rf_b} | {rf_t} | {delta(rf_b, rf_t)} | {mlp_b} | {mlp_t} | {delta(mlp_b, mlp_t)} |"
        )

    (OUT / "tuning_summary.md").write_text("\n".join(summary_lines) + "\n")
    print(f"  → {OUT / 'tuning_summary.md'}")
    print("DONE")


if __name__ == "__main__":
    main()
