"""
27_mlp_ci_and_random_entry.py

Two finalisation analyses for the v2 pipeline outputs that were missing
from the report submission:

1. **Wallet-level bootstrap 95 % CI on the MLP test ROC-AUC.**
   The v2 pipeline saved the bootstrap CI on the calibrated stacked-ensemble
   predictions (test ROC 0.540, CI [0.527, 0.556]). The reviewer flagged
   that reporting a CI on the worst-performing model rather than the
   primary model (MLP test ROC 0.579) looks defensive. We re-run the same
   wallet-resampling procedure on the MLP's `predictions_test.parquet` and
   write the result to `pontus/outputs/v2/bootstrap/roc_ci_mlp.json`.

2. **Random-entry head-to-head against the home-run trading rule.**
   §5.5.4 specifies a Bernoulli(0.5) random-prediction baseline evaluated
   over 1,000 seeded draws on the test cohort, but §6 / §8.1 never run it.
   We replace `p_hat` with `Uniform(0,1)` per row per draw, apply the same
   home-run gate (edge > 0.20, ttd < 6 h, mip < 0.30), and compute total
   flat-stake PnL across 1,000 seeds. Report the distribution and the
   one-sided proportion of random-entry seeds that beat the home-run rule's
   observed $617,873. Output: `pontus/outputs/v2/backtest/random_entry_vs_home_run.json`.

Run:
    python report/pontus/scripts/27_mlp_ci_and_random_entry.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "pontus" / "outputs" / "v2"
TEST_COHORT = DATA_DIR / "experiments" / "test.parquet"
MLP_PRED = OUT_DIR / "modelling" / "mlp" / "predictions_test.parquet"
HOME_RUN_REF = OUT_DIR / "backtest" / "clipped_home_run.json"
MIP_FLOOR = 0.05  # mirrors backtest_clipped in 22_v2_pipeline.py

# Constants (mirror 21_full_pipeline.py) ---------------------------------
TIME_COL = "timestamp"
DEADLINE_COL = "deadline_ts"
BENCHMARK = "market_implied_prob"
HOME_RUN_EDGE = 0.20
HOME_RUN_TTD_SEC = 6 * 3600
HOME_RUN_MIP_MAX = 0.30
STAKE_USD = 100.0

N_BOOT = 1000
N_RANDOM_DRAWS = 1000
SEED = 42


# -----------------------------------------------------------------------
# 1. MLP wallet-level bootstrap CI
# -----------------------------------------------------------------------
def wallet_bootstrap_roc(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    y = pred["bet_correct"].to_numpy().astype(np.int64)
    p = pred["p_hat"].to_numpy().astype(np.float64)
    wallets = pred["proxyWallet"].to_numpy()
    unique = np.unique(wallets)
    wallet_to_rows = {w: np.where(wallets == w)[0] for w in unique}

    base = float(roc_auc_score(y, p))
    samples = np.full(n_boot, np.nan, dtype=np.float64)
    for b in range(n_boot):
        choice = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([wallet_to_rows[w] for w in choice])
        try:
            samples[b] = float(roc_auc_score(y[idx], p[idx]))
        except ValueError:
            samples[b] = np.nan
    samples = samples[~np.isnan(samples)]
    return {
        "metric": "roc_auc",
        "model": "mlp",
        "point": base,
        "bootstrap_mean": float(samples.mean()),
        "ci95_lo": float(np.quantile(samples, 0.025)),
        "ci95_hi": float(np.quantile(samples, 0.975)),
        "n_boot_ok": int(samples.size),
        "n_wallets": int(unique.size),
        "n_rows": int(len(pred)),
    }


# -----------------------------------------------------------------------
# 2. Random-entry baseline over the home-run gate
# -----------------------------------------------------------------------
def _pnl_per_dollar(y: np.ndarray, mip: np.ndarray, follow: np.ndarray) -> np.ndarray:
    pnl = np.zeros_like(y, dtype=np.float64)
    if follow.any():
        mip_m = np.clip(mip[follow], 1e-4, 1 - 1e-4)
        pnl[follow] = y[follow] / mip_m - 1.0
    inv = ~follow
    if inv.any():
        mip_inv = np.clip(mip[inv], 1e-4, 1 - 1e-4)
        pnl[inv] = (1 - y[inv]) / (1 - mip_inv) - 1.0
    return pnl


def home_run_backtest(p_hat: np.ndarray, mip: np.ndarray, y: np.ndarray,
                      ttd: np.ndarray, stake_usd: float = STAKE_USD) -> dict:
    """Mirror of backtest_clipped(home_run): payoff-side mip clipped at 0.05."""
    mip_clip = np.clip(mip, MIP_FLOOR, 1 - MIP_FLOOR)
    edge = p_hat - mip  # gate uses raw mip, payoff uses clipped mip (matches v2 pipeline)
    gate_follow = edge > HOME_RUN_EDGE
    gate_inverse = edge < -HOME_RUN_EDGE
    gate = (gate_follow | gate_inverse) & (ttd < HOME_RUN_TTD_SEC) & (mip < HOME_RUN_MIP_MAX)
    follow = gate_follow & gate
    pnl_pd = _pnl_per_dollar(y, mip_clip, follow)
    trade_pnl = np.where(gate, stake_usd, 0.0) * pnl_pd
    n_trig = int(gate.sum())
    wins = int(((trade_pnl > 0) & gate).sum())
    total_pnl = float(trade_pnl.sum())
    return {
        "triggers": n_trig,
        "wins": wins,
        "hit_rate": float(wins / n_trig) if n_trig else 0.0,
        "total_pnl_usd": total_pnl,
    }


def random_entry_distribution(mip: np.ndarray, y: np.ndarray, ttd: np.ndarray,
                               n_draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(mip)
    pnls = np.empty(n_draws, dtype=np.float64)
    triggers = np.empty(n_draws, dtype=np.int64)
    hit_rates = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        p_rand = rng.random(n)
        bt = home_run_backtest(p_rand, mip, y, ttd)
        pnls[i] = bt["total_pnl_usd"]
        triggers[i] = bt["triggers"]
        hit_rates[i] = bt["hit_rate"]
    return {
        "n_draws": int(n_draws),
        "pnl_mean_usd": float(pnls.mean()),
        "pnl_std_usd": float(pnls.std(ddof=1)),
        "pnl_p025_usd": float(np.quantile(pnls, 0.025)),
        "pnl_p50_usd": float(np.quantile(pnls, 0.50)),
        "pnl_p975_usd": float(np.quantile(pnls, 0.975)),
        "pnl_max_usd": float(pnls.max()),
        "triggers_mean": float(triggers.mean()),
        "triggers_std": float(triggers.std(ddof=1)),
        "hit_rate_mean": float(hit_rates.mean()),
        "_pnls": pnls,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    print("[1/2] MLP wallet-level bootstrap 95% CI on test ROC-AUC")
    pred = pd.read_parquet(MLP_PRED)
    ci = wallet_bootstrap_roc(pred, n_boot=N_BOOT, seed=SEED)
    bs_dir = OUT_DIR / "bootstrap"
    bs_dir.mkdir(parents=True, exist_ok=True)
    with (bs_dir / "roc_ci_mlp.json").open("w") as f:
        json.dump(ci, f, indent=2)
    print(f"    point ROC = {ci['point']:.4f}  "
          f"95% CI [{ci['ci95_lo']:.4f}, {ci['ci95_hi']:.4f}]  "
          f"({ci['n_wallets']:,} wallets, {ci['n_boot_ok']} resamples)")

    print("[2/2] Random-entry head-to-head against home-run trading rule")
    te = pd.read_parquet(TEST_COHORT)
    deadline = pd.to_datetime(te[DEADLINE_COL], utc=True, errors="coerce")
    ts = pd.to_datetime(te[TIME_COL], utc=True, errors="coerce")
    ttd = (deadline - ts).dt.total_seconds().to_numpy()
    mip = te[BENCHMARK].to_numpy().astype(np.float64)
    y = te["bet_correct"].to_numpy().astype(np.int64)

    with HOME_RUN_REF.open() as f:
        observed = json.load(f)
    observed_pnl = float(observed["total_pnl_usd"])

    dist = random_entry_distribution(mip, y, ttd, n_draws=N_RANDOM_DRAWS, seed=SEED)
    pnls = dist.pop("_pnls")
    n_geq = int((pnls >= observed_pnl).sum())
    p_one_sided = (n_geq + 1) / (N_RANDOM_DRAWS + 1)

    summary = {
        "observed_home_run_pnl_usd": observed_pnl,
        "observed_home_run_triggers": int(observed["triggers"]),
        "observed_home_run_hit_rate": float(observed["hit_rate"]),
        "random_entry": dist,
        "random_seeds_beating_home_run": n_geq,
        "p_one_sided": p_one_sided,
        "stake_usd": STAKE_USD,
        "gate": {
            "edge_threshold": HOME_RUN_EDGE,
            "ttd_max_sec": HOME_RUN_TTD_SEC,
            "mip_max": HOME_RUN_MIP_MAX,
        },
    }
    bt_dir = OUT_DIR / "backtest"
    bt_dir.mkdir(parents=True, exist_ok=True)
    with (bt_dir / "random_entry_vs_home_run.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"    home-run observed PnL = ${observed_pnl:,.0f}")
    print(f"    random-entry PnL (n={N_RANDOM_DRAWS}): "
          f"mean ${dist['pnl_mean_usd']:,.0f}, "
          f"P50 ${dist['pnl_p50_usd']:,.0f}, "
          f"P97.5 ${dist['pnl_p975_usd']:,.0f}, "
          f"max ${dist['pnl_max_usd']:,.0f}")
    print(f"    seeds >= home-run: {n_geq}/{N_RANDOM_DRAWS}  "
          f"-> one-sided p = {p_one_sided:.4f}")


if __name__ == "__main__":
    main()
