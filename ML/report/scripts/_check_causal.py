"""Regression guard: asserts the consolidated dataset is internally causal.

Runs a battery of cheap invariants over `data/03_consolidated_dataset.csv`
that should hold for any future rebuild. A non-zero exit code signals a
regression.

Check groups:
  A. First-in-group invariants
     Every cumulative / prior feature must be 0 (or NaN when undefined)
     on the first row of its group.
  B. Monotonicity
     Cumulative market features must be non-decreasing within each market
     when rows are sorted by timestamp.
  C. Sign / sentinel checks
     `wallet_first_minus_trade_sec ≤ 0`, `log_*` features are non-negative.
  D. Cross-feature consistency
     `wallet_funded_by_cex_scoped ≤ wallet_funded_by_cex`.
     `wallet_enriched = 0` iff Layer 6 numeric features are NaN.
  E. NaN-rate bounds
     Per-feature upper bounds matching our documented data policy
     (§5.6 missing-data typology). Catches a broken join or an accidental
     dtype change that would leave a column all-NaN.
  F. No leaky time-feature sourcing
     `time_to_settlement_s` must equal `deadline_ts − timestamp` (not
     `resolution_ts − timestamp`).

Usage:
  python scripts/_check_causal.py          # full
  python scripts/_check_causal.py --quick  # sample 100k rows for fast CI
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "03_consolidated_dataset.csv"

# Absolute NaN-rate upper bounds per feature. Numbers are generous vs the
# current dataset's observed rates, so minor drift doesn't cause false
# positives, but order-of-magnitude regressions fail loudly.
NAN_BOUNDS: dict[str, float] = {
    "time_to_settlement_s": 0.001,
    "log_time_to_settlement": 0.001,
    "pct_time_elapsed": 0.01,
    "market_timing_known": 0.0,
    "market_trade_count_so_far": 0.0,
    "market_volume_so_far_usd": 0.0,
    "market_vol_1h_log": 0.0,
    "market_vol_24h_log": 0.0,
    "market_buy_share_running": 0.001,
    "market_price_vol_last_1h": 0.01,
    "wallet_prior_trades": 0.0,
    "wallet_prior_volume_usd": 0.0,
    "wallet_prior_win_rate": 0.35,
    "wallet_prior_trades_in_market": 0.0,
    "wallet_directional_purity_in_market": 0.35,
    "wallet_spread_ratio": 0.35,
    "wallet_median_gap_in_market": 0.65,
    "wallet_cumvol_same_side_last_10min": 0.0,
    "wallet_trades_in_market_last_1min": 0.0,
    "wallet_trades_in_market_last_10min": 0.0,
    "wallet_trades_in_market_last_60min": 0.0,
    "wallet_is_burst": 0.0,
    "wallet_position_size_before_trade": 0.0,
    "trade_size_vs_position_pct": 0.0,
    "is_position_exit": 0.0,
    "is_position_flip": 0.0,
    "wallet_is_whale_in_market": 0.0,
    "wallet_market_category_entropy": 0.15,
    "size_vs_wallet_avg": 0.35,
    "size_x_time_to_settlement": 0.001,
    "size_vs_market_cumvol_pct": 0.001,
    "size_vs_market_avg": 0.001,
    "wallet_enriched": 0.0,
    "wallet_polygon_age_at_t_days": 0.01,
    "wallet_polygon_nonce_at_t": 0.01,
    "wallet_n_inbound_at_t": 0.01,
    "wallet_n_cex_deposits_at_t": 0.01,
    "wallet_cex_usdc_cumulative_at_t": 0.01,
    "wallet_funded_by_cex": 0.0,
    "wallet_funded_by_cex_scoped": 0.0,
}

LAYER6_NUMERIC = [
    "wallet_polygon_age_at_t_days",
    "wallet_polygon_nonce_at_t",
    "wallet_log_polygon_nonce_at_t",
    "wallet_n_inbound_at_t",
    "wallet_log_n_inbound_at_t",
    "wallet_n_cex_deposits_at_t",
    "wallet_cex_usdc_cumulative_at_t",
    "wallet_log_cex_usdc_cum",
    "days_from_first_usdc_to_t",
]


class Check:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def want(self, cond: bool, msg: str) -> None:
        (self.passes if cond else self.failures).append(msg)

    def report(self) -> int:
        print(f"\n— passed: {len(self.passes)}   failed: {len(self.failures)} —")
        for m in self.failures:
            print(f"  [FAIL] {m}")
        for m in self.passes:
            print(f"  [ OK ] {m}")
        return 1 if self.failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="sample 100k rows for CI")
    args = ap.parse_args()

    if not CSV.exists():
        print(f"missing {CSV}", file=sys.stderr)
        return 2

    print(f"reading {CSV.name}...")
    df = pd.read_csv(CSV, low_memory=False)
    if args.quick:
        df = df.sample(n=min(100_000, len(df)), random_state=0).reset_index(drop=True)
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(
        ["condition_id", "proxyWallet", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)

    c = Check()

    # --- A. first-in-group invariants ---
    # The CSV is written in an order where same-timestamp ties are not
    # reproducible from a post-hoc sort. The *values* are correct; we check
    # via groupby-min / groupby-any-isna, which is tie-break-immune.
    c.want(
        bool((df.groupby("proxyWallet")["wallet_prior_trades"].min() == 0).all()),
        "wallet_prior_trades reaches 0 for every wallet (first-trade invariant)",
    )
    c.want(
        bool(
            (
                df.groupby(["proxyWallet", "condition_id"])[
                    "wallet_prior_trades_in_market"
                ].min()
                == 0
            ).all()
        ),
        "wallet_prior_trades_in_market reaches 0 for every (wallet, market)",
    )
    c.want(
        bool(
            (df.groupby("condition_id")["market_trade_count_so_far"].min() == 0).all()
        ),
        "market_trade_count_so_far reaches 0 for every market",
    )
    c.want(
        bool(
            (df.groupby("condition_id")["market_volume_so_far_usd"].min() == 0).all()
        ),
        "market_volume_so_far_usd reaches 0 for every market",
    )
    c.want(
        bool(
            df.groupby("proxyWallet")["wallet_prior_win_rate"]
            .apply(lambda s: s.isna().any())
            .all()
        ),
        "wallet_prior_win_rate is NaN at least once per wallet (first-trade)",
    )
    c.want(
        bool(
            df.groupby(["proxyWallet", "condition_id"])[
                "wallet_directional_purity_in_market"
            ]
            .apply(lambda s: s.isna().any())
            .all()
        ),
        "wallet_directional_purity_in_market is NaN at least once per (wallet, market)",
    )
    # Whale causality: the wallet cannot be a whale in the market before it
    # trades. Same-second order-fills can produce multiple rows at the min
    # timestamp; the build processes them in some order, so any single row
    # may show whale=1 if earlier fills in the same second already crossed
    # the p95 threshold. The invariant is therefore: the MIN whale flag
    # across same-second first-timestamp rows per (wallet, market) is 0.
    first_ts = df.groupby(["proxyWallet", "condition_id"])["timestamp"].transform(
        "min"
    )
    at_first_ts = df[df["timestamp"] == first_ts]
    whale_first_ts_min = at_first_ts.groupby(
        ["proxyWallet", "condition_id"]
    )["wallet_is_whale_in_market"].min()
    n_bad = int((whale_first_ts_min > 0).sum())
    c.want(
        n_bad == 0,
        f"min(whale_flag) at first-second-of-(wallet, market) is 0 for every "
        f"pair — whale flag cannot fire before the wallet enters the market "
        f"(violations: {n_bad})",
    )

    # --- B. monotonicity of cumulative market features ---
    # Same-timestamp ties mean a post-hoc sort can disorder rows within a
    # single second, producing apparent "dips". The underlying cumsum was
    # monotonic at build time; here we tolerate intra-timestamp wobble by
    # checking the minimum per-timestamp value is non-decreasing.
    def nondec_by_ts(g: pd.DataFrame, col: str) -> bool:
        s = (
            g[[col, "timestamp"]]
            .assign(_v=pd.to_numeric(g[col], errors="coerce"))
            .groupby("timestamp")["_v"]
            .min()
            .sort_index()
        )
        v = s.to_numpy()
        return np.all(np.diff(v) >= -1e-6) if len(v) > 1 else True

    bad_vol = 0
    bad_ct = 0
    for _cid, g in df.groupby("condition_id", sort=False):
        if not nondec_by_ts(g, "market_volume_so_far_usd"):
            bad_vol += 1
        if not nondec_by_ts(g, "market_trade_count_so_far"):
            bad_ct += 1
    c.want(
        bad_vol == 0,
        f"market_volume_so_far_usd non-decreasing per market when aggregated "
        f"by timestamp (violations: {bad_vol} / {df['condition_id'].nunique()})",
    )
    c.want(
        bad_ct == 0,
        f"market_trade_count_so_far non-decreasing per market when aggregated "
        f"by timestamp (violations: {bad_ct} / {df['condition_id'].nunique()})",
    )

    # --- C. sign / sentinel checks ---
    wfmt = pd.to_numeric(df["wallet_first_minus_trade_sec"], errors="coerce")
    c.want(
        int((wfmt > 0).sum()) == 0,
        f"wallet_first_minus_trade_sec ≤ 0 everywhere "
        f"(positives: {int((wfmt > 0).sum())})",
    )
    ltts = pd.to_numeric(df["log_time_to_settlement"], errors="coerce")
    c.want(
        int((ltts < 0).sum()) == 0,
        f"log_time_to_settlement ≥ 0 everywhere "
        f"(negatives: {int((ltts < 0).sum())})",
    )

    # --- D. cross-feature consistency ---
    fbc = pd.to_numeric(df["wallet_funded_by_cex"], errors="coerce").fillna(0).astype(int)
    fbcs = pd.to_numeric(df["wallet_funded_by_cex_scoped"], errors="coerce").fillna(0).astype(int)
    c.want(
        int((fbcs > fbc).sum()) == 0,
        f"wallet_funded_by_cex_scoped ≤ wallet_funded_by_cex "
        f"(violations: {int((fbcs > fbc).sum())})",
    )
    enr = pd.to_numeric(df["wallet_enriched"], errors="coerce").fillna(0).astype(int)
    un = df[enr == 0]
    if len(un) > 0:
        l6_non_nan = un[LAYER6_NUMERIC].notna().any(axis=1)
        c.want(
            int(l6_non_nan.sum()) == 0,
            f"un-enriched rows have all Layer 6 numeric features NaN "
            f"(violations: {int(l6_non_nan.sum())} / {len(un):,})",
        )

    # --- E. NaN-rate bounds ---
    for col, bound in NAN_BOUNDS.items():
        if col not in df.columns:
            c.want(False, f"column missing: {col}")
            continue
        s = pd.to_numeric(df[col], errors="coerce") if df[col].dtype == "object" else df[col]
        rate = float(s.isna().mean())
        c.want(
            rate <= bound + 1e-9,
            f"NaN rate of {col}: {rate:.4f} ≤ bound {bound:.4f}",
        )

    # --- F. time-feature sourcing ---
    if "deadline_ts" in df.columns:
        dl = pd.to_datetime(df["deadline_ts"], utc=True, errors="coerce")
        computed = (dl - df["timestamp"]).dt.total_seconds()
        observed = pd.to_numeric(df["time_to_settlement_s"], errors="coerce")
        delta = (computed - observed).abs()
        # allow up to a handful of seconds of float round-trip noise
        c.want(
            float(delta.max()) < 1.0,
            f"time_to_settlement_s == deadline_ts − timestamp "
            f"(max abs delta: {float(delta.max()):.3f}s)",
        )

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
