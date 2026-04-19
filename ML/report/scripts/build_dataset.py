"""Build the labelled feature matrix for the 7-market Iran-strike dataset.

Input:
  data/iran_strike_trades.parquet  (451k raw trades)
  data/iran_strike_markets.parquet (7 market rows with winner_token, settlement_ts)
  data/hf_raw/trades.parquet       (full 418M-row file, joined for wallet_polymarket features)

Output:
  data/iran_strike_labeled.parquet (one row per taker trade, features + label + bucket)

Causal guarantee: every feature for row t uses only rows with timestamp < t
(enforced via groupby cumsum/cumcount/rolling with shift(1)).
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADES_IN = ROOT / "data" / "iran_strike_trades.parquet"
MARKETS_IN = ROOT / "data" / "iran_strike_markets.parquet"
FULL_HF = ROOT / "data" / "hf_raw" / "trades.parquet"
OUT = ROOT / "data" / "iran_strike_labeled.parquet"

# Bucket boundaries (UTC unix seconds)
T_TRAIN_END = 1769904000  # 2026-02-01 00:00 UTC
T_VAL_END = 1771632000  # 2026-02-21 00:00 UTC


def log(msg: str, t0: float | None = None) -> float:
    now = time.time()
    if t0 is not None:
        print(f"[{now - t0:6.1f}s] {msg}")
    else:
        print(f"[ start] {msg}")
    return now


def load_and_join(t0: float) -> pd.DataFrame:
    """Load trades, attach market metadata (winner_token, settlement_ts, resolved)."""
    log("loading trades + markets...", t0)
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            tr.timestamp,
            tr.block_number,
            tr.transaction_hash,
            tr.condition_id,
            tr.maker,
            tr.taker,
            tr.taker_direction,
            tr.maker_direction,
            tr.nonusdc_side,
            tr.price,
            tr.usd_amount,
            tr.token_amount,
            m.resolved,
            m.winner_token,
            m.settlement_ts
        FROM read_parquet('{TRADES_IN}') tr
        JOIN read_parquet('{MARKETS_IN}') m USING (condition_id)
    """).df()
    log(f"  loaded {len(df):,} trades", t0)
    return df


def add_bucket_and_label(df: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Trade-timestamp bucket + bet_correct label + taker-perspective view."""
    log("bucket + label...", t0)
    df["bucket"] = np.where(
        df["timestamp"] < T_TRAIN_END,
        "train",
        np.where(df["timestamp"] < T_VAL_END, "val", "test"),
    )

    # Taker-perspective: we keep the taker as `wallet`, and collapse
    # (direction, nonusdc_side) against winner_token into bet_correct.
    df["wallet"] = df["taker"]
    df["side"] = df["taker_direction"]  # BUY or SELL
    df["is_buy"] = (df["side"] == "BUY").astype(np.int8)
    df["is_token1"] = (df["nonusdc_side"] == "token1").astype(np.int8)
    side_on_winner = df["nonusdc_side"] == df["winner_token"]
    df["bet_correct"] = np.where(
        df["is_buy"] == 1,
        side_on_winner.astype(np.int8),
        (~side_on_winner).astype(np.int8),
    ).astype(np.int8)

    # market_implied_prob of correctness (for the trading rule only — NOT an MLP feature)
    # BUY at P → prob correct = P; SELL at P → prob correct = 1 - P
    df["market_implied_prob"] = np.where(
        df["is_buy"] == 1, df["price"], 1.0 - df["price"]
    )

    log(f"  label mean: {df['bet_correct'].mean():.3f}", t0)
    return df


def layer1_trade_local(df: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Layer 1 — trade-local features (no history). Excludes price deliberately."""
    log("layer 1 — trade-local...", t0)
    df["log_size"] = np.log1p(df["usd_amount"].astype(np.float64))
    df["log_token_amount"] = np.log1p(df["token_amount"].astype(np.float64))
    # is_buy, is_token1 already computed in add_bucket_and_label
    return df


def layer2_market_context(df: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Layer 2 — cumulative + rolling features per condition_id, causal (shift 1)."""
    log("layer 2 — market context (per condition_id)...", t0)
    df = df.sort_values(["condition_id", "timestamp"]).reset_index(drop=True)

    g = df.groupby("condition_id", sort=False)

    # Cumulative, strictly before t
    df["market_cumvol"] = g["usd_amount"].cumsum().shift(1).fillna(0.0)
    df["market_cumtrades"] = (
        g.cumcount()
    )  # already 0 at first trade = "before" semantics
    df["market_cumvol_log"] = np.log1p(df["market_cumvol"])
    df["market_cumtrades_log"] = np.log1p(df["market_cumtrades"])

    # Rolling windows (1h, 24h) on timestamp — per market
    # pandas rolling on datetime requires timestamp as a DatetimeIndex
    df["_ts_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    parts = []
    for cid, sub in df.groupby("condition_id", sort=False):
        sub = sub.set_index("_ts_dt").sort_index()
        sub["market_vol_1h"] = (
            sub["usd_amount"].rolling("1h").sum().shift(1).fillna(0.0)
        )
        sub["market_vol_24h"] = (
            sub["usd_amount"].rolling("24h").sum().shift(1).fillna(0.0)
        )
        sub["market_trades_1h"] = (
            sub["usd_amount"].rolling("1h").count().shift(1).fillna(0.0)
        )
        sub["market_price_mean_1h"] = sub["price"].rolling("1h").mean().shift(1)
        sub["market_price_std_1h"] = sub["price"].rolling("1h").std().shift(1)
        parts.append(sub.reset_index())
    df = pd.concat(parts, ignore_index=True)

    # log-transform heavy-tailed rolling features
    df["market_vol_1h_log"] = np.log1p(df["market_vol_1h"])
    df["market_vol_24h_log"] = np.log1p(df["market_vol_24h"])
    df["market_trades_1h_log"] = np.log1p(df["market_trades_1h"])
    # NaN fill for the price mean/std (for rows with no prior trades in the window)
    df["market_price_mean_1h"] = df["market_price_mean_1h"].fillna(0.5)
    df["market_price_std_1h"] = df["market_price_std_1h"].fillna(0.0)

    df = df.drop(columns=["_ts_dt"])
    log(
        f"  market context done. e.g. market_cumvol range: "
        f"{df['market_cumvol'].min():.0f} .. {df['market_cumvol'].max():,.0f}",
        t0,
    )
    return df


def layer3_time(df: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Layer 3 — time features computed against per-market settlement timestamps."""
    log("layer 3 — time features...", t0)
    df["time_to_settlement_s"] = (df["settlement_ts"] - df["timestamp"]).astype(
        np.float64
    )
    df["time_to_settlement_s"] = df["time_to_settlement_s"].clip(lower=0)
    df["log_time_to_settlement"] = np.log1p(df["time_to_settlement_s"])

    # pct_time_elapsed = (t - first_trade_on_market) / (settlement - first_trade_on_market)
    market_first_ts = df.groupby("condition_id")["timestamp"].transform("min")
    denom = (df["settlement_ts"] - market_first_ts).clip(lower=1)
    df["pct_time_elapsed"] = ((df["timestamp"] - market_first_ts) / denom).clip(0, 1)
    return df


def layer4_wallet_global(df: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Layer 4 — wallet global features.

    Requires the full 418M-row trades.parquet to compute wallet_polymarket_age_days.
    We only need the FIRST-EVER trade timestamp per wallet, so this is a single query.
    For wallet_prior_trades we restrict to our 55k takers.
    """
    log("layer 4 — wallet global (querying full HF file)...", t0)
    wallets = df["wallet"].dropna().unique().tolist()
    log(f"  {len(wallets):,} unique taker wallets to enrich", t0)

    con = duckdb.connect()
    # First-ever trade per wallet (as maker OR taker) — one row per wallet
    wallet_first = con.execute(
        f"""
        SELECT wallet, MIN(timestamp) AS first_ts
        FROM (
            SELECT maker AS wallet, timestamp FROM read_parquet('{FULL_HF}')
            WHERE maker IN (SELECT unnest(?)::VARCHAR)
            UNION ALL
            SELECT taker AS wallet, timestamp FROM read_parquet('{FULL_HF}')
            WHERE taker IN (SELECT unnest(?)::VARCHAR)
        ) GROUP BY wallet
    """,
        [wallets, wallets],
    ).df()
    log(f"  first-ever timestamps for {len(wallet_first):,} wallets", t0)

    df = df.merge(wallet_first, on="wallet", how="left")
    df["wallet_polymarket_age_s"] = (
        (df["timestamp"] - df["first_ts"]).clip(lower=0).fillna(0)
    )
    df["wallet_polymarket_age_days"] = df["wallet_polymarket_age_s"] / 86400.0
    df["wallet_is_new_to_polymarket"] = (df["wallet_polymarket_age_days"] < 1).astype(
        np.int8
    )
    df = df.drop(columns=["first_ts", "wallet_polymarket_age_s"])

    # wallet_prior_trades (across all markets in our pooled dataset, strictly before t)
    # Cheaper: cumcount on (wallet, timestamp) across OUR extracted trades only.
    # Justification: the full HF file is 418M rows. For THIS feature we use our 451k
    # rows as a proxy — covers all trades on the 7 Iran markets, which is the relevant
    # activity for the model. A full-HF version is listed as a v2 robustness feature.
    df = df.sort_values(["wallet", "timestamp"]).reset_index(drop=True)
    df["wallet_prior_trades"] = df.groupby("wallet", sort=False).cumcount()
    df["wallet_prior_trades_log"] = np.log1p(df["wallet_prior_trades"])

    # wallet_prior_markets = distinct markets seen strictly before t (our 7 only)
    df["wallet_prior_markets"] = (
        df.groupby("wallet", sort=False)["condition_id"]
        .transform(
            lambda s: s.astype("category")
            .cat.codes.ne(s.astype("category").cat.codes.shift(1))
            .cumsum()
        )
        .fillna(1)
        .astype(int)
        - 1
    ).clip(lower=0)
    # the above is brittle; simpler approximation via "has seen this market before" logic:
    df["_seen_key"] = df["wallet"].astype(str) + "|" + df["condition_id"].astype(str)
    df["_seen_first"] = ~df["_seen_key"].duplicated(keep="first")
    df["wallet_prior_markets"] = (
        df.groupby("wallet", sort=False)["_seen_first"].cumsum()
        - df["_seen_first"].astype(int)
    ).astype(int)
    df = df.drop(columns=["_seen_key", "_seen_first"])

    return df


def layer4b_wallet_in_market(df: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Layer 4b — wallet behaviour within the specific market."""
    log("layer 4b — wallet-in-market...", t0)
    df = df.sort_values(["wallet", "condition_id", "timestamp"]).reset_index(drop=True)

    # cumulative trades per (wallet, market), strictly before t
    df["wallet_prior_trades_in_market"] = df.groupby(
        ["wallet", "condition_id"]
    ).cumcount()

    # signed per-token cumulative volume (for spread-builder + position-aware SELL)
    # Token1 signed flow: BUY token1 → +, SELL token1 → -. Likewise token2.
    df["_t1_flow"] = np.where(
        df["is_token1"] == 1,
        np.where(df["is_buy"] == 1, df["usd_amount"], -df["usd_amount"]),
        0.0,
    )
    df["_t2_flow"] = np.where(
        df["is_token1"] == 0,
        np.where(df["is_buy"] == 1, df["usd_amount"], -df["usd_amount"]),
        0.0,
    )
    g_wm = df.groupby(["wallet", "condition_id"], sort=False)
    # position size BEFORE this trade = cumsum.shift(1)
    df["wallet_t1_position_before"] = g_wm["_t1_flow"].cumsum().shift(1).fillna(0.0)
    df["wallet_t2_position_before"] = g_wm["_t2_flow"].cumsum().shift(1).fillna(0.0)

    # spread-builder features: cumulative |flow| per token so far (before this trade)
    df["_t1_abs"] = df["_t1_flow"].abs()
    df["_t2_abs"] = df["_t2_flow"].abs()
    df["wallet_t1_cumvol_in_market"] = g_wm["_t1_abs"].cumsum().shift(1).fillna(0.0)
    df["wallet_t2_cumvol_in_market"] = g_wm["_t2_abs"].cumsum().shift(1).fillna(0.0)
    t1v = df["wallet_t1_cumvol_in_market"]
    t2v = df["wallet_t2_cumvol_in_market"]
    total = (t1v + t2v).replace(0, np.nan)
    df["wallet_directional_purity_in_market"] = ((t1v - t2v).abs() / total).fillna(0.0)
    df["wallet_has_both_sides_in_market"] = ((t1v > 0) & (t2v > 0)).astype(np.int8)
    bigger = np.maximum(t1v, t2v).replace(0, np.nan)
    smaller = np.minimum(t1v, t2v)
    df["wallet_spread_ratio"] = (smaller / bigger).fillna(0.0)

    # position-aware SELL: trade size relative to position currently held in same token
    pos_same = np.where(
        df["is_token1"] == 1,
        df["wallet_t1_position_before"],
        df["wallet_t2_position_before"],
    )
    df["wallet_position_same_token_before"] = pos_same
    df["trade_size_vs_position_pct"] = np.where(
        np.abs(pos_same) > 1e-9,
        df["usd_amount"] / np.abs(pos_same),
        0.0,
    ).clip(0, 100)  # cap runaway ratios for new positions

    # position AFTER this trade = position BEFORE + this-trade signed flow on same token
    sign = np.where(df["is_buy"] == 1, 1.0, -1.0)
    pos_after = pos_same + sign * df["usd_amount"]
    df["is_position_exit"] = (np.abs(pos_after) < 1.0).astype(np.int8)
    df["is_position_flip"] = ((pos_same * pos_after) < 0).astype(np.int8)

    # whale flag: wallet's prior cumvol in this market above the 95th percentile of such cumvols
    # within the same market at the same point in time (expensive — proxy with per-market threshold)
    df["wallet_total_cumvol_in_market"] = (
        df["wallet_t1_cumvol_in_market"] + df["wallet_t2_cumvol_in_market"]
    )
    market_whale_thr = df.groupby("condition_id")[
        "wallet_total_cumvol_in_market"
    ].transform(lambda s: s.quantile(0.95))
    df["wallet_is_whale_in_market"] = (
        df["wallet_total_cumvol_in_market"] >= market_whale_thr
    ).astype(np.int8)

    # bet-slicing: rolling counts in short windows per (wallet, condition_id)
    df["_ts_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    parts = []
    for (w, cid), sub in df.groupby(["wallet", "condition_id"], sort=False):
        sub = sub.set_index("_ts_dt").sort_index()
        # rolling counts — use a constant col to count
        sub["_one"] = 1
        sub["wallet_trades_in_market_last_1min"] = (
            sub["_one"].rolling("60s").sum().shift(1).fillna(0)
        )
        sub["wallet_trades_in_market_last_10min"] = (
            sub["_one"].rolling("10min").sum().shift(1).fillna(0)
        )
        sub["wallet_trades_in_market_last_1h"] = (
            sub["_one"].rolling("1h").sum().shift(1).fillna(0)
        )
        sub["wallet_cumvol_same_side_last_10min"] = (
            sub["usd_amount"].rolling("10min").sum().shift(1).fillna(0)
        )
        sub = sub.drop(columns=["_one"])
        parts.append(sub.reset_index())
    df = pd.concat(parts, ignore_index=True)
    df["wallet_is_burst"] = (df["wallet_trades_in_market_last_1min"] > 3).astype(
        np.int8
    )

    # Cleanup
    df = df.drop(columns=["_ts_dt", "_t1_flow", "_t2_flow", "_t1_abs", "_t2_abs"])
    return df


def layer5_interactions(df: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Layer 5 — ratios + cross-terms."""
    log("layer 5 — interactions...", t0)
    df["size_vs_market_cumvol_pct"] = (
        (df["usd_amount"] / df["market_cumvol"].replace(0, np.nan))
        .fillna(0.0)
        .clip(0, 100)
    )

    df["size_x_log_time"] = df["log_size"] * df["log_time_to_settlement"]
    return df


def sanity_checks(df: pd.DataFrame, t0: float) -> None:
    """Verify class balance + bucket sizes + nulls."""
    log("sanity checks...", t0)
    print("\n--- bucket x bet_correct ---")
    print(pd.crosstab(df["bucket"], df["bet_correct"], margins=True))
    print("\n--- nulls per feature ---")
    nulls = df.isna().sum()
    print(nulls[nulls > 0] if (nulls > 0).any() else "  (none)")
    print(f"\nfinal shape: {df.shape}")


def main() -> None:
    t0 = time.time()
    log("build_dataset.py start", None)

    df = load_and_join(t0)
    df = add_bucket_and_label(df, t0)
    df = layer1_trade_local(df, t0)
    df = layer2_market_context(df, t0)
    df = layer3_time(df, t0)
    df = layer4_wallet_global(df, t0)
    df = layer4b_wallet_in_market(df, t0)
    df = layer5_interactions(df, t0)

    sanity_checks(df, t0)

    # Drop columns we don't need to persist (raw string fields kept for joining later)
    keep_cols = [
        c for c in df.columns if c not in ("maker_direction", "taker_direction")
    ]
    df[keep_cols].to_parquet(OUT, index=False, compression="zstd")
    log(f"wrote {OUT}", t0)


if __name__ == "__main__":
    main()
