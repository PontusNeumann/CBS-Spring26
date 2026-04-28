"""
06b_engineer_features.py

Comprehensive feature engineering for Polymarket Iran-strike → Iran-ceasefire cohort.
Reads alex/data/{train,test}.parquet (raw HF trade rows), produces:
  alex/data/train_features.parquet
  alex/data/test_features.parquet

Each output is a single dataframe with: feature columns + market_id + bet_correct + ts_dt.

Feature groups (~60 features total):
  A. Trade-local (4)
  B. Time + cyclical (8)
  C. Per-market rolling state (12)
  D. Multi-timescale order flow (5)
  E. Trade microstructure (4)
  F. Price dynamics (5)
  G. Token-side dynamics (3)
  H. Consensus + contrarian (6)
  I. Economic / payoff (2)
  J. Microstructure literature (3)
  K. Polymarket-unique (3)
  L. On-chain ordering (2)
  M. Wallet (within-HF-only, no enrichment) (5)

No-lookahead enforced via shift / closed='left' rolling. No wallet enrichment from
external sources — wallet aggregates use only data inside the HF dataset.

Constraints:
  - Withhold current trade price from the feature set (price echoes; use pre_trade_price)
  - All rolling windows use closed='left' (exclude current row)
  - Standardisation/scaling deferred to modelling scripts (so different models can choose)
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Real-world events
STRIKE_EVENT_UTC = pd.Timestamp("2026-02-28T06:35:00", tz="UTC")
CEASEFIRE_ANNOUNCEMENT_UTC = pd.Timestamp("2026-04-07T23:59:59", tz="UTC")

DEADLINE_RE = re.compile(
    r"by\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d+)(?:,\s*(\d{4}))?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Target derivation
# ---------------------------------------------------------------------------


def parse_deadline_from_question(
    question: str, default_year: int = 2026
) -> pd.Timestamp | None:
    m = DEADLINE_RE.search(question)
    if not m:
        return None
    month_str, day_str, year_str = m.groups()
    year = int(year_str) if year_str else default_year
    try:
        return pd.Timestamp(f"{month_str} {day_str} {year}", tz="UTC")
    except Exception:
        return None


def derive_winning_token(markets: pd.DataFrame) -> pd.DataFrame:
    m = markets.copy()
    winning = []
    for _, row in m.iterrows():
        try:
            prices = ast.literal_eval(row["outcome_prices"])
            p1, p2 = float(prices[0]), float(prices[1])
            if abs(p1 - 1.0) < 0.01 and abs(p2 - 0.0) < 0.01:
                winning.append("token1")
                continue
            if abs(p1 - 0.0) < 0.01 and abs(p2 - 1.0) < 0.01:
                winning.append("token2")
                continue
        except Exception:
            pass

        question = row["question"]
        deadline = parse_deadline_from_question(question)
        if deadline is None:
            ed = row["end_date"]
            deadline = (
                pd.Timestamp(ed).tz_convert("UTC")
                if hasattr(ed, "tzinfo") and ed.tzinfo
                else pd.Timestamp(ed, tz="UTC")
            )

        cohort = row["cohort"]
        if cohort == "train":
            winning.append("token1" if deadline >= STRIKE_EVENT_UTC else "token2")
        elif cohort == "test":
            winning.append(
                "token1" if deadline >= CEASEFIRE_ANNOUNCEMENT_UTC else "token2"
            )
        else:
            winning.append(None)
    m["winning_token"] = winning
    return m


def derive_bet_correct(trades: pd.DataFrame, win_map: dict) -> pd.Series:
    market_id = trades["market_id"].astype(str)
    winning = market_id.map(win_map)
    is_buy = trades["taker_direction"].astype(str).str.upper().eq("BUY")
    side = trades["nonusdc_side"].astype(str)
    correct = np.where(
        is_buy,
        (side == winning).astype(int),
        (side != winning).astype(int),
    )
    return pd.Series(correct, index=trades.index, name="bet_correct")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def add_trade_local(df: pd.DataFrame) -> pd.DataFrame:
    """A. Trade-local features."""
    # log_size from token_amount if available, else derived
    if "token_amount" in df.columns:
        df["log_size"] = np.log1p(df["token_amount"].clip(lower=0))
    elif "size" in df.columns:
        df["log_size"] = np.log1p(df["size"].clip(lower=0))
    else:
        df["log_size"] = np.log1p(df["usd_amount"].clip(lower=0))

    df["side_buy"] = df["taker_direction"].astype(str).str.upper().eq("BUY").astype(int)
    df["outcome_yes"] = (df["nonusdc_side"].astype(str) == "token1").astype(int)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """B. Time + cyclical."""
    open_ts = pd.to_datetime(df["created_at"], utc=True)
    end_ts = pd.to_datetime(df["end_date"], utc=True)
    secs_to_deadline = (end_ts - df["ts_dt"]).dt.total_seconds().clip(lower=1)
    secs_since_open = (df["ts_dt"] - open_ts).dt.total_seconds().clip(lower=1)
    total_lifetime = (end_ts - open_ts).dt.total_seconds().clip(lower=1)

    df["log_time_to_deadline_hours"] = np.log1p(secs_to_deadline / 3600)
    df["pct_time_elapsed"] = (secs_since_open / total_lifetime).clip(0, 1)

    # Per-market gap from prior trade
    df["_gap_sec"] = df.groupby("market_id")["timestamp"].diff().fillna(0).clip(lower=0)
    df["log_time_since_last_trade"] = np.log1p(df["_gap_sec"])
    df["is_first_trade_after_quiet"] = (df["_gap_sec"] > 3600).astype(int)

    # Urgency booleans
    df["is_within_24h_of_deadline"] = (secs_to_deadline < 24 * 3600).astype(int)
    df["is_within_1h_of_deadline"] = (secs_to_deadline < 3600).astype(int)
    df["is_within_5min_of_deadline"] = (secs_to_deadline < 300).astype(int)

    # Cyclical
    hour = df["ts_dt"].dt.hour + df["ts_dt"].dt.minute / 60.0
    df["hour_of_day_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_of_day_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = df["ts_dt"].dt.dayofweek.astype(float)
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["is_us_market_hours"] = (
        (df["ts_dt"].dt.hour >= 14) & (df["ts_dt"].dt.hour < 21)
    ).astype(int)
    return df


def add_market_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """C. Per-market rolling state. Uses time-indexed rolling with closed='left'."""
    df["_uval"] = df["usd_amount"].clip(lower=0)
    df["_signed_uval"] = df["_uval"] * (2 * df["side_buy"] - 1)
    df["_log_uval"] = np.log1p(df["_uval"])

    # Cumulative trade count (excludes current via cumcount)
    df["_n_to_date"] = df.groupby("market_id").cumcount()
    df["log_n_trades_to_date"] = np.log1p(df["_n_to_date"])

    # Running buy-share (cum-volume to-date, excluding current)
    grouped = df.groupby("market_id")
    df["_buy_cumvol"] = (
        (df["side_buy"] * df["_uval"])
        .groupby(df["market_id"])
        .cumsum()
        .shift(1)
        .fillna(0)
    )
    df["_total_cumvol"] = (
        df["_uval"].groupby(df["market_id"]).cumsum().shift(1).fillna(0)
    )
    # Reset shift across market boundaries
    first_idx = df.groupby("market_id").head(1).index
    df.loc[first_idx, ["_buy_cumvol", "_total_cumvol"]] = 0
    df["market_buy_share_running"] = (
        df["_buy_cumvol"] / df["_total_cumvol"].clip(lower=1.0)
    ).clip(0, 1)

    # Time-indexed rolling computations across multiple windows
    df = df.set_index("ts_dt")

    for win, label in [("5min", "5min"), ("1h", "1h"), ("24h", "24h")]:
        # Volume in window (USD)
        df[f"_vol_{label}"] = (
            df.groupby("market_id")["_uval"]
            .rolling(win, closed="left")
            .sum()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        df[f"log_recent_volume_{label}"] = np.log1p(df[f"_vol_{label}"])

        # Trade count in window
        df[f"_n_{label}"] = (
            df.groupby("market_id")["_uval"]
            .rolling(win, closed="left")
            .count()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        df[f"log_trade_count_{label}"] = np.log1p(df[f"_n_{label}"])

        # Order-flow imbalance in window
        signed_sum = (
            df.groupby("market_id")["_signed_uval"]
            .rolling(win, closed="left")
            .sum()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        df[f"order_flow_imbalance_{label}"] = (
            signed_sum / df[f"_vol_{label}"].clip(lower=1.0)
        ).clip(-1, 1)

        # Price volatility in window (std of price changes)
        if "price" in df.columns:
            df[f"_pchg"] = df.groupby("market_id")["price"].diff().fillna(0)
            df[f"market_price_vol_last_{label}"] = (
                df.groupby("market_id")["_pchg"]
                .rolling(win, closed="left")
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0)
            )
        else:
            df[f"market_price_vol_last_{label}"] = 0.0

    # Trade size relative to recent volume
    df["trade_size_to_recent_volume_ratio"] = (
        df["_uval"] / df["_vol_1h"].clip(lower=1.0)
    ).clip(0, 100)
    # Avg trade size last 1h
    df["avg_trade_size_recent_1h"] = df["_vol_1h"] / df["_n_1h"].clip(lower=1.0)
    df["trade_size_vs_recent_avg"] = (
        df["_uval"] / df["avg_trade_size_recent_1h"].clip(lower=1.0)
    ).clip(0, 100)

    df = df.reset_index()
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """F. Price dynamics + pre-trade price (Framing C). Uses lag-1 and time-indexed rolling means."""
    if "price" not in df.columns:
        df["pre_trade_price"] = 0.5
        for c in [
            "recent_price_mean_5min",
            "recent_price_mean_1h",
            "pre_trade_price_change_1h",
            "pre_trade_price_change_5min",
            "pre_trade_price_change_24h",
            "recent_price_high_1h",
            "recent_price_low_1h",
            "recent_price_range_1h",
        ]:
            df[c] = 0.0
        return df

    # Lag-1 price (most recent observable)
    df["pre_trade_price"] = df.groupby("market_id")["price"].shift(1).fillna(0.5)

    df = df.set_index("ts_dt")
    for win, label in [("5min", "5min"), ("1h", "1h"), ("24h", "24h")]:
        df[f"recent_price_mean_{label}"] = (
            df.groupby("market_id")["price"]
            .rolling(win, closed="left")
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(df["pre_trade_price"].mean())
        )
    df["recent_price_high_1h"] = (
        df.groupby("market_id")["price"]
        .rolling("1h", closed="left")
        .max()
        .reset_index(level=0, drop=True)
        .fillna(df["pre_trade_price"].mean())
    )
    df["recent_price_low_1h"] = (
        df.groupby("market_id")["price"]
        .rolling("1h", closed="left")
        .min()
        .reset_index(level=0, drop=True)
        .fillna(df["pre_trade_price"].mean())
    )
    df = df.reset_index()

    df["recent_price_range_1h"] = df["recent_price_high_1h"] - df["recent_price_low_1h"]
    df["pre_trade_price_change_5min"] = (
        df["pre_trade_price"] - df["recent_price_mean_5min"]
    )
    df["pre_trade_price_change_1h"] = df["pre_trade_price"] - df["recent_price_mean_1h"]
    df["pre_trade_price_change_24h"] = (
        df["pre_trade_price"] - df["recent_price_mean_24h"]
    )
    return df


def add_token_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """G. Token-side dynamics — YES vs NO flow asymmetry."""
    df["_yes_uval"] = df["_uval"] * df["outcome_yes"]
    df["_yes_buy_uval"] = df["_yes_uval"] * df["side_buy"]
    df = df.set_index("ts_dt")
    for win, label in [("5min", "5min"), ("1h", "1h")]:
        yes_vol = (
            df.groupby("market_id")["_yes_uval"]
            .rolling(win, closed="left")
            .sum()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        total_vol = df[f"_vol_{label}"].fillna(0)
        df[f"yes_volume_share_recent_{label}"] = (
            yes_vol / total_vol.clip(lower=1.0)
        ).clip(0, 1)

    yes_buy_vol = (
        df.groupby("market_id")["_yes_buy_uval"]
        .rolling("5min", closed="left")
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    df["yes_buy_pressure_5min"] = (yes_buy_vol / df["_vol_5min"].clip(lower=1.0)).clip(
        0, 1
    )

    df = df.reset_index()
    # Token side skew = yes_share - 0.5 over 5min (centred)
    df["token_side_skew_5min"] = df["yes_volume_share_recent_5min"] - 0.5
    return df


def add_consensus_contrarian(df: pd.DataFrame) -> pd.DataFrame:
    """H. Consensus & contrarian features."""
    p = df["pre_trade_price"]
    df["implied_variance"] = p * (1 - p)
    df["distance_from_boundary"] = np.minimum(p, 1 - p)
    df["consensus_strength"] = (np.abs(0.5 - p) * 2).clip(0, 1)

    direction = 2 * df["side_buy"] - 1
    df["contrarian_score"] = (direction * (0.5 - p) * 2).clip(-1, 1)
    df["is_long_shot_buy"] = ((df["side_buy"] == 1) & (p < 0.20)).astype(int)
    df["contrarian_strength"] = (
        df["consensus_strength"] * df["contrarian_score"].clip(lower=0)
    ).clip(0, 1)
    return df


def add_payoff_features(df: pd.DataFrame) -> pd.DataFrame:
    """I. Economic / payoff (Polymarket-unique convexity)."""
    p = df["pre_trade_price"].clip(lower=0.01, upper=0.99)
    # log_payoff_if_correct: log of payoff multiple if you're right
    # BUY at price p that wins → payoff = 1/p − 1, so log(payoff) = log((1-p)/p)
    # SELL at price p that wins → payoff = 1/(1-p) − 1, so log(payoff) = log(p/(1-p))
    direction = 2 * df["side_buy"] - 1  # +1 BUY, -1 SELL
    log_payoff_buy = np.log((1 - p) / p)
    df["log_payoff_if_correct"] = np.where(
        direction > 0, log_payoff_buy, -log_payoff_buy
    )
    df["risk_reward_ratio_pre"] = np.where(direction > 0, (1 - p) / p, p / (1 - p))
    return df


def add_microstructure_lit(df: pd.DataFrame) -> pd.DataFrame:
    """J. Microstructure literature — Kyle's lambda (static), realized vol, jump component."""
    if "price" not in df.columns:
        df["kyle_lambda_market_static"] = 0.0
        df["realized_vol_1h"] = 0.0
        df["jump_component_1h"] = 0.0
        df["signed_oi_autocorr_1h"] = 0.0
        return df

    # Kyle's lambda: per-market regression of price changes on signed sqrt-volume
    # Use first half of each market's life to fit (no leakage to test trades)
    lambdas = {}
    for mid, g in df.groupby("market_id"):
        g_sorted = g.sort_values("ts_dt")
        cutoff_idx = len(g_sorted) // 2
        if cutoff_idx < 20:
            lambdas[mid] = 0.0
            continue
        first_half = g_sorted.iloc[:cutoff_idx]
        pchg = first_half["price"].diff().dropna()
        signed_vol = (first_half["_signed_uval"].values)[1:]  # match diff
        signed_sqrt_vol = np.sign(signed_vol) * np.sqrt(np.abs(signed_vol) + 1e-9)
        if len(pchg) < 20 or signed_sqrt_vol.std() < 1e-9:
            lambdas[mid] = 0.0
            continue
        # OLS slope
        x = signed_sqrt_vol
        y = pchg.values
        if x.std() == 0:
            lambdas[mid] = 0.0
            continue
        slope = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
        lambdas[mid] = float(slope) if np.isfinite(slope) else 0.0
    df["kyle_lambda_market_static"] = df["market_id"].map(lambdas).fillna(0.0)

    # Realized volatility 1h: sum of squared price changes
    df = df.set_index("ts_dt")
    pchg2 = (df.groupby("market_id")["price"].diff() ** 2).fillna(0)
    df["_pchg2"] = pchg2.values
    df["realized_vol_1h"] = (
        df.groupby("market_id")["_pchg2"]
        .rolling("1h", closed="left")
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Bipower variation 1h: sum of |Δp_t| × |Δp_{t-1}|, then jump = realized_vol - bipower
    df["_abs_pchg"] = df.groupby("market_id")["price"].diff().abs().fillna(0)
    df["_abs_pchg_lag"] = df.groupby("market_id")["_abs_pchg"].shift(1).fillna(0)
    df["_bv_term"] = df["_abs_pchg"] * df["_abs_pchg_lag"]
    bv_1h = (
        df.groupby("market_id")["_bv_term"]
        .rolling("1h", closed="left")
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    # Jump component (simple residual; literature uses scaled bipower, this is good enough)
    df["jump_component_1h"] = (df["realized_vol_1h"] - 1.5708 * bv_1h).clip(lower=0)

    # Signed OFI autocorrelation 1h: corr(signed_uval_t, signed_uval_{t-1}) over rolling window
    # Approx via shift product mean
    df["_signed_lag"] = df.groupby("market_id")["_signed_uval"].shift(1).fillna(0)
    df["_signed_prod"] = df["_signed_uval"] * df["_signed_lag"]
    df["signed_oi_autocorr_1h"] = (
        df.groupby("market_id")["_signed_prod"]
        .rolling("1h", closed="left")
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    df = df.reset_index()
    return df


def add_polymarket_unique(df: pd.DataFrame) -> pd.DataFrame:
    """K. Polymarket-unique — sister-market dispersion across same event_id."""
    if "event_id" not in df.columns:
        df["sister_price_dispersion_5min"] = 0.0
        return df

    # For each event, at each timestamp, compute std of pre_trade_price across markets.
    # Approximate: for each market, take its pre_trade_price; group by event_id, rolling std.
    # Simpler approach: per event_id and 5-min time bucket, compute std of prices, broadcast.
    df["_5min_bucket"] = df["ts_dt"].dt.floor("5min")
    bucket_disp = (
        df.groupby(["event_id", "_5min_bucket"])["pre_trade_price"].std().fillna(0)
    )
    df["sister_price_dispersion_5min"] = (
        df.set_index(["event_id", "_5min_bucket"]).index.map(bucket_disp).astype(float)
    )
    df["sister_price_dispersion_5min"] = df["sister_price_dispersion_5min"].fillna(0)
    return df


def add_onchain_features(df: pd.DataFrame) -> pd.DataFrame:
    """L. On-chain ordering."""
    if "block_number" in df.columns:
        df["same_block_trade_count"] = df.groupby(["market_id", "block_number"])[
            "timestamp"
        ].transform("count")
        df["log_same_block_trade_count"] = np.log1p(df["same_block_trade_count"])
    else:
        df["log_same_block_trade_count"] = 0.0

    if "log_index" in df.columns:
        df["trade_intra_block_position"] = df["log_index"].fillna(0).astype(float)
    else:
        df["trade_intra_block_position"] = 0.0
    return df


def add_wallet_features(df: pd.DataFrame) -> pd.DataFrame:
    """M. Wallet features (within-HF only, no external enrichment).

    Comprehensive set of taker-focused signals plus simple maker counters.
    All aggregates are strictly prior-trade (cumcount / cumsum + shift / closed='left').

    Group I — within-market history (taker)
    Group II — global history (cross-market within cohort)
    Group III — behaviour patterns (directional purity, event-family experience, burst)
    Group IV — position/timing within market (entry recency, size vs personal avg)
    Group V — maker simple counter (cross-check)
    """
    has_taker = "taker" in df.columns
    has_maker = "maker" in df.columns

    if not has_taker:
        # Defaults: zeros so downstream models still run
        for col in [
            "log_taker_prior_trades_in_market",
            "taker_first_trade_in_market",
            "log_taker_cumvol_in_market",
            "taker_position_size_before_trade",
            "log_taker_prior_trades_total",
            "log_taker_prior_volume_total_usd",
            "log_taker_unique_markets_traded",
            "taker_yes_share_global",
            "taker_directional_purity_in_market",
            "taker_traded_in_event_id_before",
            "log_taker_burst_5min",
            "log_taker_first_minutes_ago_in_market",
            "log_size_vs_taker_avg",
        ]:
            df[col] = 0.0
        if has_maker:
            df["maker_prior_trades_in_market"] = df.groupby(
                ["market_id", "maker"]
            ).cumcount()
            df["log_maker_prior_trades_in_market"] = np.log1p(
                df["maker_prior_trades_in_market"]
            )
        else:
            df["log_maker_prior_trades_in_market"] = 0.0
        return df

    # ---- Group I: within-market wallet history --------------------------
    df["taker_prior_trades_in_market"] = df.groupby(["market_id", "taker"]).cumcount()
    df["log_taker_prior_trades_in_market"] = np.log1p(
        df["taker_prior_trades_in_market"]
    )
    df["taker_first_trade_in_market"] = (
        df["taker_prior_trades_in_market"] == 0
    ).astype(int)

    # Cumulative volume by (market, taker), strictly prior
    taker_cumvol = (
        df.groupby(["market_id", "taker"])["_uval"].cumsum().shift(1).fillna(0)
    )
    first_in_mt = df.groupby(["market_id", "taker"]).head(1).index
    taker_cumvol.loc[first_in_mt] = 0
    df["log_taker_cumvol_in_market"] = np.log1p(taker_cumvol.clip(lower=0))

    # Signed position before trade: +token_amount for BUY of token1 (YES); −token_amount for SELL of token1
    # Track per (market, taker) accumulated YES-equivalent token position
    # BUY YES = +tokens, SELL YES = −tokens
    # BUY NO  = −tokens (equivalent to SELL YES via 1-p), SELL NO = +tokens
    sign_yes_equiv = (2 * df["side_buy"] - 1) * (
        2 * df["outcome_yes"] - 1
    )  # +1 if BUY YES or SELL NO, -1 otherwise
    tokens = (df["token_amount"] if "token_amount" in df.columns else df["_uval"]).clip(
        lower=0
    )
    df["_signed_tokens"] = sign_yes_equiv * tokens
    pos_cum = (
        df.groupby(["market_id", "taker"])["_signed_tokens"].cumsum().shift(1).fillna(0)
    )
    pos_cum.loc[first_in_mt] = 0
    df["taker_position_size_before_trade"] = np.tanh(
        pos_cum / 1000.0
    )  # bounded [-1, 1]

    # ---- Group II: global wallet history (cross-market within cohort) ----
    # Sort by timestamp globally for these computations
    df_sorted = df.sort_values("timestamp")
    glob_count = df_sorted.groupby("taker").cumcount()
    df["log_taker_prior_trades_total"] = np.log1p(glob_count.reindex(df.index))

    glob_vol = df_sorted.groupby("taker")["_uval"].cumsum().shift(1).fillna(0)
    first_taker_idx = df_sorted.groupby("taker").head(1).index
    glob_vol.loc[first_taker_idx] = 0
    df["log_taker_prior_volume_total_usd"] = np.log1p(
        glob_vol.reindex(df.index).clip(lower=0)
    )

    # Unique markets traded so far (cross-market)
    # Approach: for each taker, their trades sorted by time; count distinct market_ids seen so far (including current)
    # Slight overestimate (includes current market), so subtract 1 if current market is "new" for them, else 0.
    # Simpler approach: rank of first appearance of (taker, market_id), per taker
    # = number of distinct markets touched as of this trade
    df_sorted_unique = df_sorted.copy()
    df_sorted_unique["_market_first"] = (
        df_sorted_unique.groupby(["taker", "market_id"]).cumcount() == 0
    ).astype(int)
    unique_markets = df_sorted_unique.groupby("taker")["_market_first"].cumsum()
    # Strictly prior: subtract current contribution if it was the first
    unique_markets_prior = unique_markets - df_sorted_unique["_market_first"]
    df["log_taker_unique_markets_traded"] = np.log1p(
        unique_markets_prior.reindex(df.index).clip(lower=0)
    )

    # Running global YES share (running mean of outcome_yes, strictly prior)
    yes_cum = df_sorted.groupby("taker")["outcome_yes"].cumsum().shift(1).fillna(0)
    yes_cum.loc[first_taker_idx] = 0
    yes_share = yes_cum / glob_count.replace(0, np.nan)
    df["taker_yes_share_global"] = yes_share.reindex(df.index).fillna(0.5).clip(0, 1)

    # ---- Group III: behaviour patterns ------------------------------------
    # Directional purity in market: % of prior trades on the SAME side (BUY/SELL of same token)
    # "side encoding": (side_buy, outcome_yes) → 4-way category
    df["_side_code"] = df["side_buy"].astype(int) * 2 + df["outcome_yes"].astype(int)

    # For each (market, taker), running count of trades matching current trade's side
    # Use: cumsum where 1 if same side, then shifted
    # Approach per (market, taker): for each row, how many prior rows had same _side_code?
    def _purity(group):
        codes = group["_side_code"].values
        n = len(codes)
        purity = np.zeros(n, dtype=float)
        for i in range(1, n):
            same = (codes[:i] == codes[i]).sum()
            purity[i] = same / i
        return pd.Series(purity, index=group.index)

    df["taker_directional_purity_in_market"] = df.groupby(
        ["market_id", "taker"], group_keys=False
    ).apply(_purity)

    # Has this taker traded in this event_id before (any rung of the ladder)?
    if "event_id" in df.columns:
        df["taker_event_count_prior"] = df.groupby(["event_id", "taker"]).cumcount()
        df["taker_traded_in_event_id_before"] = (
            df["taker_event_count_prior"] > 0
        ).astype(int)
    else:
        df["taker_traded_in_event_id_before"] = 0

    # Burst: count of taker's trades in this market in last 5min (exclusive)
    # Approach: per (market, taker), use timestamps; for each trade i, count prior j with t_i - t_j <= 300s
    def _burst5min(group):
        ts = group["timestamp"].values
        n = len(ts)
        burst = np.zeros(n, dtype=int)
        # two-pointer: prior trades within 300s
        left = 0
        for i in range(n):
            while ts[i] - ts[left] > 300:
                left += 1
            burst[i] = i - left  # count of prior trades in window
        return pd.Series(burst, index=group.index)

    df["taker_burst_5min"] = df.groupby(["market_id", "taker"], group_keys=False).apply(
        _burst5min
    )
    df["log_taker_burst_5min"] = np.log1p(df["taker_burst_5min"])

    # ---- Group IV: position/timing -----------------------------------------
    # Time since taker's first trade in this market (seconds)
    first_ts_per_mt = df.groupby(["market_id", "taker"])["timestamp"].transform("first")
    secs_since_first = (df["timestamp"] - first_ts_per_mt).clip(lower=0)
    df["log_taker_first_minutes_ago_in_market"] = np.log1p(secs_since_first / 60)

    # Size vs taker's running average size
    taker_avg_size = df_sorted.groupby("taker")["_uval"].cumsum().shift(1).fillna(
        0
    ) / glob_count.replace(0, np.nan)
    avg_size_aligned = taker_avg_size.reindex(df.index).fillna(df["_uval"].median())
    df["log_size_vs_taker_avg"] = np.log1p(
        (df["_uval"] / avg_size_aligned.clip(lower=1e-3)).clip(0, 1000)
    )

    # ---- Group V: maker counter --------------------------------------------
    if has_maker:
        df["maker_prior_trades_in_market"] = df.groupby(
            ["market_id", "maker"]
        ).cumcount()
        df["log_maker_prior_trades_in_market"] = np.log1p(
            df["maker_prior_trades_in_market"]
        )
    else:
        df["log_maker_prior_trades_in_market"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


FEATURE_COLS = [
    # A. Trade-local
    "log_size",
    "side_buy",
    "outcome_yes",
    # B. Time + cyclical (dropped is_first_trade_after_quiet, hour_of_day_cos,
    # is_us_market_hours — RF importance ~0 in v3)
    "log_time_to_deadline_hours",
    "pct_time_elapsed",
    "log_time_since_last_trade",
    "is_within_24h_of_deadline",
    "is_within_1h_of_deadline",
    "is_within_5min_of_deadline",
    "hour_of_day_sin",
    "day_of_week_sin",
    "day_of_week_cos",
    # C. Per-market rolling
    "log_n_trades_to_date",
    "market_buy_share_running",
    "log_recent_volume_5min",
    "log_recent_volume_1h",
    "log_recent_volume_24h",
    "log_trade_count_5min",
    "log_trade_count_1h",
    "log_trade_count_24h",
    "market_price_vol_last_5min",
    "market_price_vol_last_1h",
    "market_price_vol_last_24h",
    # D. Order flow
    "order_flow_imbalance_5min",
    "order_flow_imbalance_1h",
    "order_flow_imbalance_24h",
    # E. Trade size relative
    "trade_size_to_recent_volume_ratio",
    "trade_size_vs_recent_avg",
    "avg_trade_size_recent_1h",
    # F. Price dynamics (pre-trade only)
    "pre_trade_price",
    "recent_price_mean_5min",
    "recent_price_mean_1h",
    "recent_price_mean_24h",
    "recent_price_high_1h",
    "recent_price_low_1h",
    "recent_price_range_1h",
    "pre_trade_price_change_5min",
    "pre_trade_price_change_1h",
    "pre_trade_price_change_24h",
    # G. Token-side dynamics
    "yes_volume_share_recent_5min",
    "yes_volume_share_recent_1h",
    "yes_buy_pressure_5min",
    "token_side_skew_5min",
    # H. Consensus + contrarian
    "implied_variance",
    "distance_from_boundary",
    "consensus_strength",
    "contrarian_score",
    "is_long_shot_buy",
    "contrarian_strength",
    # I. Economic / payoff
    "log_payoff_if_correct",
    "risk_reward_ratio_pre",
    # J. Microstructure literature
    "kyle_lambda_market_static",
    "realized_vol_1h",
    "jump_component_1h",
    "signed_oi_autocorr_1h",
    # K. Polymarket-unique (dropped sister_price_dispersion_5min — RF importance ~0)
    # L. On-chain ordering (dropped trade_intra_block_position — RF importance ~0)
    "log_same_block_trade_count",
    # M. Wallet (within-HF only) — expanded for v3.5
    # Group I: within-market history
    "log_taker_prior_trades_in_market",
    "taker_first_trade_in_market",
    "log_taker_cumvol_in_market",
    "taker_position_size_before_trade",
    # Group II: global history (cross-market within cohort)
    "log_taker_prior_trades_total",
    "log_taker_prior_volume_total_usd",
    "log_taker_unique_markets_traded",
    "taker_yes_share_global",
    # Group III: behaviour patterns
    "taker_directional_purity_in_market",
    "taker_traded_in_event_id_before",
    "log_taker_burst_5min",
    # Group IV: position/timing within market
    "log_taker_first_minutes_ago_in_market",
    "log_size_vs_taker_avg",
    # Group V: maker counter
    "log_maker_prior_trades_in_market",
]


def engineer(
    trades: pd.DataFrame, markets: pd.DataFrame, win_map: dict
) -> pd.DataFrame:
    print(f"  rows in: {len(trades):,}")
    df = trades.copy()
    md = markets[["id", "created_at", "end_date", "event_id"]].copy()
    md["id"] = md["id"].astype(str)
    df["market_id"] = df["market_id"].astype(str)
    df = df.merge(
        md, left_on="market_id", right_on="id", how="left", suffixes=("", "_md")
    )
    if "event_id_md" in df.columns:
        df["event_id"] = df["event_id_md"]
        df = df.drop(columns=[c for c in df.columns if c.endswith("_md")])

    df["ts_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    print("  + trade-local")
    df = add_trade_local(df)
    print("  + time")
    df = add_time_features(df)
    print("  + market rolling")
    df = add_market_rolling(df)
    print("  + price")
    df = add_price_features(df)
    print("  + token dynamics")
    df = add_token_dynamics(df)
    print("  + consensus/contrarian")
    df = add_consensus_contrarian(df)
    print("  + payoff")
    df = add_payoff_features(df)
    print("  + microstructure lit")
    df = add_microstructure_lit(df)
    print("  + polymarket-unique")
    df = add_polymarket_unique(df)
    print("  + on-chain")
    df = add_onchain_features(df)
    print("  + wallet")
    df = add_wallet_features(df)

    # Target
    print("  + target")
    df["bet_correct"] = derive_bet_correct(df, win_map)

    # Keep only the features + target + group + time
    keep = FEATURE_COLS + ["market_id", "bet_correct", "ts_dt", "timestamp"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out = out.fillna(0)
    print(f"  rows out: {len(out):,}, cols: {len(out.columns)}")
    return out


def main():
    print("=" * 60)
    print("feature engineering: idea1 cohort, ~60 features")
    print("=" * 60)
    train_raw = pd.read_parquet(DATA / "train.parquet")
    test_raw = pd.read_parquet(DATA / "test.parquet")
    markets = pd.read_parquet(DATA / "markets_subset.parquet")
    markets = derive_winning_token(markets)
    win_map = dict(zip(markets["id"].astype(str), markets["winning_token"]))
    print(
        f"markets: {len(markets)}, winning tokens derived: {markets['winning_token'].notna().sum()}"
    )

    print("\n[train]")
    train_feat = engineer(train_raw, markets, win_map)
    train_feat.to_parquet(DATA / "train_features.parquet", index=False)
    print(f"  -> {DATA / 'train_features.parquet'}")

    print("\n[test]")
    test_feat = engineer(test_raw, markets, win_map)
    test_feat.to_parquet(DATA / "test_features.parquet", index=False)
    print(f"  -> {DATA / 'test_features.parquet'}")

    # Save feature list
    import json

    (DATA / "feature_cols.json").write_text(json.dumps(FEATURE_COLS, indent=2))
    print(
        f"\nfeature list: {DATA / 'feature_cols.json'} ({len(FEATURE_COLS)} features)"
    )
    print(f"\ndone.")


if __name__ == "__main__":
    main()
