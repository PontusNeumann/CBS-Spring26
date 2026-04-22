"""Polymarket data fetcher for the Iran prediction markets project.

Pulls three layers of data from Polymarket's public APIs:
    1. Event and market metadata            (Gamma API)
    2. Intraday price history per outcome   (CLOB API)
    3. Trade-level history per market       (Data API)

Outputs CSV files into ../data/ (i.e. report/data/, one level above scripts/).
Primarily used as a library by scripts/02_build_dataset.py; the
standalone `main()` is retained for ad-hoc API-only pulls.

Targets: events 114242, 236884, 355299, 357625 (Iran strikes, Iran-Israel/US
conflict end, Trump ceasefire announcement, ceasefire extensions).
Run:     python scripts/01_polymarket_api.py

Known limitation of the standalone path: the Data API caps pagination offset
at ~3000. Side-split fallback lifts the ceiling to ~7000 trades per market.
For the 67 markets under events 114242 and 236884 this cap is bypassed by
02_build_dataset.py, which streams the HuggingFace `SII-WANGZJ/Polymarket_data`
mirror instead.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"

TARGET_EVENT_IDS = ["114242", "236884", "355299", "357625"]

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent / "data"

SLEEP_SEC = 0.25
TIMEOUT = 30


@dataclass
class MarketRef:
    event_id: str
    market_id: str
    condition_id: str
    slug: str
    question: str
    outcomes: list[str]
    token_ids: list[str]
    closed: bool
    resolved: bool
    winning_outcome_index: int | None
    end_date: str | None


def _get(url: str, params: dict | None = None) -> dict | list:
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    time.sleep(SLEEP_SEC)
    return r.json()


def fetch_event(event_id: str) -> dict:
    result = _get(f"{GAMMA}/events/{event_id}")
    assert isinstance(result, dict)
    return result


def _parse_resolution(outcome_prices_raw: str | None) -> tuple[bool, int | None]:
    """Gamma reports final resolution via outcomePrices, e.g. ["1","0"] or ["0","1"].
    Returns (resolved, winning_outcome_index). Undecided markets return (False, None).
    """
    try:
        prices = json.loads(outcome_prices_raw or "[]")
    except json.JSONDecodeError:
        return False, None
    if len(prices) != 2:
        return False, None
    try:
        p0, p1 = float(prices[0]), float(prices[1])
    except (TypeError, ValueError):
        return False, None
    if p0 == 1.0 and p1 == 0.0:
        return True, 0
    if p0 == 0.0 and p1 == 1.0:
        return True, 1
    return False, None


def parse_markets(event: dict) -> list[MarketRef]:
    refs: list[MarketRef] = []
    for m in event.get("markets", []):
        outcomes = json.loads(m.get("outcomes") or "[]")
        token_ids = json.loads(m.get("clobTokenIds") or "[]")
        resolved, winner = _parse_resolution(m.get("outcomePrices"))
        refs.append(
            MarketRef(
                event_id=str(event["id"]),
                market_id=str(m["id"]),
                condition_id=m.get("conditionId", ""),
                slug=m.get("slug", ""),
                question=m.get("question", ""),
                outcomes=outcomes,
                token_ids=token_ids,
                closed=bool(m.get("closed", False)),
                resolved=resolved,
                winning_outcome_index=winner,
                end_date=m.get("endDate"),
            )
        )
    return refs


_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}
_DEADLINE_FULL_RE = re.compile(
    r"\bby\s+(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
_DEADLINE_NOYEAR_RE = re.compile(
    r"\bby\s+(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{1,2})\b",
    re.IGNORECASE,
)


def parse_deadline_from_question(
    question: str | None, fallback_year: int | None = None
) -> pd.Timestamp | None:
    """Extract the advertised deadline from a Polymarket question.

    Matches patterns like:
      - "US strikes Iran by February 27, 2026?"      → 2026-02-27
      - "Iran x Israel/US conflict ends by March 7?" → <fallback_year>-03-07
        (the year is inferred when not explicit; pass `fallback_year` from
         `end_date.year` on the same row so behaviour stays deterministic).

    The deadline is the market's own published resolution boundary — part of
    the question string known at market creation — so using it as a feature
    reference is strictly non-leaky. Returns None when parsing fails.
    """
    if not isinstance(question, str) or not question:
        return None
    m = _DEADLINE_FULL_RE.search(question)
    if m is not None:
        mo, d, y = m.groups()
        try:
            return pd.Timestamp(
                year=int(y), month=_MONTHS[mo.lower()], day=int(d), tz="UTC"
            )
        except (ValueError, KeyError):
            return None
    if fallback_year is None:
        return None
    m = _DEADLINE_NOYEAR_RE.search(question)
    if m is None:
        return None
    mo, d = m.groups()
    try:
        return pd.Timestamp(
            year=int(fallback_year), month=_MONTHS[mo.lower()], day=int(d), tz="UTC"
        )
    except (ValueError, KeyError):
        return None


def fetch_price_history(
    token_id: str, interval: str = "all", fidelity: int = 1
) -> pd.DataFrame:
    """Time series of the CLOB mid-price for one outcome token.

    fidelity is the sampling interval in minutes.
    """
    params = {"market": token_id, "interval": interval, "fidelity": fidelity}
    payload = _get(f"{CLOB}/prices-history", params=params)
    history = payload.get("history", []) if isinstance(payload, dict) else []
    df = pd.DataFrame(history)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.rename(columns={"p": "price"}).drop(columns=["t"])
    df["token_id"] = token_id
    return df[["timestamp", "price", "token_id"]]


OFFSET_CAP_HINT = 3000


def _paginate_trades(condition_id: str, extra: dict, page_size: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        params = {"market": condition_id, "limit": page_size, "offset": offset, **extra}
        try:
            batch = _get(f"{DATA}/trades", params=params)
        except requests.HTTPError as e:
            print(f"    stop offset={offset} params={extra}: {e}")
            break
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def fetch_trades(condition_id: str, page_size: int = 500) -> pd.DataFrame:
    """Paginated trade history, keyed on condition_id.

    The Data API caps offset at ~3000. If the unfiltered pull hits that cap,
    fall back to side-split pulls (BUY and SELL) so each partition gets its
    own offset budget. Duplicates across passes are dropped.
    """
    rows = _paginate_trades(condition_id, {}, page_size)
    if len(rows) >= OFFSET_CAP_HINT:
        for side in ("BUY", "SELL"):
            rows.extend(_paginate_trades(condition_id, {"side": side}, page_size))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    dedup_on = [
        c
        for c in ("transactionHash", "asset", "side", "size", "price", "timestamp")
        if c in df.columns
    ]
    if dedup_on:
        df = df.drop_duplicates(subset=dedup_on).reset_index(drop=True)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["condition_id"] = condition_id
    return df


LOCK_THRESHOLD = 0.995
LOCK_UNLOCK_FLOOR = 0.9


def _first_lock_timestamp(
    series: pd.DataFrame, price_col: str, ts_col: str
) -> pd.Timestamp | None:
    """Earliest timestamp at which `price_col` reaches LOCK_THRESHOLD and the
    MEDIAN of subsequent prices remains at or above LOCK_UNLOCK_FLOOR.

    Using the median instead of `.any()` tolerates the small fraction of
    outlier trades (misclicked orders, late limit-order fills) that real
    Polymarket markets exhibit after effective resolution — while still
    rejecting false locks where the price spikes to threshold briefly and
    then trades back down (e.g. an intraday news reaction that later reverses).

    Iterates candidate lock points from earliest to latest and returns the
    first whose post-lock median holds above the floor. Returns None if no
    candidate satisfies the median condition.
    """
    if series.empty:
        return None
    s = series.sort_values(ts_col).reset_index(drop=True)
    hi_mask = s[price_col] >= LOCK_THRESHOLD
    if not hi_mask.any():
        return None
    prices = s[price_col].to_numpy()
    for idx in np.flatnonzero(hi_mask.to_numpy()):
        subsequent = prices[idx:]
        if np.median(subsequent) >= LOCK_UNLOCK_FLOOR:
            return s.iloc[idx][ts_col]
    return None


def derive_resolution_timestamps(
    prices: pd.DataFrame, markets: pd.DataFrame, trades: pd.DataFrame | None = None
) -> dict[str, pd.Timestamp]:
    """Earliest timestamp at which the winning token's price first locks to
    LOCK_THRESHOLD and stays above LOCK_UNLOCK_FLOOR thereafter.

    Primary source is the CLOB mid-price history. When CLOB history is absent
    for a market (common once a market has been resolved for a while), falls
    back to the trade-execution price series on the winning token.

    Replaces the Gamma `endDate`, which is a scheduled end, not the actual
    resolution moment.
    """
    p_all = pd.DataFrame(columns=["timestamp", "price", "token_id"])
    if not prices.empty:
        p_all = prices[["timestamp", "price", "token_id"]].copy()
        p_all["token_id"] = p_all["token_id"].astype(str)
        p_all["price"] = pd.to_numeric(p_all["price"], errors="coerce")
        p_all["timestamp"] = pd.to_datetime(p_all["timestamp"], utc=True)

    t_all = pd.DataFrame(columns=["timestamp", "price", "asset"])
    if (
        trades is not None
        and not trades.empty
        and {"asset", "price", "timestamp"}.issubset(trades.columns)
    ):
        t_all = trades[["timestamp", "price", "asset"]].copy()
        t_all["asset"] = t_all["asset"].astype(str)
        t_all["price"] = pd.to_numeric(t_all["price"], errors="coerce")
        t_all["timestamp"] = pd.to_datetime(t_all["timestamp"], utc=True)

    out: dict[str, pd.Timestamp] = {}
    for _, m in markets.iterrows():
        if not bool(m.get("resolved")):
            continue
        win_idx = m.get("winning_outcome_index")
        if win_idx is None or pd.isna(win_idx):
            continue
        tokens = str(m.get("token_ids") or "").split(";")
        try:
            win_token = tokens[int(win_idx)]
        except (IndexError, ValueError):
            continue

        clob_slice = p_all[p_all["token_id"] == win_token]
        ts = _first_lock_timestamp(clob_slice, "price", "timestamp")
        if ts is None:
            trade_slice = t_all[t_all["asset"] == win_token]
            ts = _first_lock_timestamp(trade_slice, "price", "timestamp")
        if ts is not None:
            out[str(m["condition_id"])] = ts
    return out


def _add_running_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-market cumulative and rolling features, strictly prior to each trade."""
    df = df.sort_values(["condition_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    g = df.groupby("condition_id", sort=False)
    df["market_trade_count_so_far"] = g.cumcount()
    tv = pd.to_numeric(df["trade_value_usd"], errors="coerce").fillna(0.0)
    df["market_volume_so_far_usd"] = tv.groupby(df["condition_id"]).cumsum() - tv

    mp = pd.to_numeric(df["market_implied_prob"], errors="coerce")
    vol = pd.Series(float("nan"), index=df.index, dtype="float64")
    vol_1h = pd.Series(0.0, index=df.index, dtype="float64")
    vol_24h = pd.Series(0.0, index=df.index, dtype="float64")
    for _, gdf in df.groupby("condition_id", sort=False):
        order = gdf["timestamp"].argsort(kind="mergesort")
        idx_sorted = gdf.index.to_numpy()[order.to_numpy()]
        ts_sorted = gdf["timestamp"].to_numpy()[order.to_numpy()]
        mp_series = pd.Series(
            mp.loc[idx_sorted].to_numpy(), index=pd.DatetimeIndex(ts_sorted)
        )
        vol.loc[idx_sorted] = mp_series.rolling("1h", closed="left").std().to_numpy()
        tv_series = pd.Series(
            tv.loc[idx_sorted].to_numpy(), index=pd.DatetimeIndex(ts_sorted)
        )
        vol_1h.loc[idx_sorted] = tv_series.rolling("1h", closed="left").sum().to_numpy()
        vol_24h.loc[idx_sorted] = (
            tv_series.rolling("24h", closed="left").sum().to_numpy()
        )
    df["market_price_vol_last_1h"] = vol
    df["market_vol_1h_log"] = np.log1p(vol_1h.fillna(0.0))
    df["market_vol_24h_log"] = np.log1p(vol_24h.fillna(0.0))

    # Contextual market-state control: share of BUY trades in the market so
    # far, strictly before t. Bounded in [0, 1]; informed-flow pressure shifts
    # this away from 0.5 as informed wallets concentrate on one side.
    if "side" in df.columns:
        is_buy = df["side"].astype(str).str.upper().eq("BUY").astype("int64")
        prior_buys = is_buy.groupby(df["condition_id"]).cumsum() - is_buy
        n_prior = df["market_trade_count_so_far"]
        df["market_buy_share_running"] = prior_buys / n_prior.where(n_prior > 0)
    return df


def _add_running_wallet_features(df: pd.DataFrame, wallet_col: str) -> pd.DataFrame:
    """Per-wallet cumulative stats, strictly prior to each trade."""
    df = df.sort_values([wallet_col, "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    gw = df.groupby(wallet_col, sort=False)
    df["wallet_prior_trades"] = gw.cumcount()
    tv = pd.to_numeric(df["trade_value_usd"], errors="coerce").fillna(0.0)
    df["wallet_prior_volume_usd"] = tv.groupby(df[wallet_col]).cumsum() - tv

    bc = pd.to_numeric(df["bet_correct"], errors="coerce")
    bc_filled = bc.fillna(0.0)
    prior_wins = bc_filled.groupby(df[wallet_col]).cumsum() - bc_filled
    labeled = bc.notna().astype("int64")
    prior_labeled = labeled.groupby(df[wallet_col]).cumsum() - labeled
    df["wallet_prior_win_rate"] = prior_wins / prior_labeled.where(prior_labeled > 0)
    return df


def enrich_trades(
    trades: pd.DataFrame,
    markets: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Merge trades with market metadata and prices, then add derived features.

    Adds: settlement_minus_trade_sec (using true CLOB resolution timestamp where
    available, Gamma endDate as fallback), wallet_first_minus_trade_sec,
    trade_value_usd, market_implied_prob (no-lookahead), bet_correct (target),
    running market features (volume / trade count / 1h price volatility so far),
    and running wallet features (prior trades / volume / win rate).
    """
    res_ts = derive_resolution_timestamps(prices, markets, trades)

    meta_cols = [
        "condition_id",
        "slug",
        "question",
        "end_date",
        "winning_outcome_index",
        "resolved",
    ]
    if "outcomes" in markets.columns:
        meta_cols.append("outcomes")
    meta = markets[meta_cols].copy()
    meta["end_date"] = pd.to_datetime(meta["end_date"], utc=True, errors="coerce")
    meta["resolution_ts"] = meta["condition_id"].astype(str).map(res_ts)
    meta["resolution_ts"] = pd.to_datetime(
        meta["resolution_ts"], utc=True, errors="coerce"
    )
    # deadline_ts — the market's *advertised* deadline as stated in the
    # question (e.g. "by February 27, 2026"). Polymarket's Gamma `endDate`
    # field is stale or a batch-placeholder for many of the Iran strike
    # markets (34 / 74 carry end_date = 2026-01-31), so time-to-settlement
    # features use this parsed deadline instead. Fallback to end_date when
    # the question cannot be parsed. Known at market creation, so strictly
    # causal.
    fallback_year = meta["end_date"].dt.year.where(meta["end_date"].notna()).astype(
        "Int64"
    )
    meta["deadline_ts"] = [
        parse_deadline_from_question(q, fy if pd.notna(fy) else None)
        for q, fy in zip(meta["question"], fallback_year)
    ]
    meta["deadline_ts"] = pd.to_datetime(
        meta["deadline_ts"], utc=True, errors="coerce"
    ).fillna(meta["end_date"])
    df = trades.merge(meta, on="condition_id", how="left")

    eff_end = df["resolution_ts"].fillna(df["end_date"])
    df["settlement_minus_trade_sec"] = (eff_end - df["timestamp"]).dt.total_seconds()

    wallet_col = next(
        (c for c in ("proxyWallet", "user", "maker", "taker") if c in df.columns), None
    )
    if wallet_col is not None:
        first_seen = df.groupby(wallet_col)["timestamp"].transform("min")
        df["wallet_first_minus_trade_sec"] = (
            first_seen - df["timestamp"]
        ).dt.total_seconds()
    else:
        df["wallet_first_minus_trade_sec"] = pd.NA

    if "size" in df.columns and "price" in df.columns:
        size_num = pd.to_numeric(df["size"], errors="coerce")
        price_num = pd.to_numeric(df["price"], errors="coerce")
        df["trade_value_usd"] = size_num * price_num
    else:
        df["trade_value_usd"] = pd.NA

    if {"outcomeIndex", "side"}.issubset(df.columns):
        idx = pd.to_numeric(df["outcomeIndex"], errors="coerce")
        win = pd.to_numeric(df["winning_outcome_index"], errors="coerce")
        side_buy = df["side"].astype(str).str.upper() == "BUY"
        outcome_won = idx == win
        df["bet_correct"] = (outcome_won == side_buy).astype("Int64").mask(win.isna())
    else:
        df["bet_correct"] = pd.NA

    # `is_yes` — per-market YES/NO label derived from the outcomes array.
    # Polymarket markets store outcome order per-market (["Yes","No"] or ["No","Yes"]),
    # so a global outcomeIndex→YES/NO mapping doesn't exist. We resolve it here:
    # label of the winning outcome token == "Yes" ⇒ is_yes=1, else 0.
    if "outcomes" in df.columns:
        outs = df["outcomes"].fillna("").astype(str).str.split(";")
        win_idx = pd.to_numeric(df["winning_outcome_index"], errors="coerce")
        is_yes = []
        for o, w in zip(outs, win_idx):
            if pd.isna(w) or int(w) >= len(o):
                is_yes.append(pd.NA)
            else:
                label = str(o[int(w)]).strip().lower()
                is_yes.append(1 if label in {"yes", "true"} else 0)
        df["is_yes"] = pd.array(is_yes, dtype="Int64")
    else:
        df["is_yes"] = pd.NA

    if not prices.empty and "asset" in df.columns:
        pm = prices[["timestamp", "price", "token_id"]].rename(
            columns={"price": "market_implied_prob", "token_id": "asset"}
        )
        df["asset"] = df["asset"].astype(str)
        pm["asset"] = pm["asset"].astype(str)
        # pandas 3.0 returns datetime64[s, UTC] from pd.to_datetime(..., unit="s");
        # merge_asof rejects mixed precision on the sort key, so normalise both
        # sides to ns here.
        df["timestamp"] = df["timestamp"].astype("datetime64[ns, UTC]")
        pm["timestamp"] = pm["timestamp"].astype("datetime64[ns, UTC]")
        df = df.sort_values("timestamp").reset_index(drop=True)
        pm = pm.sort_values("timestamp").reset_index(drop=True)
        df = pd.merge_asof(
            df,
            pm,
            on="timestamp",
            by="asset",
            direction="backward",
            allow_exact_matches=False,
        )
    else:
        df["market_implied_prob"] = pd.NA

    trade_price = (
        pd.to_numeric(df.get("price"), errors="coerce")
        if "price" in df.columns
        else pd.Series(pd.NA, index=df.index)
    )
    df["market_implied_prob"] = pd.to_numeric(
        df["market_implied_prob"], errors="coerce"
    ).fillna(trade_price)

    df = _add_running_market_features(df)
    if wallet_col is not None:
        df = _add_running_wallet_features(df, wallet_col)

    df = expand_features(df, wallet_col or "proxyWallet")
    df = add_timestamp_split(df)

    return df


BURST_WINDOW = "600s"
BURST_K = 3
WHALE_QUANTILE = 0.95
SPLIT_QUANTILES = (0.70, 0.85)


# Canonical feature groupings for downstream feature-importance plots and
# the methodology narrative. Keeps cluster membership in one place so the
# plot stays readable as the feature count grows.
FEATURE_CLUSTERS: dict[str, list[str]] = {
    "trade_local": [
        "log_size",
        "side",
        "outcomeIndex",
        "trade_value_usd",
    ],
    "market_context": [
        "market_trade_count_so_far",
        "market_volume_so_far_usd",
        "market_price_vol_last_1h",
        "market_vol_1h_log",
        "market_vol_24h_log",
        "market_buy_share_running",
    ],
    "time": [
        "time_to_settlement_s",
        "log_time_to_settlement",
        "pct_time_elapsed",
    ],
    "wallet_global": [
        "wallet_prior_trades",
        "wallet_prior_volume_usd",
        "wallet_prior_win_rate",
        "wallet_first_minus_trade_sec",
    ],
    "bet_slicing": [
        "wallet_trades_in_market_last_1min",
        "wallet_trades_in_market_last_10min",
        "wallet_trades_in_market_last_60min",
        "wallet_is_burst",
        "wallet_prior_trades_in_market",
        "wallet_cumvol_same_side_last_10min",
        "wallet_median_gap_in_market",
    ],
    "spread_builder": [
        "wallet_directional_purity_in_market",
        "wallet_has_both_sides_in_market",
        "wallet_spread_ratio",
    ],
    "whale_exit": [
        "wallet_position_size_before_trade",
        "trade_size_vs_position_pct",
        "is_position_exit",
        "is_position_flip",
        "wallet_is_whale_in_market",
    ],
    "interactions": [
        "size_vs_wallet_avg",
        "size_x_time_to_settlement",
        "size_vs_market_cumvol_pct",
        "size_vs_market_avg",
    ],
}


def _rolling_count_by_group(
    df: pd.DataFrame, group_cols: list[str], window: str
) -> pd.Series:
    """Per-group rolling count of prior trades within `window` ending at each
    timestamp, excluding the current row (closed='left').
    """
    out = pd.Series(0, index=df.index, dtype="float64")
    for _, gdf in df.groupby(group_cols, sort=False):
        if len(gdf) == 1:
            out.loc[gdf.index] = 0.0
            continue
        order = gdf["timestamp"].argsort(kind="mergesort").to_numpy()
        idx_sorted = gdf.index.to_numpy()[order]
        ts_sorted = gdf["timestamp"].to_numpy()[order]
        s = pd.Series(1.0, index=pd.DatetimeIndex(ts_sorted))
        rolled = s.rolling(window, closed="left").count()
        out.loc[idx_sorted] = rolled.to_numpy()
    return out.fillna(0).astype("int64")


def _causal_whale_flag(
    df: pd.DataFrame, wallet_col: str, tv: pd.Series, quantile: float
) -> pd.Series:
    """Expanding-p95 whale flag, strictly causal.

    At each trade i in market m at time t, let S_{m,t} be the multiset of
    per-wallet running USD volumes in m, using only trades with timestamp
    strictly less than t (one entry per wallet that has traded in m so far).
    The threshold τ_{m,t} is the `quantile`-th percentile of S_{m,t} with
    numpy's default linear interpolation. The flag is 1 iff the current
    wallet's own running volume (excluding the current trade) is ≥ τ_{m,t}.

    Implementation: per market, walk trades in timestamp order while
    maintaining a dict of running wallet volumes and a SortedList of those
    volumes. O(log W) per update. A minimum-sample warm-up of
    `ceil(1 / (1 - quantile))` wallets is required before any flag can fire
    so that the top percentile is well-defined (no flag = 1 on a two-wallet
    market).
    """
    from math import ceil
    from sortedcontainers import SortedList

    min_wallets = max(2, ceil(1.0 / max(1e-9, 1.0 - quantile)))
    n = len(df)
    flag = np.zeros(n, dtype=np.int8)

    tv_arr = tv.to_numpy(dtype=np.float64)
    wallet_arr = df[wallet_col].to_numpy()
    ts_vals = pd.to_datetime(df["timestamp"], utc=True).astype("int64").to_numpy()

    for cid, gdf in df.groupby("condition_id", sort=False):
        order = np.argsort(ts_vals[gdf.index.to_numpy()], kind="mergesort")
        idx_sorted = gdf.index.to_numpy()[order]

        running: dict[str, float] = {}
        sl = SortedList()

        for pos in range(idx_sorted.shape[0]):
            i = int(idx_sorted[pos])
            w = wallet_arr[i]
            prev = running.get(w, 0.0)

            if len(sl) >= min_wallets:
                k = quantile * (len(sl) - 1)
                k_lo = int(np.floor(k))
                k_hi = int(np.ceil(k))
                if k_lo == k_hi:
                    tau = sl[k_lo]
                else:
                    frac = k - k_lo
                    tau = sl[k_lo] * (1.0 - frac) + sl[k_hi] * frac
                if prev >= tau:
                    flag[i] = 1

            new_val = prev + float(tv_arr[i])
            if w in running:
                sl.remove(running[w])
            sl.add(new_val)
            running[w] = new_val

    return pd.Series(flag, index=df.index, dtype="int8")


def _rolling_sum_by_group(
    df: pd.DataFrame, group_cols: list[str], value_col: str, window: str
) -> pd.Series:
    """Per-group rolling sum of `value_col` within `window` ending at each
    timestamp, excluding the current row (closed='left'). Strictly no-lookahead.
    """
    vals = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    out = pd.Series(0.0, index=df.index, dtype="float64")
    for _, gdf in df.groupby(group_cols, sort=False):
        if len(gdf) == 1:
            out.loc[gdf.index] = 0.0
            continue
        order = gdf["timestamp"].argsort(kind="mergesort").to_numpy()
        idx_sorted = gdf.index.to_numpy()[order]
        ts_sorted = gdf["timestamp"].to_numpy()[order]
        s = pd.Series(
            vals.loc[idx_sorted].to_numpy(), index=pd.DatetimeIndex(ts_sorted)
        )
        rolled = s.rolling(window, closed="left").sum()
        out.loc[idx_sorted] = rolled.to_numpy()
    return out.fillna(0.0)


def expand_features(df: pd.DataFrame, wallet_col: str = "proxyWallet") -> pd.DataFrame:
    """Add the six-layer feature taxonomy on top of `enrich_trades`'s output.

    Strictly no-lookahead: every feature at row t uses only rows with
    timestamp strictly before t within the relevant group.

    Adds:
      trade-local: log_size
      time: time_to_settlement_s, log_time_to_settlement, pct_time_elapsed
      wallet-in-market bursting: wallet_trades_in_market_last_{1,10,60}min,
        wallet_is_burst
      wallet-in-market directional: wallet_directional_purity_in_market,
        wallet_has_both_sides_in_market, wallet_spread_ratio
      wallet-in-market position-aware: wallet_position_size_before_trade,
        trade_size_vs_position_pct, is_position_exit, is_position_flip,
        wallet_is_whale_in_market
      wallet-in-market depth: wallet_prior_trades_in_market,
        wallet_cumvol_same_side_last_10min
      interactions: size_vs_wallet_avg, size_x_time_to_settlement,
        size_vs_market_cumvol_pct
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    size_num = pd.to_numeric(df.get("size"), errors="coerce")
    tv = pd.to_numeric(df.get("trade_value_usd"), errors="coerce").fillna(0.0)

    # --- Trade-local ---
    df["log_size"] = np.log1p(size_num.clip(lower=0).fillna(0))

    # --- Time features ---
    # time_to_settlement_s and pct_time_elapsed use `deadline_ts`, the
    # deadline parsed out of the question string ("by February 27, 2026").
    # `deadline_ts` is authored in the market's own question at creation
    # time and is therefore provably known at trade time — strictly causal.
    # Two weaker candidates were explicitly rejected:
    #   * `resolution_ts` (post-hoc CLOB lock) — leaks future info.
    #   * `end_date` (Gamma `endDate`) — stale or batched to a placeholder
    #     (2026-01-31) on 34 / 74 markets, producing negative time-to-
    #     settlement on ~44 % of rows.
    # `settlement_minus_trade_sec` keeps `resolution_ts` upstream because
    # its sole role is the §4 post-resolution filter; it is not a model
    # feature.
    deadline = pd.to_datetime(df.get("deadline_ts"), utc=True, errors="coerce")
    df["time_to_settlement_s"] = (deadline - df["timestamp"]).dt.total_seconds()
    df["log_time_to_settlement"] = np.log1p(
        df["time_to_settlement_s"].clip(lower=0).fillna(0)
    )

    market_start = df.groupby("condition_id")["timestamp"].transform("min")
    life_total = (deadline - market_start).dt.total_seconds()
    life_elapsed = (df["timestamp"] - market_start).dt.total_seconds()
    df["pct_time_elapsed"] = (life_elapsed / life_total.where(life_total > 0)).clip(
        0, 1
    )

    # --- Wallet-in-market bursting ---
    for label, window in [("1min", "60s"), ("10min", "600s"), ("60min", "3600s")]:
        df[f"wallet_trades_in_market_last_{label}"] = _rolling_count_by_group(
            df, [wallet_col, "condition_id"], window
        )
    df["wallet_is_burst"] = (
        df[f"wallet_trades_in_market_last_{'10min'}"] >= BURST_K
    ).astype("int64")

    # Median gap between consecutive trades within (wallet, market), over all
    # pairs strictly before t. Expanding median over .diff() gaps, shifted by 1
    # within group so row k uses only gaps from pairs fully before k.
    gaps_sec = (
        df.groupby([wallet_col, "condition_id"])["timestamp"].diff().dt.total_seconds()
    )
    df["wallet_median_gap_in_market"] = gaps_sec.groupby(
        [df[wallet_col], df["condition_id"]]
    ).transform(lambda s: s.expanding().median().shift(1))

    # --- Wallet-in-market directional purity ---
    oi = pd.to_numeric(df["outcomeIndex"], errors="coerce")
    is0 = (oi == 0).astype("int64")
    is1 = (oi == 1).astype("int64")
    g_wm = df.groupby([wallet_col, "condition_id"], sort=False)
    cum0 = is0.groupby([df[wallet_col], df["condition_id"]]).cumsum() - is0
    cum1 = is1.groupby([df[wallet_col], df["condition_id"]]).cumsum() - is1
    total = cum0 + cum1
    p0 = cum0 / total.where(total > 0)
    p1 = cum1 / total.where(total > 0)
    df["wallet_directional_purity_in_market"] = (p0**2 + p1**2).where(total > 0)
    df["wallet_has_both_sides_in_market"] = ((cum0 > 0) & (cum1 > 0)).astype("int64")
    max_c = pd.concat([cum0, cum1], axis=1).max(axis=1)
    min_c = pd.concat([cum0, cum1], axis=1).min(axis=1)
    df["wallet_spread_ratio"] = (min_c / max_c.where(max_c > 0)).where(total > 0)

    # --- Wallet-in-market position-aware (per outcome token) ---
    side_upper = df["side"].astype(str).str.upper()
    signed = np.where(side_upper == "BUY", size_num.fillna(0), -size_num.fillna(0))
    signed_s = pd.Series(signed, index=df.index, dtype="float64")
    pos_grp_keys = [df[wallet_col], df["condition_id"], oi]
    pos_before = signed_s.groupby(pos_grp_keys).cumsum() - signed_s
    df["wallet_position_size_before_trade"] = pos_before

    abs_pos = pos_before.abs()
    denom = np.maximum(abs_pos.to_numpy(), size_num.fillna(0).to_numpy())
    ratio = pd.Series(
        np.where(denom > 0, size_num.fillna(0).to_numpy() / denom, 0.0),
        index=df.index,
        dtype="float64",
    ).clip(0, 1)
    df["trade_size_vs_position_pct"] = ratio
    df["is_position_exit"] = ((side_upper == "SELL") & (ratio >= 0.9)).astype("int64")
    new_pos = pos_before + signed_s
    df["is_position_flip"] = (
        (np.sign(pos_before.fillna(0)) != 0)
        & (np.sign(pos_before.fillna(0)) != np.sign(new_pos.fillna(0)))
        & (np.sign(new_pos.fillna(0)) != 0)
    ).astype("int64")

    # Whale flag — strictly causal expanding-p95 threshold per market. See
    # `_causal_whale_flag` docstring. The earlier implementation used the p95
    # of end-of-market wallet totals, which leaked future state (a wallet's
    # running volume was compared to a threshold computed from all future
    # trades in the market).
    df["wallet_is_whale_in_market"] = _causal_whale_flag(
        df, wallet_col, tv, WHALE_QUANTILE
    ).astype("int64")

    # --- Wallet-in-market depth ---
    # Prior trade count in THIS market (cumulative, strictly prior).
    one = pd.Series(1, index=df.index, dtype="int64")
    df["wallet_prior_trades_in_market"] = (
        one.groupby([df[wallet_col], df["condition_id"]]).cumsum() - one
    )

    # Same-side rolling USD volume in the last 10 minutes.
    df["wallet_cumvol_same_side_last_10min"] = _rolling_sum_by_group(
        df,
        [wallet_col, "condition_id", "outcomeIndex"],
        "trade_value_usd",
        "600s",
    )

    # --- Interactions ---
    wpt = pd.to_numeric(df.get("wallet_prior_trades"), errors="coerce")
    wpv = pd.to_numeric(df.get("wallet_prior_volume_usd"), errors="coerce")
    wallet_avg_prior = wpv / wpt.where(wpt > 0)
    df["size_vs_wallet_avg"] = tv / wallet_avg_prior.where(wallet_avg_prior > 0)
    df["size_x_time_to_settlement"] = df["log_size"] * df["log_time_to_settlement"]

    # Trade size as a fraction of prior market cumulative volume. Large values
    # mark abnormally big bets relative to the market's activity so far.
    mv_prior = pd.to_numeric(df.get("market_volume_so_far_usd"), errors="coerce")
    df["size_vs_market_cumvol_pct"] = tv / mv_prior.where(mv_prior > 0)

    # Trade size relative to the running mean trade size in the market so far
    # (strictly prior). Complements size_vs_market_cumvol_pct: the fraction
    # version captures absolute market dominance, this one captures
    # "big-relative-to-typical-trade."
    mc_prior = pd.to_numeric(df.get("market_trade_count_so_far"), errors="coerce")
    avg_size_market = mv_prior / mc_prior.where(mc_prior > 0)
    df["size_vs_market_avg"] = tv / avg_size_market.where(avg_size_market > 0)

    return df


def add_timestamp_split(
    df: pd.DataFrame, quantiles: tuple[float, float] = SPLIT_QUANTILES
) -> pd.DataFrame:
    """Assign each trade to train / val / test based on quantiles of its
    execution timestamp. Default 0.70 / 0.85 split.
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True).astype("int64")
    q_lo, q_hi = ts.quantile(quantiles[0]), ts.quantile(quantiles[1])
    df["split"] = np.where(ts < q_lo, "train", np.where(ts < q_hi, "val", "test"))
    return df


def run(event_ids: Iterable[str] = TARGET_EVENT_IDS) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_markets: list[MarketRef] = []
    for eid in event_ids:
        print(f"[event] {eid}")
        event = fetch_event(eid)
        markets = parse_markets(event)
        all_markets.extend(markets)
        print(f"  found {len(markets)} markets")

    meta = pd.DataFrame(
        [
            {
                "event_id": m.event_id,
                "market_id": m.market_id,
                "condition_id": m.condition_id,
                "slug": m.slug,
                "question": m.question,
                "outcomes": ";".join(m.outcomes),
                "token_ids": ";".join(m.token_ids),
                "closed": m.closed,
                "resolved": m.resolved,
                "winning_outcome_index": m.winning_outcome_index,
                "end_date": m.end_date,
            }
            for m in all_markets
        ]
    )
    meta.to_csv(OUT_DIR / "01_markets_meta.csv", index=False)
    n_resolved = int(meta["resolved"].sum())
    print(
        f"[meta] wrote {len(meta)} rows -> 01_markets_meta.csv ({n_resolved} resolved)"
    )

    price_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for m in all_markets:
        if not m.resolved:
            continue
        print(f"[prices] {m.slug}")
        for tid in m.token_ids:
            try:
                price_frames.append(fetch_price_history(tid))
            except requests.HTTPError as e:
                print(f"  price error {tid}: {e}")
        print(f"[trades] {m.slug}")
        try:
            trade_frames.append(fetch_trades(m.condition_id))
        except requests.HTTPError as e:
            print(f"  trade error {m.condition_id}: {e}")

    prices = pd.DataFrame()
    if price_frames:
        prices = pd.concat([f for f in price_frames if not f.empty], ignore_index=True)
        prices.to_csv(OUT_DIR / "01_prices.csv", index=False)
        print(f"[prices] wrote {len(prices)} rows -> 01_prices.csv")
    if trade_frames:
        trades = pd.concat([f for f in trade_frames if not f.empty], ignore_index=True)
        trades.to_csv(OUT_DIR / "02_trades.csv", index=False)
        print(f"[trades] wrote {len(trades)} rows -> 02_trades.csv")

        enriched = enrich_trades(trades, meta, prices)
        enriched.to_csv(OUT_DIR / "03_consolidated_dataset.csv", index=False)
        n_labeled = int(enriched["bet_correct"].notna().sum())
        n_priced = (
            int(enriched["market_implied_prob"].notna().sum())
            if "market_implied_prob" in enriched.columns
            else 0
        )
        print(
            f"[enriched] wrote {len(enriched)} rows -> 03_consolidated_dataset.csv "
            f"({n_labeled} labeled, {n_priced} with implied prob)"
        )


if __name__ == "__main__":
    run()
