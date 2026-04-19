"""Polymarket data fetcher for the Iran prediction markets project.

Pulls three layers of data from Polymarket's public APIs:
    1. Event and market metadata            (Gamma API)
    2. Intraday price history per outcome   (CLOB API)
    3. Trade-level history per market       (Data API)

Outputs CSV files into ./data/ next to this script.

Targets: events 114242, 236884, 355299, 357625 (Iran strikes, Iran-Israel/US
conflict end, Trump ceasefire announcement, ceasefire extensions).
Run:     python fetch_polymarket.py

Known limitation: the Data API caps pagination offset at ~3000. Side-split
fallback lifts the ceiling to ~7000 trades per market. Markets with more
trades lose their earliest activity (recency bias). Breaking this cap would
require moving to the Polygon on-chain subgraph or Polygonscan, neither done
here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"

TARGET_EVENT_IDS = ["114242", "236884", "355299", "357625"]

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "data"

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


def fetch_price_history(token_id: str, interval: str = "all", fidelity: int = 1) -> pd.DataFrame:
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

    dedup_on = [c for c in ("transactionHash", "asset", "side", "size", "price", "timestamp") if c in df.columns]
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
    """First timestamp at which `price_col` >= LOCK_THRESHOLD and does not fall
    back below LOCK_UNLOCK_FLOOR for the remainder of the series. Returns
    None if no such lock exists.
    """
    if series.empty:
        return None
    s = series.sort_values(ts_col)
    locked = s[s[price_col] >= LOCK_THRESHOLD]
    if locked.empty:
        return None
    first_ts = locked.iloc[0][ts_col]
    after = s[s[ts_col] >= first_ts]
    if (after[price_col] < LOCK_UNLOCK_FLOOR).any():
        return None
    return first_ts


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
    if trades is not None and not trades.empty and {"asset", "price", "timestamp"}.issubset(trades.columns):
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
    df = df.sort_values(["condition_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("condition_id", sort=False)
    df["market_trade_count_so_far"] = g.cumcount()
    tv = pd.to_numeric(df["trade_value_usd"], errors="coerce").fillna(0.0)
    df["market_volume_so_far_usd"] = tv.groupby(df["condition_id"]).cumsum() - tv

    mp = pd.to_numeric(df["market_implied_prob"], errors="coerce")
    vol = pd.Series(float("nan"), index=df.index, dtype="float64")
    for _, gdf in df.groupby("condition_id", sort=False):
        order = gdf["timestamp"].argsort(kind="mergesort")
        idx_sorted = gdf.index.to_numpy()[order.to_numpy()]
        ts_sorted = gdf["timestamp"].to_numpy()[order.to_numpy()]
        s = pd.Series(mp.loc[idx_sorted].to_numpy(), index=pd.DatetimeIndex(ts_sorted))
        rolled = s.rolling("1h", closed="left").std()
        vol.loc[idx_sorted] = rolled.to_numpy()
    df["market_price_vol_last_1h"] = vol
    return df


def _add_running_wallet_features(df: pd.DataFrame, wallet_col: str) -> pd.DataFrame:
    """Per-wallet cumulative stats, strictly prior to each trade."""
    df = df.sort_values([wallet_col, "timestamp"], kind="mergesort").reset_index(drop=True)
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

    meta = markets[
        ["condition_id", "slug", "question", "end_date", "winning_outcome_index", "resolved"]
    ].copy()
    meta["end_date"] = pd.to_datetime(meta["end_date"], utc=True, errors="coerce")
    meta["resolution_ts"] = meta["condition_id"].astype(str).map(res_ts)
    meta["resolution_ts"] = pd.to_datetime(meta["resolution_ts"], utc=True, errors="coerce")
    df = trades.merge(meta, on="condition_id", how="left")

    eff_end = df["resolution_ts"].fillna(df["end_date"])
    df["settlement_minus_trade_sec"] = (eff_end - df["timestamp"]).dt.total_seconds()

    wallet_col = next((c for c in ("proxyWallet", "user", "maker", "taker") if c in df.columns), None)
    if wallet_col is not None:
        first_seen = df.groupby(wallet_col)["timestamp"].transform("min")
        df["wallet_first_minus_trade_sec"] = (first_seen - df["timestamp"]).dt.total_seconds()
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

    if not prices.empty and "asset" in df.columns:
        pm = prices[["timestamp", "price", "token_id"]].rename(
            columns={"price": "market_implied_prob", "token_id": "asset"}
        )
        df["asset"] = df["asset"].astype(str)
        pm["asset"] = pm["asset"].astype(str)
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

    trade_price = pd.to_numeric(df.get("price"), errors="coerce") if "price" in df.columns else pd.Series(pd.NA, index=df.index)
    df["market_implied_prob"] = pd.to_numeric(df["market_implied_prob"], errors="coerce").fillna(trade_price)

    df = _add_running_market_features(df)
    if wallet_col is not None:
        df = _add_running_wallet_features(df, wallet_col)

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
    meta.to_csv(OUT_DIR / "markets.csv", index=False)
    n_resolved = int(meta["resolved"].sum())
    print(f"[meta] wrote {len(meta)} rows -> markets.csv ({n_resolved} resolved)")

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
        prices.to_csv(OUT_DIR / "prices.csv", index=False)
        print(f"[prices] wrote {len(prices)} rows -> prices.csv")
    if trade_frames:
        trades = pd.concat([f for f in trade_frames if not f.empty], ignore_index=True)
        trades.to_csv(OUT_DIR / "trades.csv", index=False)
        print(f"[trades] wrote {len(trades)} rows -> trades.csv")

        enriched = enrich_trades(trades, meta, prices)
        enriched.to_csv(OUT_DIR / "trades_enriched.csv", index=False)
        n_labeled = int(enriched["bet_correct"].notna().sum())
        n_priced = int(enriched["market_implied_prob"].notna().sum()) if "market_implied_prob" in enriched.columns else 0
        print(
            f"[enriched] wrote {len(enriched)} rows -> trades_enriched.csv "
            f"({n_labeled} labeled, {n_priced} with implied prob)"
        )


if __name__ == "__main__":
    run()
