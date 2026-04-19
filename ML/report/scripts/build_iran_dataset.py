"""Hybrid builder for the Iran prediction markets dataset.

Combines two trade-level sources so the four target events are covered without
the Polymarket Data API's ~7k-per-market cap restricting any market:

  HF  path  —  67 resolved sub-markets under events 114242 and 236884.
               Full trade history from the HuggingFace mirror
               `SII-WANGZJ/Polymarket_data` (on-chain CTF Exchange events,
               MIT licensed). All 67 markets resolved strictly before the
               HF snapshot cutoff (2026-03-31), so HF carries their complete
               histories with no cap.

  API path  —   7 resolved sub-markets under events 355299 and 357625 (Iran
               ceasefire). These markets were created and resolved after the
               HF cutoff, so HF does not have them. They are small (< 7k trades
               each) so the Data API's offset cap is not a problem.

Pipeline:
  1. Fetch market metadata for all four events via Polymarket Gamma API.
  2. Ensure HF `markets.parquet` is present locally; use its condition_id
     column to route each resolved market to 'hf' or 'api' source.
  3. Build the 67-market HF trade subset:
       - If `data/trades.parquet` (38.7 GB) is present, filter it locally
         via duckdb.
       - Otherwise stream the remote file over HTTPS via duckdb httpfs,
         filter server-side by condition_id, and cache the result as
         `data/trades_iran_subset.parquet` (~hundreds of MB). This keeps the
         on-disk footprint small for users with limited space.
  4. Fetch the 7 ceasefire markets' trades via Polymarket Data API (no cap
     impact: max per-market trade count is ~4.5k).
  5. Normalise schemas, concatenate, write `trades.csv`.
  6. Run `fetch_polymarket.enrich_trades` (+ expand_features + split) and
     write `trades_enriched.csv`.

Run from the project root:
    python ML/report/scripts/build_iran_dataset.py

Flags:
    --skip-download     Fail if parquets are missing instead of downloading.
    --force-download    Re-download parquets even if present locally.
    --dry-run           Print bucket counts and exit without fetching anything.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent
DATA_DIR = REPORT_DIR / "data"

sys.path.insert(0, str(SCRIPT_DIR))
import fetch_polymarket as fp  # noqa: E402

HF_REPO = "SII-WANGZJ/Polymarket_data"
HF_TRADES_URL = (
    "https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data/"
    "resolve/main/trades.parquet"
)
HF_EVENT_IDS = {"114242", "236884"}
API_EVENT_IDS = {"355299", "357625"}
ALL_EVENT_IDS = HF_EVENT_IDS | API_EVENT_IDS

# Local cache file containing only the Iran subset of HF trades. This is the
# output of the remote-stream-filter step; all subsequent reads use it.
TRADES_SUBSET_NAME = "trades_iran_subset.parquet"


def log(msg: str, t0: float | None = None) -> float:
    now = time.monotonic()
    stamp = f"[{now - t0:6.1f}s] " if t0 else ""
    print(f"{stamp}{msg}", flush=True)
    return t0 or now


def ensure_parquet(filename: str, skip_download: bool, force: bool) -> Path:
    path = DATA_DIR / filename
    if path.exists() and not force:
        return path
    if skip_download:
        raise SystemExit(
            f"{path} is missing and --skip-download was set. "
            f"Run without --skip-download to download from HuggingFace "
            f"({HF_REPO})."
        )
    from huggingface_hub import hf_hub_download

    log(f"downloading {filename} from HF dataset {HF_REPO}...")
    local = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=filename,
        local_dir=str(DATA_DIR),
    )
    return Path(local)


def _parse_outcome_prices(op) -> tuple[bool, int | None]:
    """HF encodes `outcome_prices` as Python-repr strings like "['0', '1']",
    whereas the Polymarket API uses JSON like '["1","0"]'. Handle both.
    """
    if op is None:
        return False, None
    if isinstance(op, str):
        try:
            prices = ast.literal_eval(op)
        except (ValueError, SyntaxError):
            try:
                prices = json.loads(op)
            except json.JSONDecodeError:
                return False, None
    else:
        try:
            prices = list(op)
        except TypeError:
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


def build_markets_meta(skip_download: bool, force: bool, t0: float) -> pd.DataFrame:
    """Build metadata for all resolved sub-markets across the four target events
    using the Polymarket Gamma API as the authoritative source of resolution
    status. Each row is then tagged with `source='hf'` if its condition_id is
    present in the HF markets.parquet (trades will come from HF) or
    `source='api'` otherwise (trades will come from the Polymarket Data API).
    """
    log("fetching market metadata for all four events via Polymarket Gamma API...", t0)
    refs: list[fp.MarketRef] = []
    for eid in sorted(ALL_EVENT_IDS):
        event = fp.fetch_event(eid)
        refs.extend(fp.parse_markets(event))
    resolved = [r for r in refs if r.resolved]
    log(f"Gamma markets across target events: {len(refs)} total, "
        f"{len(resolved)} resolved", t0)

    markets = pd.DataFrame(
        [
            {
                "event_id": r.event_id,
                "market_id": r.market_id,
                "condition_id": r.condition_id,
                "slug": r.slug,
                "question": r.question,
                "outcomes": ";".join(r.outcomes),
                "token_ids": ";".join(r.token_ids),
                "closed": r.closed,
                "resolved": r.resolved,
                "winning_outcome_index": r.winning_outcome_index,
                "end_date": pd.to_datetime(r.end_date, utc=True, errors="coerce"),
            }
            for r in resolved
        ]
    )

    markets_pq = ensure_parquet("markets.parquet", skip_download, force)
    hf_cids = set(pd.read_parquet(markets_pq, columns=["condition_id"])["condition_id"].astype(str))
    markets["source"] = markets["condition_id"].astype(str).apply(
        lambda c: "hf" if c in hf_cids else "api"
    )
    log(
        f"markets routed:  hf={(markets['source']=='hf').sum()}  "
        f"api={(markets['source']=='api').sum()}  "
        f"(by-event: {markets.groupby('event_id')['source'].value_counts().to_dict()})",
        t0,
    )
    return markets


def build_trades_subset_remote(
    markets: pd.DataFrame, force: bool, t0: float
) -> Path:
    """Stream HF trades.parquet over HTTPS via duckdb's httpfs, filter to the
    HF-covered condition_ids server-side, and write only matching rows to a
    local parquet subset (`data/trades_iran_subset.parquet`).

    Rationale: the full HF trades.parquet is ~38.7 GB, which may not fit on
    disk. The filtered subset is ~200-500 MB. duckdb streams the remote file
    in row groups, reads only required columns, keeps only rows matching the
    WHERE clause in memory, then writes the filtered result locally.

    Network transfer = full 38.7 GB one time (no parquet filter-pushdown on
    condition_id unless the producer wrote row-group stats); local disk usage
    = subset size only.
    """
    subset = DATA_DIR / TRADES_SUBSET_NAME
    if subset.exists() and not force:
        log(f"subset cache present: {subset.name} "
            f"({subset.stat().st_size/1e6:,.1f} MB); reusing", t0)
        return subset

    import duckdb

    hf_mask = markets["source"] == "hf"
    hf_cids = markets.loc[hf_mask, "condition_id"].astype(str).tolist()
    log(f"streaming {HF_TRADES_URL} via duckdb httpfs; "
        f"filtering to {len(hf_cids)} condition_ids; "
        f"writing {subset.name}", t0)
    log("  network transfer ~38.7 GB one-time; local disk footprint = subset only", t0)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar = true;")
    con.execute(
        """
        COPY (
            SELECT timestamp, block_number, transaction_hash, condition_id,
                   maker, taker, taker_direction, maker_direction,
                   nonusdc_side, price, usd_amount, token_amount
            FROM read_parquet($url)
            WHERE CAST(condition_id AS VARCHAR) IN (SELECT UNNEST(CAST($cids AS VARCHAR[])))
        ) TO $out (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        {"url": HF_TRADES_URL, "cids": hf_cids, "out": str(subset)},
    )
    con.close()

    size_mb = subset.stat().st_size / 1e6
    log(f"subset written: {subset.name}  ({size_mb:,.1f} MB)", t0)
    return subset


def build_hf_trades(
    markets: pd.DataFrame, skip_download: bool, force: bool, t0: float
) -> pd.DataFrame:
    """Load the local HF trades subset (streaming it in from HF if needed),
    then map the HF schema to our canonical trade schema.

    HF trade schema (per Alex's pipeline):
        timestamp, block_number, transaction_hash, condition_id,
        maker, taker, taker_direction, maker_direction,
        nonusdc_side, price, usd_amount, token_amount

    If `data/trades.parquet` (the full 38.7 GB file) is present locally, it is
    used directly. Otherwise, the remote-stream-filter path is used, which
    writes `data/trades_iran_subset.parquet` and reads from that. The subset
    becomes the on-disk cache for future runs.
    """
    import duckdb

    full_local = DATA_DIR / "trades.parquet"
    if full_local.exists() and not force:
        log(f"full trades.parquet present locally ({full_local.stat().st_size/1e9:,.1f} GB); "
            f"filtering via duckdb", t0)
        source_path = full_local
    else:
        if skip_download:
            raise SystemExit(
                f"{full_local} and {DATA_DIR / TRADES_SUBSET_NAME} both missing and "
                f"--skip-download was set. Either download trades.parquet manually "
                f"or run without --skip-download to stream-filter from HF."
            )
        source_path = build_trades_subset_remote(markets, force, t0)

    hf_mask = markets["source"] == "hf"
    hf_cids = markets.loc[hf_mask, "condition_id"].astype(str).tolist()

    con = duckdb.connect()
    try:
        cols = con.execute(
            "SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet($p) LIMIT 0)",
            {"p": str(source_path)},
        ).fetchall()
        log(f"trades parquet columns: {[c[0] for c in cols]}", t0)
    except Exception as e:
        log(f"schema probe warning: {e}", t0)

    df = con.execute(
        """
        SELECT *
        FROM read_parquet($p)
        WHERE CAST(condition_id AS VARCHAR) IN (SELECT UNNEST(CAST($cids AS VARCHAR[])))
        """,
        {"p": str(source_path), "cids": hf_cids},
    ).df()
    con.close()
    log(f"HF trades after local filter: {len(df):,}", t0)

    if df.empty:
        return pd.DataFrame()

    df["condition_id"] = df["condition_id"].astype(str)
    token_map = {
        row.condition_id: str(row.token_ids).split(";")
        for row in markets.loc[hf_mask].itertuples()
    }
    def _oidx(row):
        toks = token_map.get(row["condition_id"])
        if not toks or len(toks) != 2:
            return pd.NA
        ns = str(row["nonusdc_side"])
        if ns == toks[0]:
            return 0
        if ns == toks[1]:
            return 1
        return pd.NA

    df["outcomeIndex"] = df.apply(_oidx, axis=1)

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "proxyWallet": df["taker"].astype(str),
            "side": df["taker_direction"].astype(str).str.upper(),
            "asset": df["nonusdc_side"].astype(str),
            "size": pd.to_numeric(df["token_amount"], errors="coerce"),
            "price": pd.to_numeric(df["price"], errors="coerce"),
            "timestamp": ts,
            "outcomeIndex": df["outcomeIndex"],
            "transactionHash": df["transaction_hash"].astype(str),
            "condition_id": df["condition_id"].astype(str),
            "source": "hf",
        }
    )
    return out


def build_api_trades(markets: pd.DataFrame, t0: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Polymarket Data API fetch for ceasefire sub-markets. Returns (trades, prices)."""
    api_mask = markets["source"] == "api"
    api_cids = markets.loc[api_mask, "condition_id"].tolist()
    log(f"fetching {len(api_cids)} ceasefire markets via Polymarket Data/CLOB APIs...", t0)

    trade_frames: list[pd.DataFrame] = []
    price_frames: list[pd.DataFrame] = []

    for _, m in markets.loc[api_mask].iterrows():
        cid = m["condition_id"]
        tokens = str(m["token_ids"]).split(";")
        try:
            tdf = fp.fetch_trades(cid)
            if not tdf.empty:
                tdf["source"] = "api"
                trade_frames.append(tdf)
        except Exception as e:
            log(f"  trade error {cid}: {e}", t0)
        for tid in tokens:
            try:
                pdf = fp.fetch_price_history(tid)
                if not pdf.empty:
                    price_frames.append(pdf)
            except Exception as e:
                log(f"  price error {tid[:16]}...: {e}", t0)

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    log(f"API trades fetched: {len(trades):,}   prices: {len(prices):,}", t0)
    return trades, prices


def main():
    parser = argparse.ArgumentParser(description="Hybrid Iran markets dataset builder.")
    parser.add_argument("--skip-download", action="store_true",
                        help="fail if parquets are missing instead of downloading")
    parser.add_argument("--force-download", action="store_true",
                        help="re-download parquets even if present")
    parser.add_argument("--dry-run", action="store_true",
                        help="print plan and exit without fetching trades")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = log("building Iran markets dataset (hybrid HF + Polymarket API)...")

    markets = build_markets_meta(args.skip_download, args.force_download, t0)
    log(f"markets union: {len(markets)} rows   "
        f"(hf={ (markets['source']=='hf').sum() }, "
        f"api={ (markets['source']=='api').sum() })", t0)

    if args.dry_run:
        log("dry-run: exiting before trade fetch", t0)
        return

    hf_trades = build_hf_trades(markets, args.skip_download, args.force_download, t0)
    api_trades, api_prices = build_api_trades(markets, t0)

    trades = pd.concat(
        [x for x in [hf_trades, api_trades] if not x.empty],
        ignore_index=True,
    )
    if trades.empty:
        raise SystemExit("no trades collected from either source")

    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    log(f"combined trades: {len(trades):,} rows "
        f"(hf={(trades['source']=='hf').sum():,}, "
        f"api={(trades['source']=='api').sum():,})", t0)

    # Enrich. The HF path has no CLOB prices, so only `api_prices` is passed;
    # enrich_trades falls back to the per-trade execution price when CLOB is
    # empty, which is correct for the HF trades (they already have `price`).
    log("enriching trades (+ running features + expanded features + split)...", t0)
    enriched = fp.enrich_trades(trades, markets, api_prices)

    # Write outputs.
    markets_out = markets.copy()
    markets_out["end_date"] = markets_out["end_date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    markets_out.to_csv(DATA_DIR / "markets.csv", index=False)
    trades.to_csv(DATA_DIR / "trades.csv", index=False)
    if not api_prices.empty:
        api_prices.to_csv(DATA_DIR / "prices.csv", index=False)
    enriched.to_csv(DATA_DIR / "trades_enriched.csv", index=False)

    log(f"DONE. enriched: {len(enriched):,} rows x {len(enriched.columns)} cols", t0)
    log(f"  markets.csv        {(DATA_DIR / 'markets.csv').stat().st_size/1e3:,.0f} KB", t0)
    log(f"  trades.csv         {(DATA_DIR / 'trades.csv').stat().st_size/1e6:,.1f} MB", t0)
    log(f"  trades_enriched.csv {(DATA_DIR / 'trades_enriched.csv').stat().st_size/1e6:,.1f} MB", t0)


if __name__ == "__main__":
    main()
