"""Add cross-market category-entropy feature for each wallet (Bucket 2 #4).

Streams the HuggingFace mirror `SII-WANGZJ/Polymarket_data` trades parquet
via fsspec in row-group chunks (no 38 GB local cache), filters server-side
on our 109K proxyWallet addresses (either side of the trade), looks up
`condition_id → event_slug` via the locally-cached
`data/00_hf_markets_master.parquet`, buckets each visited market into a
coarse category by keyword, and computes an expanding Shannon entropy
over categories per wallet — strictly causal: entropy at trade k uses only
trades with timestamp < trade_k's timestamp.

Output: new column on `data/03_trades_features.csv`:
  - wallet_market_category_entropy    Shannon entropy (nats) over prior category
                                      distribution for the wallet; NaN on first
                                      cross-market trade (< 2 prior markets).

Runtime estimate: 1–2 h HF stream (network bound) + ~0.5 h bucketing +
expanding entropy. Resumable — the per-wallet cross-market trade set is
cached at `data/00_hf_wallet_cross_markets.parquet`; rerunning skips the
stream if the cache covers all our wallets.

Requires `pyarrow`, `fsspec` (already used by 02_build_dataset.py).

Usage:
  python scripts/10_wallet_category_entropy.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data" / "03_trades_features.csv"
BACKUP_CSV = ROOT / "data" / "03_trades_features.pre10.csv"
CROSS_MARKETS_CACHE = ROOT / "data" / "00_hf_wallet_cross_markets.parquet"
CHUNKS_DIR = ROOT / "data" / "00_hf_wallet_cross_markets_chunks"
MARKETS_MASTER = ROOT / "data" / "00_hf_markets_master.parquet"

# Write a parquet shard every CHUNK_SIZE_RG row groups so a dropped
# connection mid-stream resumes from the last persisted shard, not from
# scratch. Each shard is named rg_{start:05d}-{end:05d}.parquet.
CHUNK_SIZE_RG = 10

WALLET_COL = "proxyWallet"
TIMESTAMP_COL = "timestamp"

HF_TRADES_URL = (
    "https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data/"
    "resolve/main/trades.parquet"
)
HF_TRADES_COLUMNS = ["timestamp", "condition_id", "maker", "taker"]
REMOTE_READ_MAX_ATTEMPTS = 6
REMOTE_READ_BACKOFF_CAP_S = 60


# Coarse category buckets, applied to event_slug. Order matters — first
# match wins. Keeps K small so entropy is interpretable (max ln K ≈ 2.08
# for K=8). "other" is residual.
CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "geopolitics",
        (
            "iran", "israel", "russia", "ukraine", "china", "taiwan",
            "putin", "zelensky", "netanyahu", "gaza", "hamas", "nato",
            "ceasefire", "war", "strike", "sanction",
        ),
    ),
    (
        "us_politics",
        (
            "trump", "biden", "harris", "desantis", "pelosi", "congress",
            "senate", "house", "election", "primary", "debate", "supreme-court",
            "impeach", "pardon",
        ),
    ),
    (
        "crypto",
        (
            "btc", "bitcoin", "eth", "ethereum", "sol", "solana", "doge",
            "usdc", "usdt", "tether", "stablecoin", "crypto", "nft",
            "binance", "coinbase",
        ),
    ),
    (
        "macro_finance",
        (
            "fed", "recession", "inflation", "cpi", "rate-cut", "sp500",
            "nasdaq", "gold", "oil", "gdp", "unemployment", "tariff",
        ),
    ),
    (
        "sports",
        (
            "nfl", "nba", "mlb", "nhl", "premier-league", "champions-league",
            "world-cup", "super-bowl", "olympics", "tennis", "ufc",
            "f1", "formula-1", "boxing",
        ),
    ),
    (
        "entertainment",
        (
            "oscar", "grammy", "emmy", "netflix", "taylor-swift", "drake",
            "kardashian", "beyonce", "movie", "box-office", "album",
        ),
    ),
    (
        "tech_business",
        (
            "apple", "tesla", "openai", "gpt", "claude", "anthropic",
            "google", "microsoft", "nvidia", "meta", "x-twitter", "ipo",
            "musk", "zuckerberg",
        ),
    ),
]

CATEGORY_LABELS = [label for label, _ in CATEGORY_KEYWORDS] + ["other"]
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORY_LABELS)}
K = len(CATEGORY_LABELS)


def bucket_slug(slug: str | None) -> str:
    if not isinstance(slug, str) or not slug:
        return "other"
    s = slug.lower()
    for label, keywords in CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in s:
                return label
    return "other"


def _open_remote_parquet_handle():
    """Fresh fsspec handle to the HF parquet. Mirrors 02_build_dataset.py."""
    import fsspec

    of = fsspec.open(HF_TRADES_URL, "rb")
    return of.open()


def _covered_row_groups(chunks_dir: Path) -> set[int]:
    """Scan chunks dir and return the set of row-group indices already saved."""
    if not chunks_dir.exists():
        return set()
    covered: set[int] = set()
    for p in chunks_dir.glob("rg_*.parquet"):
        stem = p.stem  # rg_00020-00029
        try:
            rng = stem.split("_", 1)[1]
            lo_s, hi_s = rng.split("-")
            for i in range(int(lo_s), int(hi_s) + 1):
                covered.add(i)
        except (ValueError, IndexError):
            continue
    return covered


def _write_chunk(
    chunks_dir: Path, rg_start: int, rg_end: int, frames: list[pd.DataFrame]
) -> None:
    """Atomic chunk write: concat + write to tmp + rename."""
    if not frames:
        # Write an empty marker so resume doesn't re-stream dead row groups.
        empty = pd.DataFrame(columns=["wallet", "timestamp", "condition_id"])
        frame = empty
    else:
        frame = pd.concat(frames, ignore_index=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    dest = chunks_dir / f"rg_{rg_start:05d}-{rg_end:05d}.parquet"
    tmp = dest.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(dest)


def stream_cross_market_history(wallets: set[str]) -> pd.DataFrame:
    """Row-group-chunked stream of HF trades.parquet, kept rows where
    maker OR taker is in `wallets`. Writes a parquet shard every
    CHUNK_SIZE_RG row groups into CHUNKS_DIR so a mid-stream interruption
    resumes from the last persisted shard. Returns the concatenated frame.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wallets_arr = pa.array(list(wallets), type=pa.string())

    t0 = time.time()
    fh = _open_remote_parquet_handle()
    pf = pq.ParquetFile(fh)
    num_rg = pf.num_row_groups
    total_rows = pf.metadata.num_rows
    print(f"  remote parquet: {num_rg} row groups, {total_rows:,} rows")

    covered = _covered_row_groups(CHUNKS_DIR)
    if covered:
        print(
            f"  resume: {len(covered):,} row groups already covered by "
            f"{len(list(CHUNKS_DIR.glob('rg_*.parquet')))} shards; "
            f"skipping those"
        )

    batch_frames: list[pd.DataFrame] = []
    batch_start_rg: int | None = None
    rows_read = 0
    rows_kept = 0

    def flush_batch(end_rg: int) -> None:
        nonlocal batch_frames, batch_start_rg
        if batch_start_rg is None:
            return
        _write_chunk(CHUNKS_DIR, batch_start_rg, end_rg, batch_frames)
        batch_frames = []
        batch_start_rg = None

    for rg_idx in range(num_rg):
        if rg_idx in covered:
            continue

        if batch_start_rg is None:
            batch_start_rg = rg_idx

        tbl = None
        last_err: Exception | None = None
        for attempt in range(1, REMOTE_READ_MAX_ATTEMPTS + 1):
            try:
                tbl = pf.read_row_group(rg_idx, columns=HF_TRADES_COLUMNS)
                break
            except Exception as e:
                last_err = e
                backoff = min(REMOTE_READ_BACKOFF_CAP_S, 2**attempt)
                print(
                    f"  row group {rg_idx}: attempt {attempt}/"
                    f"{REMOTE_READ_MAX_ATTEMPTS} failed "
                    f"({type(e).__name__}: {e}); reopening in {backoff}s"
                )
                try:
                    fh.close()
                except Exception:
                    pass
                time.sleep(backoff)
                fh = _open_remote_parquet_handle()
                pf = pq.ParquetFile(fh)
        if tbl is None:
            # Flush whatever we have so a rerun picks up from here.
            if batch_start_rg is not None and batch_start_rg < rg_idx:
                flush_batch(rg_idx - 1)
            raise RuntimeError(
                f"row group {rg_idx} failed after {REMOTE_READ_MAX_ATTEMPTS} "
                f"attempts: {last_err}"
            )

        rows_read += tbl.num_rows

        maker_mask = pc.is_in(pc.cast(tbl["maker"], pa.string()), value_set=wallets_arr)
        taker_mask = pc.is_in(pc.cast(tbl["taker"], pa.string()), value_set=wallets_arr)
        both = pc.or_(maker_mask, taker_mask)
        filt = tbl.filter(both)

        if filt.num_rows:
            df = filt.to_pandas()
            taker_in = df["taker"].isin(wallets)
            maker_in = df["maker"].isin(wallets)
            df["wallet"] = np.where(taker_in, df["taker"], df["maker"])
            df = df[maker_in | taker_in]
            batch_frames.append(df[["wallet", "timestamp", "condition_id"]])
            rows_kept += len(df)

        # Flush a shard at batch boundary or on last row group.
        at_boundary = (rg_idx + 1) % CHUNK_SIZE_RG == 0
        is_last = rg_idx == num_rg - 1
        if at_boundary or is_last:
            flush_batch(rg_idx)
            pct = 100.0 * (rg_idx + 1) / num_rg
            elapsed = time.time() - t0
            print(
                f"  row group {rg_idx + 1}/{num_rg} ({pct:5.1f}%) "
                f"rows_read={rows_read:,} kept={rows_kept:,} "
                f"elapsed={elapsed:.0f}s — shard flushed"
            )

    try:
        fh.close()
    except Exception:
        pass

    # Consolidate all shards into the single final parquet.
    shard_paths = sorted(CHUNKS_DIR.glob("rg_*.parquet"))
    print(f"  consolidating {len(shard_paths)} shards...")
    out = (
        pd.concat([pd.read_parquet(p) for p in shard_paths], ignore_index=True)
        if shard_paths
        else pd.DataFrame(columns=["wallet", "timestamp", "condition_id"])
    )
    print(f"  total: {len(out):,} cross-market trades in {time.time() - t0:.0f}s")
    return out


def attach_category(cross: pd.DataFrame) -> pd.DataFrame:
    """Look up event_slug for each condition_id via the cached markets master,
    then bucket into a category. Falls back to 'other' for missing slugs.
    """
    if not MARKETS_MASTER.exists():
        raise FileNotFoundError(
            f"markets master not found at {MARKETS_MASTER}. Run 02_build_dataset.py "
            f"first (or copy from another cluster's build)."
        )

    print(f"loading markets master for category lookup...")
    master = pd.read_parquet(MARKETS_MASTER)
    # The master is expected to have `condition_id` and `event_slug`; if the
    # column name differs, surface a clear error.
    if "condition_id" not in master.columns:
        raise RuntimeError(
            f"markets master missing `condition_id` column. "
            f"Columns present: {list(master.columns)}"
        )
    slug_col = next(
        (c for c in ("event_slug", "eventSlug", "slug") if c in master.columns),
        None,
    )
    if slug_col is None:
        raise RuntimeError(
            f"markets master has no event_slug/eventSlug/slug column. "
            f"Columns present: {list(master.columns)}"
        )
    lookup = master[["condition_id", slug_col]].drop_duplicates("condition_id")
    lookup = lookup.rename(columns={slug_col: "event_slug"})

    cross = cross.merge(lookup, on="condition_id", how="left")
    missing = cross["event_slug"].isna().sum()
    if missing:
        print(
            f"  {missing:,} / {len(cross):,} rows missing event_slug "
            f"({missing / len(cross) * 100:.2f}%) — bucketed as 'other'"
        )

    print("bucketing slugs into categories...")
    cross["category"] = cross["event_slug"].map(bucket_slug)
    cross["cat_idx"] = cross["category"].map(CAT_TO_IDX).astype(np.int8)
    print("  category distribution (row counts):")
    print(cross["category"].value_counts().to_string())
    return cross


def compute_expanding_entropy(cross: pd.DataFrame) -> pd.DataFrame:
    """Per wallet, over time: Shannon entropy (nats) over the distinct-market
    category distribution, strictly before each trade's timestamp.
    """
    cross = cross.sort_values(["wallet", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )

    print("computing expanding entropy per wallet...")
    t0 = time.time()
    n = len(cross)
    entropy = np.full(n, np.nan, dtype=np.float64)

    cur_wallet: str | None = None
    seen: set[str] = set()
    counts = np.zeros(K, dtype=np.int32)
    total_markets = 0

    wallets = cross["wallet"].values
    conds = cross["condition_id"].values
    cat_idxs = cross["cat_idx"].values

    for i in range(n):
        w = wallets[i]
        if w != cur_wallet:
            cur_wallet = w
            seen = set()
            counts = np.zeros(K, dtype=np.int32)
            total_markets = 0

        if total_markets >= 2:
            p = counts / total_markets
            nz = p > 0
            entropy[i] = float(-(p[nz] * np.log(p[nz])).sum())

        c = conds[i]
        if c not in seen:
            seen.add(c)
            counts[cat_idxs[i]] += 1
            total_markets += 1

        if (i + 1) % 500_000 == 0:
            print(
                f"  {i + 1:,} / {n:,} "
                f"({(i + 1) / n * 100:.1f}%, {time.time() - t0:.0f}s)"
            )

    cross["wallet_market_category_entropy"] = entropy
    print(f"  done in {time.time() - t0:.0f}s")
    return cross[
        ["wallet", "timestamp", "condition_id", "wallet_market_category_entropy"]
    ]


def main() -> None:
    t_total = time.time()

    print("loading mother dataframe join keys...")
    df_keys = pd.read_csv(
        FEATURES_CSV,
        usecols=[WALLET_COL, TIMESTAMP_COL, "condition_id"],
        dtype={WALLET_COL: "string", "condition_id": "string"},
    )
    print(f"  {len(df_keys):,} rows")

    wallets = set(df_keys[WALLET_COL].dropna().unique().tolist())
    print(f"  {len(wallets):,} unique wallets")

    if CROSS_MARKETS_CACHE.exists():
        print(f"cache found at {CROSS_MARKETS_CACHE.name} — loading")
        cross = pd.read_parquet(CROSS_MARKETS_CACHE)
        cached_wallets = set(cross["wallet"].unique().tolist())
        missing = wallets - cached_wallets
        if missing:
            print(
                f"  WARNING: cache missing {len(missing):,} of our wallets — "
                f"re-streaming"
            )
            cross = stream_cross_market_history(wallets)
            cross.to_parquet(CROSS_MARKETS_CACHE, index=False)
        else:
            print(f"  cache covers all {len(wallets):,} wallets — skipping stream")
    else:
        print("no cache — streaming HF mirror (this is the long step, ~1–2 h)")
        cross = stream_cross_market_history(wallets)
        cross.to_parquet(CROSS_MARKETS_CACHE, index=False)
        print(f"  cached to {CROSS_MARKETS_CACHE}")

    cross = attach_category(cross)
    entropy_frame = compute_expanding_entropy(cross)

    print("joining entropy feature back to mother dataframe...")
    t_join = time.time()
    full = pd.read_csv(FEATURES_CSV)
    # Deduplicate entropy frame on (wallet, timestamp, condition_id) to
    # guarantee a one-to-one join with the mother frame.
    entropy_frame = entropy_frame.drop_duplicates(
        ["wallet", "timestamp", "condition_id"], keep="last"
    )
    merged = full.merge(
        entropy_frame.rename(columns={"wallet": WALLET_COL}),
        on=[WALLET_COL, TIMESTAMP_COL, "condition_id"],
        how="left",
    )
    n_matched = int(merged["wallet_market_category_entropy"].notna().sum())
    print(
        f"  matched {n_matched:,} / {len(merged):,} rows "
        f"({n_matched / len(merged) * 100:.1f}%) in {time.time() - t_join:.0f}s"
    )

    print(
        f"\nwallet_market_category_entropy summary:\n"
        f"{merged['wallet_market_category_entropy'].describe().round(3)}"
    )

    print(f"\nbacking up to {BACKUP_CSV.name} and writing patched CSV...")
    shutil.copy2(FEATURES_CSV, BACKUP_CSV)
    merged.to_csv(FEATURES_CSV, index=False)

    print(
        f"\nwrote {FEATURES_CSV.name} — shape {merged.shape} "
        f"in {time.time() - t_total:.0f}s total"
    )


if __name__ == "__main__":
    main()
