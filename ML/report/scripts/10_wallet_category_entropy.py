"""Add cross-market category-entropy feature for each wallet (Bucket 2 #4).

Pipeline (4 phases, each resumable and memory-bounded under ~2 GB):

  1. Stream HF mirror `SII-WANGZJ/Polymarket_data` trades.parquet via fsspec
     row-group chunks (no 38 GB local cache). Filter server-side on our
     109K proxyWallet addresses (either maker or taker). Writes one shard
     per 10 row groups into `data/00_hf_wallet_cross_markets_chunks/`, then
     consolidates into `data/00_hf_wallet_cross_markets.parquet`. Skipped
     on rerun if all 592 row-group shards are already on disk.

  2. duckdb stream-join of cross-markets with `00_hf_markets_master.parquet`
     to attach `event_slug`, sorted by (wallet, timestamp). Output:
     `data/00_hf_wallet_cross_markets_enriched.parquet`. Streams — peak
     RSS stays under ~1 GB regardless of input size. Skipped on rerun if
     output present.

  3. Arrow-batch streamed expanding entropy: read the enriched parquet in
     200K-row batches, maintain per-wallet state across batch boundaries,
     emit one (wallet, timestamp, condition_id, entropy) row per input.
     Peak RSS ~500 MB. Output: `data/00_wallet_entropy.parquet`. Skipped
     on rerun if row count matches the enriched parquet.

  4. duckdb stream-join the entropy parquet into the mother
     `data/03_trades_features.csv`, writing a new CSV with one new
     column — `wallet_market_category_entropy` (Shannon nats; NaN on first
     cross-market trade). Peak RSS ~2 GB. Backup written to
     `data/03_trades_features.pre10.csv`.

Runtime estimate: phase 1 ≈ 1.5–2 h network-bound (skipped on rerun);
phase 2 ≈ 5 min; phase 3 ≈ 15–25 min (pure-Python loop, I/O-bounded by the
Arrow batch reader); phase 4 ≈ 5–10 min. End-to-end on a warm cache: ~30 min.

Requires `pyarrow`, `fsspec`, `duckdb` (already in the py312 env).

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
BUCKETS_DIR = ROOT / "data" / "00_hf_wallet_cross_markets_buckets"
ENTROPY_PARQUET = ROOT / "data" / "00_wallet_entropy.parquet"
CHUNKS_DIR = ROOT / "data" / "00_hf_wallet_cross_markets_chunks"
MARKETS_MASTER = ROOT / "data" / "00_hf_markets_master.parquet"

# Hash-bucket count for wallet-partitioned entropy compute. 64 buckets
# over ~130M rows gives ~2M rows per bucket → sorts in pandas under
# 500 MB RAM per bucket. Avoids the global 130M-row sort which
# overwhelmed 16 GB of disk spill space on the last run.
NUM_BUCKETS = 64

# Write a parquet shard every CHUNK_SIZE_RG row groups so a dropped
# connection mid-stream resumes from the last persisted shard, not from
# scratch. Each shard is named rg_{start:05d}-{end:05d}.parquet.
CHUNK_SIZE_RG = 10

# Arrow batch size for wallet-partitioned entropy streaming. 200K rows of
# four columns is ~15 MB — keeps peak RSS on the entropy compute below
# 500 MB even on 130M-row inputs.
ENTROPY_BATCH_ROWS = 200_000

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


def attach_category_and_bucket() -> str:
    """duckdb stream-joins cross-markets with markets master to attach
    event_slug, then writes ONE parquet file per wallet-hash bucket into
    BUCKETS_DIR. No global sort — the sort happens per bucket inside
    compute_expanding_entropy_streaming().

    Returns the slug column name that was matched (for logging).
    """
    if not MARKETS_MASTER.exists():
        raise FileNotFoundError(
            f"markets master not found at {MARKETS_MASTER}. Run 02_build_dataset.py "
            f"first (or copy from another cluster's build)."
        )

    import duckdb

    con = duckdb.connect()
    # Conservative caps: streaming JOIN + partitioned write needs no sort,
    # so 2 GB RAM + minimal temp disk is enough.
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_entropy'")
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA preserve_insertion_order=false")

    master_cols = [
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{MARKETS_MASTER}') LIMIT 0"
        ).fetchall()
    ]
    slug_col = next(
        (c for c in ("event_slug", "eventSlug", "slug") if c in master_cols),
        None,
    )
    if slug_col is None:
        raise RuntimeError(
            f"markets master has no event_slug/eventSlug/slug column. "
            f"Columns present: {master_cols}"
        )
    if "condition_id" not in master_cols:
        raise RuntimeError(
            f"markets master missing `condition_id` column. "
            f"Columns present: {master_cols}"
        )

    BUCKETS_DIR.mkdir(parents=True, exist_ok=True)
    # Clean any stale bucket files so PARTITION_BY writes don't mix runs.
    for p in BUCKETS_DIR.glob("**/*.parquet"):
        p.unlink()
    for d in sorted(BUCKETS_DIR.glob("bucket=*"), reverse=True):
        if d.is_dir():
            for f in d.iterdir():
                f.unlink()
            d.rmdir()

    print(
        f"duckdb: streaming JOIN + hash-partitioned write into "
        f"{NUM_BUCKETS} buckets (no global sort)..."
    )
    t0 = time.time()
    # Hash-bucket the wallet so all trades for a given wallet land in the
    # same bucket file. Per-bucket sort happens in phase 3.
    con.execute(
        f"""
        COPY (
            SELECT c.wallet                             AS wallet,
                   c.timestamp                          AS timestamp,
                   c.condition_id                       AS condition_id,
                   m.{slug_col}                         AS event_slug,
                   CAST(hash(c.wallet) % {NUM_BUCKETS} AS INTEGER) AS bucket
            FROM read_parquet('{CROSS_MARKETS_CACHE}') c
            LEFT JOIN (
                SELECT DISTINCT ON (condition_id) condition_id, {slug_col}
                FROM read_parquet('{MARKETS_MASTER}')
            ) m ON c.condition_id = m.condition_id
        )
        TO '{BUCKETS_DIR}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (bucket),
         OVERWRITE_OR_IGNORE);
        """
    )
    bucket_dirs = sorted(BUCKETS_DIR.glob("bucket=*"))
    total_bytes = sum(
        p.stat().st_size for p in BUCKETS_DIR.rglob("*.parquet")
    )
    print(
        f"  wrote {len(bucket_dirs)} bucket dirs ({total_bytes / 1e6:.0f} MB) "
        f"in {time.time() - t0:.0f}s"
    )
    return slug_col


def compute_expanding_entropy_streaming() -> None:
    """Process each wallet-hash bucket in isolation: load the bucket's
    parquet into pandas (max ~2 M rows), sort in memory by (wallet,
    timestamp), compute causal expanding-entropy per wallet, append to
    the output parquet. Peak RSS per bucket is ~500 MB.

    Input:  BUCKETS_DIR/bucket=K/*.parquet for K in 0..NUM_BUCKETS-1.
    Output: ENTROPY_PARQUET with (wallet, timestamp, condition_id,
            wallet_market_category_entropy).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    print("per-bucket entropy compute (sort + causal expanding)...")
    t_total = time.time()

    bucket_dirs = sorted(
        BUCKETS_DIR.glob("bucket=*"),
        key=lambda p: int(p.name.split("=")[1]),
    )
    if not bucket_dirs:
        raise RuntimeError(
            f"no bucket directories found in {BUCKETS_DIR} — "
            f"attach_category_and_bucket() must run first"
        )

    out_schema = pa.schema(
        [
            pa.field("wallet", pa.string()),
            pa.field("timestamp", pa.int64()),
            pa.field("condition_id", pa.string()),
            pa.field("wallet_market_category_entropy", pa.float64()),
        ]
    )
    writer = pq.ParquetWriter(ENTROPY_PARQUET, out_schema, compression="zstd")

    category_counts: dict[str, int] = {c: 0 for c in CATEGORY_LABELS}
    rows_done = 0
    rows_total = 0

    for bd in bucket_dirs:
        bi = int(bd.name.split("=")[1])
        t_bucket = time.time()
        files = sorted(bd.glob("*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.sort_values(["wallet", "timestamp"], kind="mergesort").reset_index(
            drop=True
        )
        # Dedup on (wallet, timestamp, condition_id). A wallet making
        # multiple order fills at the same second on the same market
        # produces duplicate rows — but entropy only depends on
        # strictly-prior DISTINCT markets, so the entropy value is
        # identical across duplicates. Dedupe here (inside the bucket,
        # guaranteed safe because a wallet's rows are all in one bucket)
        # keeps the output one-to-one with (wallet, ts, cond) triples
        # and prevents a 5× row explosion at merge time.
        n_pre = len(df)
        df = df.drop_duplicates(
            subset=["wallet", "timestamp", "condition_id"], keep="first"
        ).reset_index(drop=True)
        n = len(df)
        rows_total += n
        n_dedup_dropped = n_pre - n

        wallets = df["wallet"].values
        timestamps = df["timestamp"].values.astype(np.int64)
        conds = df["condition_id"].values
        slugs = df["event_slug"].values

        entropy = np.full(n, np.nan, dtype=np.float64)
        cur_wallet: str | None = None
        seen: set[str] = set()
        counts = np.zeros(K, dtype=np.int32)
        total_markets = 0

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
                cat = bucket_slug(slugs[i])
                counts[CAT_TO_IDX[cat]] += 1
                total_markets += 1
                category_counts[cat] += 1

        out_batch = pa.record_batch(
            [
                pa.array(wallets, type=pa.string()),
                pa.array(timestamps),
                pa.array(conds, type=pa.string()),
                pa.array(entropy, type=pa.float64()),
            ],
            schema=out_schema,
        )
        writer.write_batch(out_batch)

        rows_done += n
        del df, wallets, timestamps, conds, slugs, entropy

        print(
            f"  bucket {bi:>2d}/{NUM_BUCKETS}  rows={n:>9,}  "
            f"(dedup dropped {n_dedup_dropped:>9,})  "
            f"cumulative={rows_done:>11,}  "
            f"bucket_secs={time.time() - t_bucket:5.1f}  "
            f"elapsed={time.time() - t_total:5.0f}s"
        )

    writer.close()
    print(
        f"\n  wrote {ENTROPY_PARQUET.name} — {rows_total:,} rows "
        f"in {time.time() - t_total:.0f}s"
    )
    print("  category distribution (distinct-market counts across all wallets):")
    total_cat = sum(category_counts.values())
    for c in CATEGORY_LABELS:
        n = category_counts[c]
        pct = n / max(1, total_cat) * 100
        print(f"    {c:20s} {n:>12,}  ({pct:5.1f}%)")


def merge_entropy_into_csv_duckdb() -> None:
    """Stream-join the entropy parquet into the mother CSV via duckdb and
    write a new CSV. Memory stays under ~2 GB — duckdb streams rows rather
    than loading pandas DataFrames.
    """
    import duckdb

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_entropy'")
    con.execute("PRAGMA threads=4")

    # Preserve the first-good backup. If .pre10.csv already exists and
    # its column count is 65 (pre-entropy), trust it as the canonical
    # pre-merge state and do not overwrite it with a possibly-broken
    # intermediate state from a previous failed run.
    t0 = time.time()
    if BACKUP_CSV.exists():
        with BACKUP_CSV.open("r") as fh:
            backup_cols = len(fh.readline().split(","))
        if backup_cols == 65:
            print(
                f"  {BACKUP_CSV.name} already present (65 cols, pre-entropy) "
                f"— reusing, not overwriting"
            )
        else:
            print(
                f"  {BACKUP_CSV.name} exists but has {backup_cols} cols "
                f"(expected 65) — overwriting with current CSV"
            )
            shutil.copy2(FEATURES_CSV, BACKUP_CSV)
    else:
        print(f"backing up current CSV to {BACKUP_CSV.name}...")
        shutil.copy2(FEATURES_CSV, BACKUP_CSV)
    print(f"  backup step done in {time.time() - t0:.0f}s")

    print("duckdb: stream-joining entropy parquet into CSV...")
    t1 = time.time()
    tmp_out = FEATURES_CSV.with_suffix(".csv.tmp")
    if tmp_out.exists():
        tmp_out.unlink()

    # The mother CSV's `timestamp` is ISO-formatted strings (e.g.
    # "2025-12-22 16:57:07+00:00") from 02_build_dataset.py, while the
    # entropy parquet's `timestamp` is BIGINT Unix seconds. The CSV is
    # auto-detected as TIMESTAMP; we convert to epoch seconds inside the
    # JOIN via EPOCH(). The entropy parquet is already deduped per
    # (wallet, timestamp, condition_id) during phase 3.
    con.execute(
        f"""
        COPY (
            SELECT c.*,
                   e.wallet_market_category_entropy
            FROM read_csv(
                '{BACKUP_CSV}',
                HEADER=TRUE,
                AUTO_DETECT=TRUE,
                SAMPLE_SIZE=-1
            ) c
            LEFT JOIN read_parquet('{ENTROPY_PARQUET}') e
              ON c.{WALLET_COL}                            = e.wallet
             AND CAST(EPOCH(c.{TIMESTAMP_COL}) AS BIGINT)  = e.timestamp
             AND c.condition_id                            = e.condition_id
        )
        TO '{tmp_out}'
        (FORMAT CSV, HEADER);
        """
    )
    tmp_out.replace(FEATURES_CSV)
    print(f"  merge + CSV write done in {time.time() - t1:.0f}s")

    # Quick sanity check — col count and match rate
    col_count = len(
        con.execute(
            f"SELECT * FROM read_csv('{FEATURES_CSV}', HEADER=TRUE, "
            f"AUTO_DETECT=TRUE, SAMPLE_SIZE=-1) LIMIT 0"
        ).description
    )
    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_csv('{FEATURES_CSV}', HEADER=TRUE, "
        f"AUTO_DETECT=TRUE, SAMPLE_SIZE=-1)"
    ).fetchone()[0]
    n_matched = con.execute(
        f"SELECT COUNT(*) FROM read_csv('{FEATURES_CSV}', HEADER=TRUE, "
        f"AUTO_DETECT=TRUE, SAMPLE_SIZE=-1) "
        f"WHERE wallet_market_category_entropy IS NOT NULL"
    ).fetchone()[0]
    print(
        f"  final CSV shape: {n_rows:,} rows × {col_count} cols; "
        f"entropy matched on {n_matched:,} rows "
        f"({n_matched / max(1, n_rows) * 100:.1f}%)"
    )


def _shards_cover_all_row_groups() -> bool:
    """True if every HF row group 0..591 is present in CHUNKS_DIR. Avoids
    the false-alarm re-stream when some of our wallets simply never
    appeared on the HF mirror as maker/taker (which is legitimate — they
    may have only traded on post-cutoff ceasefire markets not in the HF
    snapshot)."""
    if not CHUNKS_DIR.exists():
        return False
    covered = _covered_row_groups(CHUNKS_DIR)
    # Total row group count lives in the consolidated parquet's metadata
    # once we have one; for the common case we just trust that 0..591 are
    # expected and verify against the schema we've used throughout.
    # Hard-coded here because the 592 total is a property of the dataset,
    # not our code.
    expected = set(range(592))
    return expected.issubset(covered)


def main() -> None:
    t_total = time.time()

    print("phase 1 — cross-market trade history (HF mirror stream)")
    print("loading mother dataframe join keys...")
    df_keys = pd.read_csv(
        FEATURES_CSV,
        usecols=[WALLET_COL],
        dtype={WALLET_COL: "string"},
    )
    wallets = set(df_keys[WALLET_COL].dropna().unique().tolist())
    print(f"  {len(wallets):,} unique wallets")

    need_stream = True
    if CROSS_MARKETS_CACHE.exists() and _shards_cover_all_row_groups():
        print(f"  cache + all 592 shards present — skipping stream entirely")
        need_stream = False
    elif CROSS_MARKETS_CACHE.exists():
        print(
            f"  cache exists but shards incomplete — re-streaming missing "
            f"row groups only"
        )
    else:
        print(f"  no cache — full HF mirror stream (long step, ~1–2 h)")

    if need_stream:
        cross = stream_cross_market_history(wallets)
        cross.to_parquet(CROSS_MARKETS_CACHE, index=False)
        print(f"  cached to {CROSS_MARKETS_CACHE}")
        del cross  # free pandas frame before the duckdb phase

    print("\nphase 2 — markets-master join + hash-bucketed write (duckdb)")
    existing_buckets = sorted(BUCKETS_DIR.glob("bucket=*")) if BUCKETS_DIR.exists() else []
    if len(existing_buckets) == NUM_BUCKETS:
        print(
            f"  all {NUM_BUCKETS} bucket dirs present at {BUCKETS_DIR.name} — reusing"
        )
    else:
        attach_category_and_bucket()

    print("\nphase 3 — expanding entropy per bucket")
    skip_phase3 = False
    if ENTROPY_PARQUET.exists():
        import pyarrow.parquet as pq

        entropy_rows = pq.ParquetFile(ENTROPY_PARQUET).metadata.num_rows
        # Sum rows across all bucket parquets as the source-of-truth.
        bucket_rows = 0
        for bd in BUCKETS_DIR.glob("bucket=*"):
            for f in bd.glob("*.parquet"):
                bucket_rows += pq.ParquetFile(f).metadata.num_rows
        if entropy_rows == bucket_rows and entropy_rows > 0:
            print(
                f"  {ENTROPY_PARQUET.name} matches bucket total "
                f"({entropy_rows:,} rows) — reusing"
            )
            skip_phase3 = True
        else:
            print(
                f"  row count mismatch (entropy={entropy_rows:,} "
                f"vs buckets={bucket_rows:,}) — recomputing"
            )
            ENTROPY_PARQUET.unlink()
    if not skip_phase3:
        compute_expanding_entropy_streaming()

    print("\nphase 4 — CSV merge (duckdb, streamed)")
    merge_entropy_into_csv_duckdb()

    print(
        f"\nwrote {FEATURES_CSV.name} in {time.time() - t_total:.0f}s total. "
        f"Layer-6 integrator (11_add_layer6.py) can now run on top."
    )


if __name__ == "__main__":
    main()
