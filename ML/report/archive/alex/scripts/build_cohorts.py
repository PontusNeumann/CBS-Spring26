"""Build train (Iran strike) and test (US-Iran ceasefire) cohort parquets from HF.

Dataset: SII-WANGZJ/Polymarket_data (HuggingFace).
Output:
  - data/archive/alex/train.parquet   Iran-strike canonical ladder trades, ts < 2026-02-28 06:35 UTC
  - data/archive/alex/test.parquet    US-Iran-ceasefire canonical ladder trades, ts < 2026-04-07 00:00 UTC
  - alex/outputs/cohort_inventory/keyword_matches.csv   every title matched by broad keyword filter
  - data/archive/alex/markets_subset.parquet  metadata for every included market

Dependencies (install in an ephemeral venv):
    python -m venv .venv && source .venv/bin/activate
    pip install huggingface_hub pyarrow pandas polars tqdm

Run:
    python alex/scripts/build_cohorts.py

Runtime: first run downloads metadata (~1GB of column chunks) then streams ~35MB
per matching row group from HF. Expect 15-30 minutes over a decent connection.

Design choices made autonomously (audit these in alex/notes/cohort_inventory.md):
  - Train canonical ladder = markets whose question matches regex
        ^us strikes iran by \\w+ \\d+,?\\s*20\\d\\d\\??\\s*$
    This excludes "Will the US next strike Iran on [date]" (parallel ladder),
    "Will the US strike Iran next?" (open-ended), "US or Israel", "US not strike",
    odds-of-odds markets, and pre-2025 historical markets.
  - Test canonical ladder = "US x Iran ceasefire by [date]" regex.
    NO Polymarket market literally matches "Will Trump announce Iran ceasefire by X"
    (keyword hits = 0). The closest family whose resolution is explicitly tied to
    Trump's public announcement of a US-Iran ceasefire is the "US x Iran ceasefire
    by [date]" ladder. Israel x Iran ceasefire, broken-ceasefire, duration, and
    peace-deal families are excluded per the brief.
  - Cutoffs:
      * Train: timestamp < 2026-02-28 06:35 UTC (Operation Epic Fury launch)
      * Test:  timestamp < 2026-04-07 00:00 UTC (conservative, drops all Apr 7
               trades since exact announcement time is unknown)
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

# ---- config ----------------------------------------------------------------
HF_REPO = "datasets/SII-WANGZJ/Polymarket_data"
HF_TRADES = f"{HF_REPO}/trades.parquet"
HF_MARKETS = f"{HF_REPO}/markets.parquet"

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../ML/report/
ALEX = REPO_ROOT / "alex"
DATA_DIR = REPO_ROOT / "data" / "archive" / "alex"
NOTES_DIR = ALEX / "notes"
OUT_DIR = ALEX / "outputs" / "cohort_inventory"
SCRATCH = ALEX / ".scratch"

TRAIN_CUTOFF_UTC = datetime(2026, 2, 28, 6, 35, 0, tzinfo=timezone.utc)
TEST_CUTOFF_UTC = datetime(2026, 4, 7, 0, 0, 0, tzinfo=timezone.utc)
LIQUIDITY_FLOOR = 500
LIQUIDITY_FLOOR_FALLBACK = 100  # only for test cohort if primary floor kills it

# ---- regexes --------------------------------------------------------------
# Canonical train: "US strikes Iran by [Month] [Day], [Year]?"
TRAIN_CANONICAL_RE = re.compile(
    r"^us strikes iran by \w+ \d+,?\s*20\d\d\??\s*$", re.IGNORECASE
)
# Canonical test: "US x Iran ceasefire by [Month] [Day]?" (year optional)
TEST_CANONICAL_RE = re.compile(r"^us x iran ceasefire by \w+ \d+\??\s*$", re.IGNORECASE)


# Broad keyword filters for audit dump (task requirement: list every match with decision)
def broad_train_mask(q: pd.Series) -> pd.Series:
    us = q.str.contains(
        r"\b(?:us|u\.s\.|america|american|united states|trump)\b", regex=True, na=False
    )
    strike = q.str.contains(
        r"\b(?:strike|strikes|attack|attacks|bomb|bombs|bombed|bombing|military action|military strike)\b",
        regex=True,
        na=False,
    )
    iran = q.str.contains(r"iran", regex=True, na=False)
    return us & strike & iran


def broad_test_mask(q: pd.Series) -> pd.Series:
    trump = q.str.contains("trump", regex=True, na=False)
    announce = q.str.contains(r"announce(?:s|d|ment|ments)?", regex=True, na=False)
    cf = q.str.contains("ceasefire", regex=True, na=False)
    iran = q.str.contains("iran", regex=True, na=False)
    return trump & announce & cf & iran


# Exclusion pattern explanations for inventory dump
def explain_exclusion(question: str) -> str:
    ql = question.lower()
    if TRAIN_CANONICAL_RE.match(ql):
        return "INCLUDE (train canonical)"
    if TEST_CANONICAL_RE.match(ql):
        return "INCLUDE (test canonical)"
    if "next strike iran" in ql:
        return "EXCLUDE (parallel 'next strike' ladder, different resolution mechanic)"
    if "us or israel" in ql or "us x israel" in ql:
        return "EXCLUDE (joint US/Israel family, not canonical)"
    if "will the us strike iran next" in ql:
        return "EXCLUDE (open-ended, no deadline rung)"
    if "not strike iran" in ql:
        return "EXCLUDE (inverse market, not canonical ladder)"
    if "iran strike on us" in ql:
        return "EXCLUDE (reversed direction: Iran→US)"
    if "accuse iran" in ql:
        return "EXCLUDE (not a strike market)"
    if "odds " in ql or "odds>" in ql or ">30%" in ql or ">50%" in ql or ">20%" in ql:
        return "EXCLUDE (meta-odds market)"
    if "big game" in ql or "durring sotu" in ql or "during sotu" in ql:
        return "EXCLUDE (novelty/conditional timing variant)"
    if re.match(r"^us military action against iran", ql):
        return "EXCLUDE (different family - broader 'military action' ladder)"
    if re.match(r"^another us military action", ql):
        return "EXCLUDE (sequel-action family)"
    if re.match(r"^us strike on iran on june", ql):
        return "EXCLUDE (2025 June strike ladder - different real-world event)"
    if re.match(r"^will us attack iran", ql):
        return "EXCLUDE (2023/2024 historical markets)"
    if "publicly state" in ql:
        return "EXCLUDE (accusation market, not strike)"
    if "trump announces fed nominee" in ql:
        return "EXCLUDE (compound market)"
    if "drone attack" in ql:
        return "EXCLUDE (response to Iran drone, different event)"
    return "EXCLUDE (did not match canonical regex; review)"


# ---- step 1: load markets metadata ---------------------------------------
def load_markets() -> pd.DataFrame:
    local = SCRATCH / "markets.parquet"
    if local.exists():
        print(f"[markets] using cache {local}")
        return pd.read_parquet(local)
    print("[markets] downloading from HF")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    with fs.open(HF_MARKETS, "rb") as f:
        data = f.read()
    local.write_bytes(data)
    return pd.read_parquet(local)


# ---- step 2: pick cohort markets -----------------------------------------
def pick_markets(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q = m["question"].fillna("").astype(str)
    ql = q.str.lower()

    # Broad keyword audit sets (for inventory)
    train_broad = m[broad_train_mask(ql)].copy()
    train_broad["decision"] = train_broad["question"].apply(explain_exclusion)
    train_broad["cohort"] = "train"

    test_broad = m[broad_test_mask(ql)].copy()
    test_broad["decision"] = test_broad["question"].apply(explain_exclusion)
    test_broad["cohort"] = "test"

    # Also audit "US x Iran ceasefire" matches since test_broad may be empty
    # (the strict keyword brief wants trump+announce, but canonical family
    #  is "US x Iran ceasefire" — include as a separate audit slice).
    cf_broad = m[
        ql.str.contains("ceasefire", regex=True, na=False)
        & ql.str.contains("iran", regex=True, na=False)
    ].copy()
    cf_broad["decision"] = cf_broad["question"].apply(
        lambda x: "INCLUDE (test canonical)"
        if TEST_CANONICAL_RE.match(x.lower())
        else (
            "EXCLUDE (Israel x Iran family)"
            if "israel x iran" in x.lower() or "israel or iran" in x.lower()
            else (
                "EXCLUDE (ceasefire broken family)"
                if "broken" in x.lower()
                else (
                    "EXCLUDE (conditional / compound market)"
                    if any(
                        k in x.lower()
                        for k in [
                            "before trump visits",
                            "oil hits",
                            "kevin warsh",
                            "leadership change",
                            "russia x ukraine",
                            "fed nominee",
                        ]
                    )
                    else "EXCLUDE (did not match canonical regex; review)"
                )
            )
        )
    )
    cf_broad["cohort"] = "test"

    audit = pd.concat([train_broad, cf_broad], ignore_index=True)
    audit = audit.drop_duplicates(subset=["id"])
    audit_cols = [
        "cohort",
        "id",
        "question",
        "event_title",
        "created_at",
        "end_date",
        "volume",
        "decision",
    ]
    audit = audit[audit_cols].sort_values(["cohort", "decision", "created_at"])

    # Strict canonical picks
    train_mask = ql.str.match(
        r"^us strikes iran by \w+ \d+,?\s*20\d\d\??\s*$", na=False
    )
    test_mask = ql.str.match(r"^us x iran ceasefire by \w+ \d+\??\s*$", na=False)

    train_markets = m[train_mask].copy()
    test_markets = m[test_mask].copy()
    return train_markets, test_markets, audit


# ---- step 3: stream trades.parquet and filter ----------------------------
def stream_trades(
    target_market_ids: set[str], ts_min_utc: datetime, ts_max_utc: datetime
) -> pd.DataFrame:
    """Scan HF trades.parquet row-group by row-group, keep only rows matching
    market_id + timestamp range. Two-phase: (1) cheap market_id probe with
    just market_id + timestamp columns, (2) full read only for matching rgs.
    """
    ts_min = int(ts_min_utc.timestamp())
    ts_max = int(ts_max_utc.timestamp())

    fs = HfFileSystem()
    print(f"[trades] opening HF parquet (38GB)")
    with fs.open(HF_TRADES, "rb") as f:
        pf = pq.ParquetFile(f)
        meta = pf.metadata
        n_rg = meta.num_row_groups
        print(f"[trades] {n_rg} row groups")

        # schema index of timestamp col (for stats)
        field_names = list(pf.schema_arrow.names)
        ts_col = field_names.index("timestamp")

        # Phase 1: prune row groups by timestamp stats
        candidate_rgs = []
        for i in range(n_rg):
            rg = meta.row_group(i)
            ts_stats = rg.column(ts_col).statistics
            if ts_stats is None:
                candidate_rgs.append(i)
                continue
            if ts_stats.max < ts_min or ts_stats.min > ts_max:
                continue
            candidate_rgs.append(i)
        print(f"[trades] {len(candidate_rgs)} row groups pass timestamp pruning")

        # Phase 2: light probe - read only market_id + timestamp, find matching rgs
        probe_cols = ["market_id", "timestamp"]
        matching_rgs = []
        t0 = time.time()
        for n, i in enumerate(candidate_rgs):
            tab = pf.read_row_group(i, columns=probe_cols)
            mids = tab.column("market_id").to_pylist()
            any_hit = any(mid in target_market_ids for mid in mids)
            if any_hit:
                matching_rgs.append(i)
            if (n + 1) % 25 == 0 or n == len(candidate_rgs) - 1:
                elapsed = time.time() - t0
                print(
                    f"  probe {n + 1}/{len(candidate_rgs)}: {len(matching_rgs)} matching rgs so far ({elapsed:.0f}s)"
                )
        print(
            f"[trades] phase-1 probe: {len(matching_rgs)} row groups contain target markets"
        )

        # Phase 3: read matching row groups in full, filter rows
        full_cols = field_names  # keep schema as-is
        chunks = []
        t0 = time.time()
        for n, i in enumerate(matching_rgs):
            tab = pf.read_row_group(i, columns=full_cols)
            df = tab.to_pandas()
            keep = (
                df["market_id"].isin(target_market_ids)
                & (df["timestamp"] >= ts_min)
                & (df["timestamp"] < ts_max)
            )
            if keep.any():
                chunks.append(df.loc[keep])
            if (n + 1) % 10 == 0 or n == len(matching_rgs) - 1:
                elapsed = time.time() - t0
                kept = sum(len(c) for c in chunks)
                print(
                    f"  full {n + 1}/{len(matching_rgs)}: {kept:,} rows kept ({elapsed:.0f}s)"
                )

    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    return out


# ---- step 4: liquidity filter + write -----------------------------------
def apply_liquidity_and_write(
    trades: pd.DataFrame, markets: pd.DataFrame, out_path: Path, floor: int, label: str
) -> pd.DataFrame:
    if trades.empty:
        print(f"[{label}] no trades to write")
        return pd.DataFrame()
    counts = trades.groupby("market_id").size()
    keep_ids = counts[counts >= floor].index
    dropped_ids = counts[counts < floor].index
    print(f"[{label}] markets with >= {floor} pre-event trades: {len(keep_ids)}")
    print(f"[{label}] markets dropped below floor: {len(dropped_ids)}")
    trades_out = trades[trades["market_id"].isin(keep_ids)].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades_out.to_parquet(out_path, index=False)
    print(f"[{label}] wrote {len(trades_out):,} trades to {out_path}")
    return trades_out


# ---- step 5: build inventory doc -----------------------------------------
def build_inventory(
    audit: pd.DataFrame,
    train_markets: pd.DataFrame,
    test_markets: pd.DataFrame,
    train_trades: pd.DataFrame,
    test_trades: pd.DataFrame,
    train_trades_all: pd.DataFrame,
    test_trades_all: pd.DataFrame,
    final_train_floor: int,
    final_test_floor: int,
) -> str:
    def market_row(
        mid: str,
        markets: pd.DataFrame,
        trades_all: pd.DataFrame,
        trades_kept: pd.DataFrame,
    ):
        mrow = markets[markets["id"] == mid]
        if mrow.empty:
            return None
        r = mrow.iloc[0]
        sub_all = trades_all[trades_all["market_id"] == mid]
        sub_keep = trades_kept[trades_kept["market_id"] == mid]
        total = len(sub_all)
        pre = len(sub_keep)
        first_ts = last_ts = ""
        if pre > 0:
            first_ts = pd.to_datetime(
                sub_keep["timestamp"].min(), unit="s", utc=True
            ).isoformat()
            last_ts = pd.to_datetime(
                sub_keep["timestamp"].max(), unit="s", utc=True
            ).isoformat()
        end = r["end_date"]
        end_s = end.isoformat() if hasattr(end, "isoformat") else str(end)
        return {
            "market_id": mid,
            "question": r["question"],
            "deadline": end_s,
            "total_trades_in_window": total,
            "pre_event_trades": pre,
            "first_pre_event_ts": first_ts,
            "last_pre_event_ts": last_ts,
        }

    def cohort_table(markets, trades_all, trades_kept):
        rows = []
        for mid in sorted(markets["id"].tolist()):
            row = market_row(mid, markets, trades_all, trades_kept)
            if row:
                rows.append(row)
        return pd.DataFrame(rows)

    train_tbl = cohort_table(train_markets, train_trades_all, train_trades)
    test_tbl = cohort_table(test_markets, test_trades_all, test_trades)

    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_(none)_"
        return df.to_markdown(index=False)

    # sanity checks
    train_mids = (
        set(train_trades["market_id"].unique()) if not train_trades.empty else set()
    )
    test_mids = (
        set(test_trades["market_id"].unique()) if not test_trades.empty else set()
    )
    overlap_markets = train_mids & test_mids

    if not train_trades.empty and not test_trades.empty:
        train_ts_max = pd.to_datetime(
            train_trades["timestamp"].max(), unit="s", utc=True
        )
        test_ts_min = pd.to_datetime(test_trades["timestamp"].min(), unit="s", utc=True)
        timestamp_overlap = train_ts_max > test_ts_min
    else:
        train_ts_max = test_ts_min = None
        timestamp_overlap = False

    lines = []
    lines.append("# Cohort inventory\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_\n")

    lines.append("## 1. HF dataset access notes\n")
    lines.append(
        "Source: `SII-WANGZJ/Polymarket_data` (HuggingFace dataset).\n\nFile layout:\n"
    )
    lines.append(
        "| File | Size | Records | Notes |\n|---|---|---|---|\n"
        "| `trades.parquet` | 38 GB | 418.3M | Processed trades with market linkage — used here |\n"
        "| `markets.parquet` | ~120 MB | 734,790 | Market metadata — cached locally under `alex/.scratch/markets.parquet` |\n"
        "| `orderfilled_part[1-4].parquet` | 84 GB total | 689M | Raw blockchain events, not used |\n"
        "| `quant.parquet` | 28 GB | 418M | YES-normalised trades, not used |\n"
        "| `users.parquet` | 23 GB | 341M | Maker/taker split, not used |\n\n"
    )
    lines.append(
        "Trade schema columns: `timestamp (uint64 epoch s)`, `block_number`, `transaction_hash`, `log_index`, `contract`, `market_id`, `condition_id`, `event_id`, `maker`, `taker`, `price (float, 0-1)`, `usd_amount`, `token_amount`, `maker_direction`, `taker_direction`, `nonusdc_side` (`token1`=YES / `token2`=NO), `asset_id`.\n\n"
    )
    lines.append(
        "Market schema columns: `id` (join key with `market_id`), `question`, `slug`, `condition_id`, `token1`, `token2`, `answer1`, `answer2`, `closed`, `active`, `archived`, `outcome_prices` (JSON string), `volume`, `event_id`, `event_slug`, `event_title`, `created_at`, `end_date`, `updated_at`, `neg_risk`.\n\n"
    )

    lines.append("### Reusable query snippet (10 lines)\n")
    lines.append("```python\n")
    lines.append(
        "import pyarrow.parquet as pq\nfrom huggingface_hub import HfFileSystem\nfs = HfFileSystem()\nTARGET_IDS = {'1198423', '1386664'}  # market_ids of interest\nwith fs.open('datasets/SII-WANGZJ/Polymarket_data/trades.parquet','rb') as f:\n    pf = pq.ParquetFile(f)\n    for rg in range(pf.num_row_groups):\n        tab = pf.read_row_group(rg, columns=pf.schema_arrow.names)\n        df = tab.to_pandas()\n        hit = df[df.market_id.isin(TARGET_IDS)]\n        if len(hit): print(rg, len(hit))\n"
    )
    lines.append("```\n\n")

    lines.append("## 2. Event definitions\n")
    lines.append(
        "- **Train event**: US strike on Iran, _Operation Epic Fury_ — CENTCOM announcement ~06:35 UTC on 2026-02-28. Cutoff used: `timestamp < 2026-02-28 06:35 UTC` (unix epoch < 1772260500). Source: brief provided by thesis supervisor.\n"
    )
    lines.append(
        '- **Test event**: Trump announces Iran ceasefire on 2026-04-07. Exact announcement time unknown. Cutoff used: `timestamp < 2026-04-07 00:00 UTC` (unix epoch < 1775606400). This drops the entire announcement day — chosen over a permissive same-day cutoff because the brief explicitly requested "better to over-filter than leak."\n\n'
    )

    lines.append(
        "## 3. Keyword-match dump (broad filter — every hit, with include/exclude decision)\n"
    )
    lines.append("_Broad filters used:_\n")
    lines.append(
        "- **Train (broad)**: title contains `(us|u.s.|america|united states|trump)` AND `(strike|attack|bomb|military action)` AND `iran`.\n"
    )
    lines.append(
        '- **Test (broad)**: I ran the brief\'s strict filter (`trump` AND `announce*` AND `ceasefire` AND `iran`) and it returned **0 markets** on Polymarket (no market literally asks "Will Trump announce Iran ceasefire by X"). I also audited the superset `iran` AND `ceasefire` (30 markets) since that is the actual canonical ladder family on Polymarket. See §11 for the autonomous decision.\n\n'
    )
    lines.append(
        md_table(audit[["cohort", "id", "question", "end_date", "volume", "decision"]])
    )
    lines.append("\n\n")

    lines.append(
        f"## 4. Train cohort (canonical strike ladder) — {len(train_tbl)} markets\n"
    )
    lines.append(md_table(train_tbl))
    lines.append("\n\n")

    lines.append(
        f"## 5. Test cohort (canonical ceasefire ladder) — {len(test_tbl)} markets\n"
    )
    lines.append(md_table(test_tbl))
    lines.append("\n\n")

    lines.append("## 6. Summary stats\n")

    def summary_line(label, trades, markets):
        if trades is None or trades.empty:
            return f"- **{label}**: 0 markets, 0 trades."
        ts_min = pd.to_datetime(trades["timestamp"].min(), unit="s", utc=True)
        ts_max = pd.to_datetime(trades["timestamp"].max(), unit="s", utc=True)
        return f"- **{label}**: {trades['market_id'].nunique()} markets, {len(trades):,} pre-event trades, trade ts range {ts_min.isoformat()} → {ts_max.isoformat()}."

    lines.append(summary_line("Train", train_trades, train_markets) + "\n")
    lines.append(summary_line("Test", test_trades, test_markets) + "\n\n")

    lines.append(
        f"Liquidity floors applied: train = {final_train_floor}, test = {final_test_floor}.\n\n"
    )

    lines.append("## 7. Sanity checks\n")
    lines.append(f"- Timestamp overlap (train max > test min)? **{timestamp_overlap}**")
    if train_ts_max is not None:
        lines.append(
            f" (train max = {train_ts_max.isoformat()}, test min = {test_ts_min.isoformat()})"
        )
    lines.append("\n")
    lines.append(
        f"- Markets appearing in both cohorts: **{len(overlap_markets)}**"
        + (f" ({overlap_markets})" if overlap_markets else "")
        + "\n"
    )
    lines.append(
        f"- All train trades before {TRAIN_CUTOFF_UTC.isoformat()}: **{(train_trades['timestamp'].max() < TRAIN_CUTOFF_UTC.timestamp()) if not train_trades.empty else 'n/a'}**\n"
    )
    lines.append(
        f"- All test trades before {TEST_CUTOFF_UTC.isoformat()}: **{(test_trades['timestamp'].max() < TEST_CUTOFF_UTC.timestamp()) if not test_trades.empty else 'n/a'}**\n\n"
    )

    lines.append("## 8. Fields the HF dataset does NOT contain\n")
    lines.append(
        "- **No `resolved` / `winning_outcome` field on trades.** Market resolution must be inferred from `markets.outcome_prices` (string JSON like `\"['0.99','0.01']\"`) — answer1 wins iff first value > 0.5.\n"
    )
    lines.append(
        "- **No trade-level side indicator normalised to BUY/SELL on YES.** `trades.parquet` exposes `maker_direction`/`taker_direction` (BUY/SELL from each party's perspective) and `nonusdc_side` (token1/token2). Normalise separately; `quant.parquet` does this already.\n"
    )
    lines.append(
        "- **No `proxy_wallet` field.** The HF schema has `maker` / `taker` wallets. The shared pipeline's `proxyWallet` comes from Polymarket's DataAPI and is **not** in this dataset.\n"
    )
    lines.append(
        "- **No market title on trades.** Must be joined from `markets.parquet` via `market_id` ↔ `id`.\n"
    )
    lines.append(
        "- **No explicit `side = YES/NO`.** Derive from `nonusdc_side` + `taker_direction`.\n"
    )
    lines.append(
        "- **No volume column on trades** (trades have `usd_amount` per trade; market-level `volume` lives on `markets.parquet`).\n"
    )
    lines.append(
        "- **`outcome_prices` is a stringified list**, not an array. Parse with `ast.literal_eval`.\n"
    )
    lines.append(
        "- **`created_at` / `end_date` on markets are millisecond-precision datetime64[ms, UTC].** Trade `timestamp` is uint64 epoch-seconds.\n\n"
    )

    lines.append("## 9. Build script reproducibility\n")
    lines.append("```bash\n")
    lines.append("cd /Users/alex/Documents/Claude\\ Projects/CBS-Spring26/ML/report\n")
    lines.append("python -m venv .venv && source .venv/bin/activate\n")
    lines.append("pip install huggingface_hub pyarrow pandas tqdm\n")
    lines.append("python alex/scripts/build_cohorts.py\n")
    lines.append("```\n\n")
    lines.append("Outputs on success:\n")
    lines.append(
        "- `data/archive/alex/train.parquet`\n- `data/archive/alex/test.parquet`\n- `data/archive/alex/markets_subset.parquet` (metadata for included markets)\n- `alex/outputs/cohort_inventory/keyword_matches.csv`\n- `alex/notes/cohort_inventory.md` (this file)\n\n"
    )

    lines.append("## 10. Autonomous decisions requiring user audit\n")
    lines.append(
        "1. **Test ladder interpretation.** The brief asked for the \"Will Trump announce Iran ceasefire by [date]\" family. Zero Polymarket markets match this literal phrasing (with or without Trump in the title). The canonical ceasefire ladder that was active between Feb 28 and Trump's Apr 7 announcement is **\"US x Iran ceasefire by [date]\"** (10 markets, created 2026-02-28 → 2026-03-24). These markets resolve on Trump/administration announcement of a US-Iran ceasefire — they are substantively the same event family. **I included the 'US x Iran ceasefire' ladder as the test cohort.** Alternative routes the user may prefer: (a) treat test as empty and document the gap, (b) broaden to include 'Israel x Iran ceasefire' markets (I excluded these — different parties, Jun-Aug 2025 ladder around a different ceasefire event).\n"
    )
    lines.append(
        '2. **Ladder narrowness.** On the train side I excluded the parallel "Will the US **next** strike Iran on [date] (ET)?" ladder (34 markets, created Jan-Feb 2026). Rationale: different resolution mechanic (single-day strike windows vs deadline-style cumulative) and the brief says "canonical question family only". The user may prefer to include them as the same ladder — easy swap (change `TRAIN_CANONICAL_RE`).\n'
    )
    lines.append(
        "3. **`condition_id` duplicates.** `US strikes Iran by March 14, 2026?` appears twice (ids 1437735 + 1437737, the first has zero volume, likely a re-list). Both are kept if they pass the liquidity floor. Filter if the user wants one canonical per deadline.\n"
    )
    lines.append(
        "4. **Cache.** `markets.parquet` is cached under `alex/.scratch/markets.parquet` (~120MB). Delete to re-fetch from HF.\n"
    )
    return "\n".join(lines)


# ---- main ----------------------------------------------------------------
def main():
    print("=" * 60)
    print("cohort build start")
    print(f"train cutoff: {TRAIN_CUTOFF_UTC.isoformat()}")
    print(f"test cutoff:  {TEST_CUTOFF_UTC.isoformat()}")
    print("=" * 60)

    m = load_markets()
    print(f"[markets] total markets: {len(m):,}")

    train_markets, test_markets, audit = pick_markets(m)
    print(f"[pick] train canonical markets: {len(train_markets)}")
    print(f"[pick] test canonical markets: {len(test_markets)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = OUT_DIR / "keyword_matches.csv"
    audit.to_csv(audit_path, index=False)
    print(f"[pick] audit dump -> {audit_path}")

    # Save market subset
    included = pd.concat(
        [
            train_markets.assign(cohort="train"),
            test_markets.assign(cohort="test"),
        ],
        ignore_index=True,
    )
    included.to_parquet(DATA_DIR / "markets_subset.parquet", index=False)

    # Single streaming pass: read trades for union of market ids + combined window
    target_ids = set(train_markets["id"].tolist()) | set(test_markets["id"].tolist())
    print(f"[stream] target market_ids: {len(target_ids)}")

    ts_min = min(train_markets["created_at"].min(), test_markets["created_at"].min())
    ts_max = TEST_CUTOFF_UTC  # widest cutoff
    print(f"[stream] timestamp range: {ts_min.isoformat()} → {ts_max.isoformat()}")

    trades_all = stream_trades(target_ids, ts_min.to_pydatetime(), ts_max)
    print(
        f"[stream] total trades fetched for target markets (pre-cutoff): {len(trades_all):,}"
    )

    # Split by cohort + apply train/test cutoff + liquidity floor
    train_ids = set(train_markets["id"].tolist())
    test_ids = set(test_markets["id"].tolist())

    train_trades_all = trades_all[
        trades_all["market_id"].isin(train_ids)
        & (trades_all["timestamp"] < TRAIN_CUTOFF_UTC.timestamp())
    ].copy()
    test_trades_all = trades_all[
        trades_all["market_id"].isin(test_ids)
        & (trades_all["timestamp"] < TEST_CUTOFF_UTC.timestamp())
    ].copy()

    train_trades = apply_liquidity_and_write(
        train_trades_all,
        train_markets,
        DATA_DIR / "train.parquet",
        LIQUIDITY_FLOOR,
        "train",
    )

    test_trades = apply_liquidity_and_write(
        test_trades_all,
        test_markets,
        DATA_DIR / "test.parquet",
        LIQUIDITY_FLOOR,
        "test",
    )
    final_test_floor = LIQUIDITY_FLOOR
    if test_trades.empty or test_trades["market_id"].nunique() == 0:
        print("[test] primary liquidity floor yielded 0 markets; falling back to 100")
        test_trades = apply_liquidity_and_write(
            test_trades_all,
            test_markets,
            DATA_DIR / "test.parquet",
            LIQUIDITY_FLOOR_FALLBACK,
            "test",
        )
        final_test_floor = LIQUIDITY_FLOOR_FALLBACK

    # Inventory
    inv = build_inventory(
        audit,
        train_markets,
        test_markets,
        train_trades,
        test_trades,
        train_trades_all,
        test_trades_all,
        LIQUIDITY_FLOOR,
        final_test_floor,
    )
    inv_path = NOTES_DIR / "cohort_inventory.md"
    inv_path.write_text(inv)
    print(f"[inventory] wrote {inv_path}")

    print("=" * 60)
    print("DONE")
    print(
        f"  train: {len(train_trades):,} trades, {train_trades['market_id'].nunique() if not train_trades.empty else 0} markets -> {DATA_DIR / 'train.parquet'}"
    )
    print(
        f"  test:  {len(test_trades):,} trades, {test_trades['market_id'].nunique() if not test_trades.empty else 0} markets -> {DATA_DIR / 'test.parquet'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
