"""Physical-drop step: produce the final modelling-ready consolidated CSV.

Removes four classes of columns from `data/03_consolidated_dataset.csv`:

  * **Tier 1 — confirmed leak / bug.** `wallet_prior_win_rate` (future-info
    peek, replaced by `wallet_prior_win_rate_causal` — P0-9);
    `is_position_exit` and `is_position_flip` (denominator bug fires on
    first SELL — P0-2); `wallet_funded_by_cex` (unscoped lifetime flag,
    structurally leaky even though 0 rows leak empirically in this build —
    `wallet_funded_by_cex_scoped` is retained).

  * **Tier 2 — market-identity memorisation (P0-8).** Seven absolute-scale
    features that are strictly causal but scale with the market's deadline
    or cumulative activity, letting a classifier memorise which sub-market
    a trade belongs to and shortcut generalisation:
    `time_to_settlement_s`, `log_time_to_settlement`,
    `market_volume_so_far_usd`, `market_vol_1h_log`, `market_vol_24h_log`,
    `market_trade_count_so_far`, `size_x_time_to_settlement`. Bounded /
    market-normalised substitutes are retained: `pct_time_elapsed`,
    `market_buy_share_running`, `market_price_vol_last_1h`.

  * **Tier 3 — metadata bloat.** Polymarket API join fields that are never
    features and duplicate information already captured elsewhere:
    `conditionId, title, slug_x, slug_y, icon, eventSlug, outcome, name,
    pseudonym, bio, profileImage, profileImageOptimized, outcomes`.

  * **Tier 4 — obsolete.** `split` (legacy trade-timestamp quantile column,
    replaced by the market-cohort split in `data/experiments/`).

Result: a single `03_consolidated_dataset.csv` containing only (a) columns
used as model features, (b) columns needed to build the cohort split and
the post-resolution filter, and (c) labels / IDs.

The pre-drop state is preserved at
`data/03_consolidated_dataset.pre_dropped_variables.csv` (produced before
this script runs; recreated if missing).

Idempotent. Safe to re-run.

Usage:
  python scripts/20_finalize_dataset.py
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "03_consolidated_dataset.csv"
BACKUP = ROOT / "data" / "03_consolidated_dataset.pre_dropped_variables.csv"

TIER1_LEAK_OR_BUG = [
    "wallet_prior_win_rate",        # leak — replaced by wallet_prior_win_rate_causal
    "is_position_exit",             # denominator bug
    "is_position_flip",             # same signed-size family as is_position_exit
    "wallet_funded_by_cex",         # structurally leaky; keep scoped variant
]

TIER2_MARKET_IDENTITY = [
    "time_to_settlement_s",
    "log_time_to_settlement",
    "market_volume_so_far_usd",
    "market_vol_1h_log",
    "market_vol_24h_log",
    "market_trade_count_so_far",
    "size_x_time_to_settlement",
]

TIER3_METADATA_BLOAT = [
    "conditionId",
    "title",
    "slug_x",
    "slug_y",
    "icon",
    "eventSlug",
    "outcome",
    "name",
    "pseudonym",
    "bio",
    "profileImage",
    "profileImageOptimized",
    "outcomes",
]

TIER4_OBSOLETE = [
    "split",
]

ALL_DROPS = TIER1_LEAK_OR_BUG + TIER2_MARKET_IDENTITY + TIER3_METADATA_BLOAT + TIER4_OBSOLETE


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}")

    t0 = time.monotonic()
    print(f"reading {CSV.name}...")
    df = pd.read_csv(CSV, low_memory=False)
    print(
        f"  {len(df):,} rows × {len(df.columns)} cols "
        f"({time.monotonic() - t0:.1f}s)"
    )

    if not BACKUP.exists():
        print(f"writing backup {BACKUP.name} (pre-drop snapshot)...")
        shutil.copy2(CSV, BACKUP)
        print(f"  backup written ({time.monotonic() - t0:.1f}s)")
    else:
        print(f"  backup {BACKUP.name} already present — not overwriting")

    present = [c for c in ALL_DROPS if c in df.columns]
    absent = [c for c in ALL_DROPS if c not in df.columns]
    print(
        f"\ndrop plan: {len(ALL_DROPS)} columns total   "
        f"present: {len(present)}   absent: {len(absent)}"
    )
    if absent:
        print(f"  already absent (will skip): {absent}")

    if not present:
        print("nothing to drop. exiting.")
        return

    print("\ncolumns being dropped:")
    for tier_name, tier_cols in [
        ("Tier 1 — leak / bug", TIER1_LEAK_OR_BUG),
        ("Tier 2 — market-identity", TIER2_MARKET_IDENTITY),
        ("Tier 3 — metadata bloat", TIER3_METADATA_BLOAT),
        ("Tier 4 — obsolete", TIER4_OBSOLETE),
    ]:
        hit = [c for c in tier_cols if c in df.columns]
        print(f"  {tier_name} ({len(hit)}):")
        for c in hit:
            print(f"    - {c}")

    df = df.drop(columns=present)
    print(f"\nfinal shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    print("writing CSV...")
    t1 = time.monotonic()
    df.to_csv(CSV, index=False)
    print(
        f"  wrote {CSV.name} in {time.monotonic() - t1:.1f}s "
        f"(size {CSV.stat().st_size / 1e6:,.0f} MB; pre-drop was "
        f"{BACKUP.stat().st_size / 1e6:,.0f} MB)"
    )
    print(f"total: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
