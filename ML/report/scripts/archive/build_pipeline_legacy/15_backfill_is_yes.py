"""
15_backfill_is_yes.py
One-shot backfill: adds the `outcomes` (per-market outcome array) and `is_yes`
(YES/NO label) columns to `data/03_consolidated_dataset.csv`.

Why this exists:
The current CSV was built before the `01_polymarket_api.py:enrich_trades` fix
that preserves `outcomes` through the meta-join. This script backfills the two
columns without re-running the whole pipeline, using the existing
`01_markets_meta.csv` (written by `02_build_dataset.py`, already carries the
`outcomes` array) as the source.

After the upstream fix lands, a full rebuild will populate these columns
natively and this script becomes unnecessary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "03_consolidated_dataset.csv"
META = ROOT / "data" / "01_markets_meta.csv"
BACKUP = ROOT / "data" / "03_consolidated_dataset.pre_is_yes.csv"


def derive_is_yes(row) -> int | None:
    outs = str(row.get("outcomes") or "").split(";")
    wi = row.get("winning_outcome_index")
    if pd.isna(wi) or int(wi) >= len(outs):
        return None
    return 1 if str(outs[int(wi)]).strip().lower() in {"yes", "true"} else 0


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing: {CSV}")
    if not META.exists():
        raise SystemExit(
            f"missing: {META}\n"
            "This script needs 01_markets_meta.csv (ask Pontus or regenerate via 02)."
        )

    print(f"[load] {CSV} ({CSV.stat().st_size / 1e6:.1f} MB)")
    df = pd.read_csv(CSV, low_memory=False)
    print(f"[load] {len(df):,} rows x {df.shape[1]} cols")

    if "is_yes" in df.columns and "outcomes" in df.columns:
        print("[skip] both columns already present; nothing to do")
        return

    print(f"[load] {META}")
    meta = pd.read_csv(META, low_memory=False)
    if "outcomes" not in meta.columns:
        raise SystemExit("01_markets_meta.csv missing `outcomes` column")
    if "condition_id" not in meta.columns:
        raise SystemExit("01_markets_meta.csv missing `condition_id` column")

    # Sanity-check: meta's winning_outcome_index should match the CSV's for
    # every market. Flag anything weird but don't abort — the CSV is still
    # internally consistent (verified separately).
    if "winning_outcome_index" in meta.columns:
        stored = (
            df.groupby("condition_id")["winning_outcome_index"].first().reset_index()
        )
        cmp = stored.merge(
            meta[["condition_id", "winning_outcome_index"]].rename(
                columns={"winning_outcome_index": "wi_meta"}
            ),
            on="condition_id",
            how="left",
        )
        mask = cmp["wi_meta"].notna() & cmp["winning_outcome_index"].notna()
        mismatches = int(
            (
                cmp.loc[mask, "wi_meta"].astype(int)
                != cmp.loc[mask, "winning_outcome_index"].astype(int)
            ).sum()
        )
        print(f"[check] winning_outcome_index mismatches CSV vs meta: {mismatches}")

    meta_slim = meta[["condition_id", "outcomes"]].drop_duplicates("condition_id")
    print(f"[meta] {len(meta_slim)} unique markets")

    print(f"[backup] writing {BACKUP.name}")
    df.to_csv(BACKUP, index=False)

    print("[merge] joining outcomes on condition_id")
    df = df.merge(meta_slim, on="condition_id", how="left")
    unmatched_rows = int(df["outcomes"].isna().sum())
    if unmatched_rows:
        unmatched_markets = df.loc[df["outcomes"].isna(), "condition_id"].nunique()
        print(
            f"[warn] {unmatched_rows:,} rows / {unmatched_markets} markets "
            "have no outcomes match (should be zero)"
        )

    print("[derive] computing is_yes")
    df["is_yes"] = df.apply(derive_is_yes, axis=1).astype("Int64")

    per_market = df.groupby("condition_id")["is_yes"].first()
    n_yes = int((per_market == 1).sum())
    n_no = int((per_market == 0).sum())
    n_na = int(per_market.isna().sum())
    print(f"[result] per-market is_yes breakdown: YES={n_yes}  NO={n_no}  NA={n_na}")

    print(f"[write] {CSV}")
    df.to_csv(CSV, index=False)
    print(f"[done] {len(df):,} rows x {df.shape[1]} cols")


if __name__ == "__main__":
    main()
