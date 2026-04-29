"""
14_build_experiment_splits.py
Slice the consolidated dataset into small, self-contained parquet files for
the market-cohort split defined in project_plan.md §4 → Split.

Each cohort is written to `outputs/experiments/<name>.parquet`, fully
feature-complete, so the modelling script can load a cohort in <1 s without
re-reading the ~1 GB CSV every time.

NOTE (2026-04-29): superseded by the team-shared
`data/consolidated_modeling_data.parquet`, which already carries train/test in
a `split` column. This script is kept as historical pipeline; if re-run it
reads the legacy CSV from `data/archive/pipeline/`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "archive" / "pipeline" / "03_consolidated_dataset.csv"
OUT_DIR = ROOT / "outputs" / "experiments"

# Pre-modelling filter per §4: drop post-resolution close-out trades.
APPLY_SETTLEMENT_FILTER = True


# Market-cohort definitions — matched on the `question` column, which is
# populated and unique per condition_id across the 74-market frame.
COHORTS: dict[str, list[str]] = {
    "train": [
        "US strikes Iran by February 25, 2026?",
        "US strikes Iran by February 26, 2026?",
        "US strikes Iran by February 27, 2026?",
        "US strikes Iran by February 28, 2026?",
    ],
    "val": [
        # Post-training NO market, cross-event-family (event 236884).
        # Replaces earlier Jan 23 strike NO which was temporally BEFORE training
        # (violating train→val→test chronology). Mar 15 conflict-end is
        # post-Feb-28, not strike-family (no arbitrage-leakage channel),
        # and sized for stable early-stopping signal (~20k post-filter).
        "Iran x Israel/US conflict ends by March 15?",
    ],
    "test": [
        # All 7 ceasefire markets (events 355299 + 357625). Cross-event-family
        # from the strike-market training cohort — no price-arbitrage leakage
        # channel. All resolve NO; YES hedge is a deferred future addition
        # (see alex/notes/test-cohort-no-bias.md).
        "Trump announces US x Iran ceasefire end by April 8, 2026?",
        "Trump announces US x Iran ceasefire end by April 10, 2026?",
        "Trump announces US x Iran ceasefire end by April 12, 2026?",
        "Trump announces US x Iran ceasefire end by April 15, 2026?",
        "Trump announces US x Iran ceasefire end by April 18, 2026?",
        "Will the US x Iran ceasefire be extended by April 14, 2026?",
        "Will the US x Iran ceasefire be extended by April 18, 2026?",
    ],
}


def load_frame() -> pd.DataFrame:
    if not CSV.exists():
        raise SystemExit(f"missing: {CSV}")
    print(f"[load] {CSV} ({CSV.stat().st_size / 1e6:,.0f} MB)")
    df = pd.read_csv(CSV, low_memory=False)
    print(f"[load] {len(df):,} rows x {df.shape[1]} cols")
    return df


def slice_cohort(df: pd.DataFrame, name: str, questions: list[str]) -> pd.DataFrame:
    sel = df[df["question"].isin(questions)].copy()
    if len(sel) == 0:
        raise SystemExit(
            f"cohort '{name}' matched 0 rows. Check question spellings:\n  "
            + "\n  ".join(questions)
        )

    matched = set(sel["question"].unique())
    missing = [q for q in questions if q not in matched]
    if missing:
        print(f"[warn] cohort '{name}' missing {len(missing)} question(s):")
        for q in missing:
            print(f"         {q}")

    if APPLY_SETTLEMENT_FILTER:
        pre = len(sel)
        sel = sel[sel["settlement_minus_trade_sec"] > 0].copy()
        dropped = pre - len(sel)
        pct = 100.0 * dropped / pre if pre else 0.0
        print(
            f"[{name}] post-settlement filter: dropped {dropped:,} / {pre:,} ({pct:.1f}%)"
        )

    # Concise per-market breakdown
    summary = (
        sel.groupby("question")
        .agg(
            trades=("question", "size"),
            is_yes=("is_yes", "first"),
            bc_rate=("bet_correct", "mean"),
        )
        .sort_values("trades", ascending=False)
    )
    print(
        f"[{name}] {len(sel):,} rows x {sel.shape[1]} cols across {len(summary)} markets"
    )
    print(f"[{name}] per-market breakdown:")
    for q, row in summary.iterrows():
        y = "YES" if int(row["is_yes"]) == 1 else "NO"
        print(f"         [{int(row['trades']):>7,}]  {y}  bc={row['bc_rate']:.3f}  {q}")

    return sel


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame()

    if "is_yes" not in df.columns:
        raise SystemExit(
            "'is_yes' column missing. Run scripts/15_backfill_is_yes.py first, "
            "or rebuild the dataset with the patched 01_polymarket_api.py."
        )

    for name, questions in COHORTS.items():
        if not questions:
            print(f"[skip] cohort '{name}' has no questions defined")
            continue
        cohort = slice_cohort(df, name, questions)
        out = OUT_DIR / f"{name}.parquet"
        cohort.to_parquet(out, index=False)
        print(f"[{name}] wrote {out} ({out.stat().st_size / 1e6:,.1f} MB)\n")


if __name__ == "__main__":
    main()
